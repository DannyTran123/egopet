import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax
from einops import rearrange
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F
import pandas as pd
from matplotlib import pyplot as plt
import os
from sklearn.metrics import explained_variance_score
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC


def train_class_batch(model, samples, interactions, object_labels, criterion_interactions, criterion_object_labels, args):
    outputs = model(samples)
    interactions_output = outputs[:, :1]
    object_labels_output = outputs[:, 1:]
    loss_interaction = criterion_interactions(interactions_output, interactions)
    loss_object_labels = criterion_object_labels(object_labels_output, object_labels).unsqueeze(1)
    loss = (loss_interaction + args.alpha * interactions * loss_object_labels).mean()
    return loss, interactions_output, object_labels_output


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn=None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    # print('before for loop')
    for data_iter_step, (samples, interactions, object_labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):       
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None and 'lr_scale' in param_group:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and 'weight_decay' in param_group and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        samples = samples.permute(0, 2, 1, 3, 4)
        interactions = interactions.to(device, non_blocking=True)
        object_labels = object_labels.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        criterion_interactions = torch.nn.BCEWithLogitsLoss(reduction='none')
        criterion_object_labels = torch.nn.CrossEntropyLoss(reduction='none')
        if loss_scaler is None:
            samples = samples.half()
            loss, interactions_output, object_labels_output = train_class_batch(
                model, samples, interactions, object_labels, criterion_interactions, criterion_object_labels, args)
        else:
            with torch.cuda.amp.autocast():
                loss, interactions_output, object_labels_output = train_class_batch(
                    model, samples, interactions, object_labels, criterion_interactions, criterion_object_labels, args)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(dataset_val, model, device, args):
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    criterion_interactions = torch.nn.BCEWithLogitsLoss(reduction='none')
    criterion_object_labels = torch.nn.CrossEntropyLoss(reduction='none')

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    
    interactions_output_prob_list = []
    interactions_list = []
    object_labels_output_prob_list = []
    object_labels_idx_list = []
    

    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0].to(device, non_blocking=True)
        samples = samples.permute(0, 2, 1, 3, 4)
        interactions = batch[1].to(device, non_blocking=True)
        object_labels = batch[2].to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(samples)          
            interactions_output = outputs[:, :1]
            object_labels_output = outputs[:, 1:]
            loss_interaction = criterion_interactions(interactions_output, interactions)
            loss_object_labels = criterion_object_labels(object_labels_output, object_labels).unsqueeze(1)   
            loss = (loss_interaction + args.alpha * interactions * loss_object_labels).mean()

        batch_size = samples.shape[0]
        interactions_output_prob = torch.sigmoid(interactions_output)
        interactions_output_prob_0 = 1 - interactions_output_prob
        interactions_output_prob_full = torch.cat([interactions_output_prob_0, interactions_output_prob], dim=1)
        acc1_interaction = accuracy(interactions_output_prob_full, interactions, topk=(1, ))[0]
        metric_logger.meters['acc1_interaction'].update(acc1_interaction.item(), n=batch_size)

        interactions_output_prob_list.append(interactions_output_prob.flatten())
        interactions_list.append(interactions.flatten())
        
        indices = torch.nonzero(interactions.squeeze(1)).squeeze(1)
        total_indices = len(indices)
        if total_indices > 0:
            # Compute accuracy for object interaction for only cases where there are interactions
            object_labels_output = object_labels_output[indices, :]
            object_labels_idx = object_labels[indices]
            
            num_classes = object_labels_output.shape[1]
            object_labels_full = torch.zeros([total_indices, num_classes])
            object_labels_full[torch.arange(total_indices), object_labels[indices]] = 1
            object_labels_full = object_labels_full.to(device, non_blocking=True)
            
            softmax = torch.nn.Softmax(dim=1)
            object_labels_output_prob = softmax(object_labels_output)
            acc1_object, acc3_object = accuracy(object_labels_output_prob, object_labels_idx, topk=(1, 3))
            metric_logger.meters['acc1_object'].update(acc1_object.item(), n=total_indices)
            metric_logger.meters['acc3_object'].update(acc3_object.item(), n=total_indices)
            
            object_labels_output_prob_list.append(object_labels_output_prob)
            object_labels_idx_list.append(object_labels_idx.flatten())
        
        metric_logger.update(loss=loss.item())
        
    all_interactions_output_prob = torch.cat(interactions_output_prob_list, dim=0)
    all_interactions = torch.cat(interactions_list, dim=0)
    all_object_labels_output_prob = torch.cat(object_labels_output_prob_list, dim=0)
    all_object_labels_idx = torch.cat(object_labels_idx_list, dim=0)
    num_classes = all_object_labels_output_prob.shape[-1]
    
    b_auroc = BinaryAUROC(thresholds=None)
    interaction_auroc = b_auroc(preds=all_interactions_output_prob, target=all_interactions)
    metric_logger.meters['auroc_interaction'].update(interaction_auroc.item(), n=len(all_interactions_output_prob))
    
    m_auroc = MulticlassAUROC(num_classes=num_classes, average="macro", thresholds=None)
    object_auroc = m_auroc(preds=all_object_labels_output_prob, target=all_object_labels_idx)
    metric_logger.meters['auroc_object'].update(object_auroc.item(), n=len(all_object_labels_output_prob))
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))
    print('* Acc@1_interaction {top1.global_avg:.3f} auroc_interaction {auroc.global_avg:.3f}'
          .format(top1=metric_logger.acc1_interaction, auroc=metric_logger.auroc_interaction))
    print('* Acc@1_object {top1.global_avg:.3f} Acc@3_object {top3.global_avg:.3f} auroc_object {auroc.global_avg:.3f}'
          .format(top1=metric_logger.acc1_object, top3=metric_logger.acc3_object, auroc=metric_logger.auroc_object))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i].replace('[', '').replace(']', ''), \
                                                str(output.data[i].cpu().numpy().tolist()), \
                                                str(int(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1, final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100, final_top5*100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]



class Normalization:
    def __init__(self, normalization_coeffs):
        self.normalization_coeffs = [c.astype(np.float32) for c in normalization_coeffs]

    def get_coeffs(self):
        return self.normalization_coeffs

    def normalize_inputs(self, prop):
        '''

        :param features:
        :return:
        '''
        prop = (prop - self.normalization_coeffs[0]) / (self.normalization_coeffs[1] + 1e-10)
        return prop

    def normalize_labels(self, labels):
        '''

        :param labels:
        :return:
        '''
        labels = (labels - self.normalization_coeffs[2]) / self.normalization_coeffs[3]
        return labels

    def unnormalize_inputs(self, prop):
        '''
        :param features:
        :return:
        '''
        prop = prop * self.normalization_coeffs[1] + self.normalization_coeffs[0]
        return prop

    def unnormalize_labels(self, labels):
        '''

        :param labels:
        :return:
        '''
        labels = labels * self.normalization_coeffs[3] + self.normalization_coeffs[2]
        return labels
    
@torch.no_grad()
def evaluate_trajectories(data_loader, norm, model, device, p):
    criterion = torch.nn.MSELoss(reduction='none')

    # switch to evaluation mode
    model.eval()
    val_loss = 0
    nmse_loss = 0
    eva_score = 0
    mse_loss = 0
    target_ls = []
    pred_ls = []

    for batch in data_loader:
        images = batch[1]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            curr_nmse_loss = criterion(output, target)

            nmse_loss += torch.mean(curr_nmse_loss).item()
            val_loss += torch.mean(curr_nmse_loss)

            # compute other metrics
            pred_latent = output.cpu().numpy()
            # pred_latent = norm.unnormalize_labels(pred_latent)
            # unnormalized score
            latent_loss = criterion(torch.tensor(pred_latent).to(target), target)
            mse_loss += torch.mean(latent_loss).item()
            eva_score += explained_variance_score(target.cpu(),
                                                    pred_latent,
                                                    multioutput ='uniform_average')
            
        pred_ls.append(pred_latent)
        target_ls.append(target.cpu().numpy())

    target_ls = np.vstack(target_ls)
    pred_ls = np.vstack(pred_ls)
    ts = data_loader.dataset.rollout_ts
    stacked_data = np.hstack((ts,target_ls, pred_ls))

    # Create plots
    output_dim = 10
    num_exps = data_loader.dataset.num_experiments
    columns = ['ts'] 
    for i in range(output_dim):
        columns.append(f'target_latent_{i}')
    for i in range(output_dim):
        columns.append(f'pred_latent_{i}')

    for k in range(num_exps):
        interval = slice(data_loader.dataset.rollout_boundaries[k],
                            data_loader.dataset.rollout_boundaries[k+1])
        pd_frame = pd.DataFrame(data=stacked_data[interval],
                                columns=columns)
        try:
            fig = plt.figure()
            for i in range(output_dim):
                fig = plt.figure()
                plt.plot(np.array(pd_frame['ts']), np.array(pd_frame[f'target_latent_{i}']), label = "Target")
                plt.plot(np.array(pd_frame['ts']), np.array(pd_frame[f'pred_latent_{i}']), label = "Prediction")
                plt.xlabel('Time')
                plt.legend()
                plt.savefig(os.path.join(p, '%s_%s.png'%(k, i)))
        except Exception as e:
            print("Could not draw figure", e)