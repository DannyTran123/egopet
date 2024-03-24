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

parent = os.path.dirname(os.path.abspath(__file__))
parent_parent = os.path.join(parent, '../')
sys.path.append(os.path.dirname(parent_parent))

from locomotion_prediction.locomotion_prediction_utils import *


def train_class_batch(model, samples, dP_tensor, criterion, args):
    outputs = model(samples)
    dP_pred = outputs.unflatten(1, (args.act_pose_prediction, args.num_pred))
        
    loss = criterion(dP_pred, dP_tensor)

    return loss, dP_pred


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

    for data_iter_step, (samples, dP_tensor, start_idx, end_idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
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
        dP_tensor = dP_tensor.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, dP_pred = train_class_batch(
                    model, samples, dP_tensor, criterion, args)
        else:
            with torch.cuda.amp.autocast():
                loss, dP_pred = train_class_batch(
                    model, samples, dP_tensor, criterion, args)

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
def validation_one_epoch(dataset_val, model, criterion, device, args):
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
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for (video_path, trajectory_path, start_idx, end_idx) in metric_logger.log_every(data_loader, 10, header):
        try:
            video_path, trajectory_path, start_idx, end_idx = video_path[0], trajectory_path[0], start_idx[0].item(), end_idx[0].item()
            with torch.cuda.amp.autocast():
                loss, ate, rpe_trans, rpe_rot = evaluate_segment(model, dataset_val, device, video_path, trajectory_path, criterion, start_idx, end_idx, args, save_plot=True)
            metric_logger.meters['loss'].update(loss.item(), n=1)
            metric_logger.meters['ate'].update(ate, n=1)
            metric_logger.meters['rpe_trans'].update(rpe_trans, n=1)
            metric_logger.meters['rpe_rot'].update(rpe_rot, n=1)
        except Exception as e:
            print('Evaluation Error: ', e)        
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.5f}'
          .format(losses=metric_logger.loss))
    print('* ate {ate.global_avg:.5f}'
          .format(ate=metric_logger.ate))
    print('* rpe_trans {rpe_trans.global_avg:.5f}'
          .format(rpe_trans=metric_logger.rpe_trans))
    print('* rpe_rot {rpe_rot.global_avg:.5f}'
          .format(rpe_rot=metric_logger.rpe_rot))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
