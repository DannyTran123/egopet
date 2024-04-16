# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import random
import math

import torch
import numpy as np
import torch.utils.data
from iopath.common.file_io import g_pathmgr as pathmgr

import sys
from decoder.utils import decode_ffmpeg
from torchvision import transforms
from torch.nn.functional import normalize

### evo evaluation library ###
import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync, metrics
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation
from evo.tools import plot
from dpvo.plot_utils import plot_trajectory, save_trajectory_tum_format

import sys
parent = os.path.dirname(os.path.abspath(__file__))
parent_parent = os.path.join(parent, '../')
sys.path.append(os.path.dirname(parent_parent))

from dpvo.plot_utils import *
from pathlib import Path


def get_poses(trajectory_path, start_idx, end_idx):
    """
    Gets the camera poses for the frames start_idx to end_idx.
    Args:
        trajectory_path (string): the path to the file containing the trajectory
        in TUM format. 
        start_idx (int): the start index to get poses
        end_idx (int): the end index to get poses
    Returns:
        traj_poses (tensor): The poses (end_idx - start_idx) x 7
    """
    traj = file_interface.read_tum_trajectory_file(trajectory_path)
    traj_poses = np.concatenate((traj.positions_xyz, traj.orientations_quat_wxyz[:, [1, 2, 3, 0]]), axis=1)
    traj_poses = torch.from_numpy(traj_poses)
    
    return traj_poses[start_idx:end_idx]


def relative_poses(trajectory_path, start_idx, end_idx, scale_invariance, pose_skip):
    """
    Gets dP, the relative pose to the prior frame, between camera poses for the
    frames start_idx to end_idx. 
    Args:
        trajectory_path (string): the path to the file containing the trajectory
        in TUM format. 
        start_idx (int): the start index to get relative poses
        end_idx (int): the end index to get relative poses
        scale_invariance (str): scale invariance type ('dir' or None)
        pose_skip (int): stride of pose prediction
    Returns:
        dP_tensor (tensor): the dPs . The dimension
                            is (end_idx - start_idx) x 7.
        scale (tensor): if scale_invariance=='dir' return the scale of the
                        poses
    """
    traj = file_interface.read_tum_trajectory_file(trajectory_path)
    translation = traj.positions_xyz[1:] - traj.positions_xyz[:-1]
    
    dP_tensor = torch.from_numpy(translation)
    zero_tensor = torch.tensor([[0, 0, 0]]) # Hacky adding 0'th relative pose for indexing
    dP_tensor = torch.cat([zero_tensor, dP_tensor], dim=0)
    
    dP_tensor = dP_tensor[start_idx:end_idx]
    dP_tensor = dP_tensor.unflatten(0, (-1, pose_skip))
    dP_tensor = torch.sum(dP_tensor, dim=1)
    
    if scale_invariance == 'dir':
        scale = torch.norm(dP_tensor, p=2, dim=1, keepdim=True)
        dP_tensor = dP_tensor / (scale + 1e-10)
        return dP_tensor, scale
    
    return dP_tensor


def make_traj_from_tensor(traj_tensor, start_idx, end_idx, num_condition_frames, num_pose_prediction, act_pose_prediction, pose_skip):
    tstamps = np.arange(0, end_idx, 1, dtype=np.float64)
    tstamps = tstamps.reshape((-1, num_condition_frames + num_pose_prediction))[:, num_condition_frames+(pose_skip-1)::pose_skip]
    tstamps = tstamps.flatten()
    
    tstamps_idx = np.arange(0, len(traj_tensor), 1, dtype=np.float64)
    tstamps_idx = tstamps_idx.reshape((-1, num_condition_frames + act_pose_prediction))[:, num_condition_frames:]
    tstamps_idx = tstamps_idx.flatten()
    
    traj = PoseTrajectory3D(positions_xyz=traj_tensor[tstamps_idx.astype(int),:3], orientations_quat_wxyz=traj_tensor[tstamps_idx.astype(int),3:][:, [3, 0, 1, 2]], timestamps=tstamps)
    
    return traj


def eval_metrics(traj_ref, traj_pred):
    traj_ref, traj_pred = sync.associate_trajectories(traj_ref, traj_pred)
    
    result = main_ape.ape(traj_ref, traj_pred, est_name='traj',
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
    ate = result.stats['rmse']

    result = main_rpe.rpe(traj_ref, traj_pred, est_name='traj',
        pose_relation=PoseRelation.rotation_angle_deg, align=True, correct_scale=True,
        delta=1.0, delta_unit=metrics.Unit.frames, rel_delta_tol=0.1)
    rpe_rot = result.stats['rmse']

    result = main_rpe.rpe(traj_ref, traj_pred, est_name='traj',
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True,
        delta=1.0, delta_unit=metrics.Unit.frames, rel_delta_tol=0.1)
    rpe_trans = result.stats['rmse']

    return ate, rpe_trans, rpe_rot


def evaluate_segment(model, dataset_val, device, video_path, trajectory_path, criterion, start_idx, end_idx, args, save_plot=False, image_model=False):
    # Assume batch of 1 
    loss = 0
    total_idx = end_idx - start_idx

    frames_per_sub_clip = args.num_condition_frames + args.num_pose_prediction
    act_frames_per_sub_clip = args.num_condition_frames + args.act_pose_prediction
    num_sub_clips = total_idx // frames_per_sub_clip
    num_sub_batch = math.ceil(num_sub_clips / args.validation_batch_size)

    all_cond_poses = get_poses(trajectory_path, start_idx, end_idx).cpu()
    all_cond_poses = all_cond_poses.unflatten(0, (-1, frames_per_sub_clip))
    all_cond_poses = torch.cat((all_cond_poses[:, :args.num_condition_frames ,:], all_cond_poses[:, args.num_condition_frames+(args.pose_skip-1)::args.pose_skip]), 1)
    all_cond_poses = all_cond_poses.flatten(0, 1)
    traj_ref = make_traj_from_tensor(all_cond_poses, start_idx, end_idx, args.num_condition_frames, args.num_pose_prediction, args.act_pose_prediction, args.pose_skip)

    traj_pred_poses = all_cond_poses.clone()
    traj_pred_poses = traj_pred_poses.unflatten(0, (-1, act_frames_per_sub_clip))
    traj_pred_poses[:, args.num_condition_frames:] = 0
    traj_pred_poses = traj_pred_poses.flatten(0, 1)
    traj_pred_poses[:, 3:] = all_cond_poses[:, 3:]
    traj_pred_poses = traj_pred_poses.to(device=device, dtype=torch.float32)

    for i in range(num_sub_batch):
        sub_start_idx = i * args.validation_batch_size * frames_per_sub_clip
        sub_end_idx = min((i + 1) * args.validation_batch_size * frames_per_sub_clip, total_idx)
        
        frames, dP_tensor = dataset_val._get_sub_clip(video_path, trajectory_path, sub_start_idx, sub_end_idx)

        frames = frames.to(device, non_blocking=True)
        dP_tensor = dP_tensor.to(device= device, dtype=torch.float32, non_blocking=True)

        frames = frames.unflatten(0, (-1, frames_per_sub_clip))
        frames = frames[:, :args.num_condition_frames, :, :, :]
        frames = frames.permute(0, 2, 1, 3, 4)

        dP_tensor = dP_tensor.unflatten(0, (-1, frames_per_sub_clip))
        dP_tensor = dP_tensor[:, args.num_condition_frames:, :].unflatten(1, (-1, args.pose_skip))
        dP_tensor = torch.sum(dP_tensor, dim=2)
    
        if args.scale_invariance == 'dir':
            scale = torch.norm(dP_tensor, p=2, dim=2, keepdim=True)
            dP_tensor = dP_tensor / (scale + 1e-10)
        
        if image_model:
            frames = frames[:, :, -1, :, :]

        outputs = model(frames)
        dP_pred = outputs.unflatten(1, (args.act_pose_prediction, args.num_pred))
        
        loss += criterion(dP_pred, dP_tensor)
        
        if args.scale_invariance == 'dir':
            dP_pred_unnorm = dP_pred * scale
            
        i_start_idx = i*args.validation_batch_size*act_frames_per_sub_clip
        for j in range(dP_pred_unnorm.shape[0]):
            j_start_idx = i_start_idx + j * act_frames_per_sub_clip + args.num_condition_frames

            start_xyz = traj_pred_poses[j_start_idx - 1, :3].unsqueeze(0)

            pred_xyz = torch.cat([start_xyz, dP_pred_unnorm[j]], dim=0)
            pred_xyz = torch.cumsum(pred_xyz, dim=0)[1:]
            traj_pred_poses[j_start_idx:j_start_idx + args.act_pose_prediction, :3] = pred_xyz

    traj_pred = make_traj_from_tensor(traj_pred_poses.detach().cpu(), start_idx, end_idx, args.num_condition_frames, args.num_pose_prediction, args.act_pose_prediction, args.pose_skip)
    ate, rpe_trans, rpe_rot = eval_metrics(traj_ref, traj_pred)
    
    segment_id = video_path.split('/')[-1][:-4]
    if save_plot and args.output_dir:
        segment_id = video_path.split('/')[-1][:-4]
        title = 'Reconstruction: {}'.format(segment_id)
        filename = os.path.join(args.output_dir, 'traj_viz', '{}_reconstruction.png'.format(segment_id))
        plot_trajectory(traj_pred, gt_traj=traj_ref, title=title, filename=filename, align=False, correct_scale=True)
        
    return loss, ate, rpe_trans, rpe_rot
