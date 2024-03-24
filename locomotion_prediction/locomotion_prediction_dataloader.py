# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import random
import math

import torch
from lietorch import SE3
import lietorch
import numpy as np
import torch.utils.data
from iopath.common.file_io import g_pathmgr as pathmgr

import sys
parent = os.path.dirname(os.path.abspath(__file__))
parent_parent = os.path.join(parent, '../')
sys.path.append(os.path.dirname(parent_parent))

from locomotion_prediction.locomotion_prediction_utils import *
from decoder.utils import decode_ffmpeg
from torchvision import transforms
from evo.tools import file_interface

from pathlib import Path


class Locomotion_Prediction_dataloader(torch.utils.data.Dataset):
    """
    Locomotion Prediction video loader. Construct the Locomotion Prediction video loader, 
    then sample clips from the videos. For training, a single clip is randomly sampled 
    from every video with normalization. For validation, the video path and trajectories are
    returned. 
    """

    def __init__(
        self,
        mode,
        path_to_data_dir, 
        path_to_trajectories_dir, 
        path_to_csv, 
        # transformation
        transform,
        # decoding settings
        fps=30,
        # frame aug settings
        crop_size=224,
        # pose estimation settings
        animals=['cat', 'dog'],
        num_condition_frames=16,
        num_pose_prediction=16, 
        scale_invariance='None',
        pps=30, 
        # other parameters
        enable_multi_thread_decode=False,
        use_offset_sampling=True,
        inverse_uniform_sampling=False,
        num_retries=10,
    ):
        """
        Construct the Object Interaction video loader with a given csv file. The format of
        the csv file is:
        ```
        animal_1, ds_type_1, segment_id_1, stride_1, start_time_1, end_time_1, video_path_1
        animal_2, ds_type_2, segment_id_2, stride_2, start_time_2, end_time_2, video_path_2
        ...
        animal_N, ds_type_N, segment_id_N, stride_N, start_time_N, end_time_N, video_path_N
        ```
        Args:
            mode (string): Options includes `train` or `test`.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
            path_to_data_dir (string): Path to EgoPet Dataset
            path_to_csv (string): Path to Object Interaction data
            num_frames (int): number of frames used for model
            num_condition_frames (int): number of frames to condition model on
            num_pose_prediction (int): number of future poses to predict
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "test",
        ], "mode has to be 'train' or 'test'"
        self.mode = mode

        self._num_retries = num_retries
        self._path_to_data_dir = path_to_data_dir
        self._path_to_trajectories_dir = path_to_trajectories_dir
        self._path_to_csv = path_to_csv

        self._crop_size = crop_size

        self._num_frames = num_condition_frames
        self._num_sec = math.ceil((num_condition_frames + num_pose_prediction) / fps) #double check logic
        self._fps=fps
        self._pps = pps
        self._pose_skip = fps // pps

        self.transform = transform
        
        # Pose Estimation Settings
        self._animals = animals
        self._num_condition_frames = num_condition_frames
        self._num_pose_prediction = num_pose_prediction
        self._scale_invariance = scale_invariance

        self._enable_multi_thread_decode = enable_multi_thread_decode
        self._inverse_uniform_sampling = inverse_uniform_sampling
        self._use_offset_sampling = use_offset_sampling
        self._num_retries = num_retries

        print(self)
        print(locals())
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        self._path_to_videos = []
        self._path_to_trajectories = []
        self._start_times = []
        self._end_times = []
        self._stride = []
        
        with pathmgr.open(self._path_to_csv, "r") as f:
            for curr_clip in f.read().splitlines():
                curr_values = curr_clip.split(',')

                animal, ds_type, segment_id, stride, start_time, end_time, video_path = curr_values # NEED TO GENERATE THIS CSV
                video_path = os.path.join(self._path_to_data_dir, video_path)
                if ((self.mode == 'train' and ds_type == 'training_set') or (self.mode == 'test' and ds_type == 'validation_set')) and stride != -1 and animal in self._animals:
                    start_time, end_time = int(start_time), int(end_time)
                    
                    if self.mode == 'test':
                        end_time = min(end_time, 50) # Clip Evaluation Videos to 50 Seconds
                    
                    trajectory = "{}_calib_eth_stride_{}_interp.txt".format(segment_id, stride)
                    trajectory_path = os.path.join(self._path_to_trajectories_dir, trajectory)
                    
                    if os.path.isfile(trajectory_path) and os.path.isfile(video_path) and (end_time - start_time) >= self._num_sec:
                        video_path = os.path.join(self._path_to_data_dir, video_path)
                        self._path_to_videos.append(video_path)
                        self._path_to_trajectories.append(trajectory_path)
                        self._start_times.append(start_time)
                        self._end_times.append(end_time)
                        self._stride.append(stride)
            
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Pose Trajectories from {}".format(
            self._path_to_csv
        )
        print(
            "Constructing Object Interaction dataloader (size: {}) from {}".format(
                len(self._path_to_videos), self._path_to_csv
            )
        )
        
    
    def _get_sub_clip(self, video_path, trajectory_path, sub_start_idx, sub_end_idx):
        """
        Get sub clip of video of from sub_start_idx to sub_end_idx.
        Args: 
            video_path (string): path to video
            trajectory_path (string): path to associated trajectory
            sub_start_idx (int): start idx for this sub clip
            sub_end_idx (int): end idx for this sub clip
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `num_condition_frames` x `channel` x `height` x `width`.
            dP_tensor (tensor): the relative poses for the entire clip. 
        """
        # For validation should get the entire segment
        start_seek = sub_start_idx // self._fps
        num_sec = (sub_end_idx - sub_start_idx) / self._fps + 1.0
        dP_tensor = relative_poses(trajectory_path, sub_start_idx, sub_end_idx, scale_invariance=None, pose_skip=1)
        
        real_num_frames = (sub_end_idx - sub_start_idx)
        frame_relative_start_idx = sub_start_idx - start_seek * self._fps
        num_frames = int(num_sec * self._fps + 512) # Set the num_frames to be more frames than decoded in order to get all the decoded frames
        frames = decode_ffmpeg(video_path, start_seek=start_seek, num_sec=num_sec, num_frames=num_frames, fps=self._fps)
        frames = frames[frame_relative_start_idx:frame_relative_start_idx + real_num_frames]
        
        frames = frames.permute(0, 3, 1, 2) / 255.
        frames = torch.stack([self.transform(f) for f in frames])
        
        return frames, dP_tensor


    def __getitem__(self, index):
        """
        If self.mode is train, randomnly choose a self.num_condition_frames + 
        self.num_pose_prediction frame clip. If self.mode is test, choose the 
        entire video segment. If the video cannot be fetched and decoded 
        successfully, find a random video that can be decoded as a replacement. 
        Args:
            index (int): the video index provided by the pytorch sampler. (not used)
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `num_condition_frames` x `channel` x `height` x `width`.
            cond_poses (tensor): the associated poses to the conditioning frames. 
                The dimension is `num_condition_frames` x `channel` x `height` x `width`.
            dP_tensor (tensor): the relative poses for the entire clip. 
            start_idx (int): the start index of this clip
            end_idx (int): the end index of this clip
        """
        for i_try in range(self._num_retries):
            if self.mode == 'train': # If 'train', sample a random video but if 'test' use index
                index = random.randint(0, len(self._path_to_videos) - 1)
            
            video_path = self._path_to_videos[index]
            trajectory_path = self._path_to_trajectories[index]
            start_time = self._start_times[index]
            end_time = self._end_times[index]
            stride = self._stride[index]

            # Decode Video
            try:
                if self.mode == 'train':
                    # For training randomnly select start of clip
                    start_seek = np.random.randint(start_time, max(end_time - self._num_sec, 1))
                    start_idx = (start_seek * self._fps)
                    end_idx = start_idx + self._num_condition_frames + self._num_pose_prediction
                    
                    if self._scale_invariance == 'dir':
                        dP_tensor, _ = relative_poses(trajectory_path, start_idx + self._num_condition_frames, end_idx, self._scale_invariance, self._pose_skip)
                    
                    frames = decode_ffmpeg(video_path, start_seek=start_seek, num_sec=self._num_sec, num_frames=self._num_frames, fps=self._fps)
                elif self.mode == 'test':
                    # For validation should get the entire segment
                    start_idx = start_time * self._fps
                    num_sub_clips = ((end_time - start_time) * self._fps) // (self._num_condition_frames + self._num_pose_prediction)
                    end_idx = start_idx + num_sub_clips * (self._num_condition_frames + self._num_pose_prediction)
                    return video_path, trajectory_path, start_idx, end_idx
                
                if frames.shape[0] == 0:
                    raise ValueError('Decoder Error, 0 frames decoded at video path {video_path}, start_seek: {start_seek}, num_sec: {num_sec}, self._num_frames: {num_frames}, fps: {fps}, start_time: {start_time}, end_time: {end_time}, total_time: {total_time}'.format(video_path=video_path, start_seek=start_seek, num_sec=num_sec, num_frames=self._num_frames, fps=self._fps, start_time=start_time, end_time=end_time, total_time=total_time))        
            except Exception as e:
                print(
                    "Failed to decode video idx {} from {} with error {}".format(
                        index, video_path, e
                    )
                )
                # Random selection logic in getitem so random video will be decoded
                return self.__getitem__(0)
            
            # Keep just the first num_condition_frames frames
            if self.mode == 'train':
                frames = frames[:self._num_condition_frames]
            elif self.mode == 'test':
                frames = frames[start_idx:end_idx]
            frames = frames.permute(0, 3, 1, 2) / 255.
            frames = torch.stack([self.transform(f) for f in frames])
            return frames, dP_tensor.to(torch.float32), start_idx, end_idx
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(self._num_retries)
            )


    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        if self.mode == 'train':
            return 20000
        elif self.mode == 'test':
            return self.num_videos


    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)