# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import random

import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr as pathmgr

import sys
parent = os.path.dirname(os.path.abspath(__file__))
parent_parent = os.path.join(parent, '../')
sys.path.append(os.path.dirname(parent_parent))

from decoder.utils import decode_ffmpeg
from object_interaction.object_interaction_utils import *
from torchvision import transforms

from pathlib import Path


class Object_Interaction(torch.utils.data.Dataset):
    """
    Object Interaction video loader. Construct the Object Interaction video loader, 
    then sample clips from the videos. For training and validation, a single clip 
    is randomly sampled from every video with normalization.
    """

    def __init__(
        self,
        mode,
        path_to_data_dir,
        path_to_csv, 
        # transformation
        transform,
        # decoding settings
        num_frames=8,
        target_fps=30,
        num_sec=2,
        fps=4,
        # frame aug settings
        crop_size=224,
        # other parameters
        enable_multi_thread_decode=False,
        use_offset_sampling=True,
        inverse_uniform_sampling=False,
        num_retries=10,
        # object or no object
        object_interaction_ratio=0.5,
    ):
        """
        Construct the Object Interaction video loader with a given csv file. The format of
        the csv file is:
        ```
        animal_1, ds_type_1, video_id_1, segment_id_1, start_time_1, end_time_1, total_time_1, interacting_object_1, video_path_1
        animal_2, ds_type_2, video_id_2, segment_id_2, start_time_2, end_time_2, total_time_2, interacting_object_2, video_path_2
        ...
        animal_N, ds_type_N, video_id_N, segment_id_N, start_time_N, end_time_N, total_time_N, interacting_object_N, video_path_N
        ```
        Args:
            mode (string): Options includes `train` or `test`.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
            path_to_data_dir (string): Path to EgoPet Dataset
            path_to_csv (string): Path to Object Interaction data
            num_frames (int): number of frames used for model
            object_interaction_ratio (float): ratio of clips with interactions during training
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "test",
        ], "mode has to be 'train' or 'test'"
        self.mode = mode

        self._num_retries = num_retries
        self._path_to_data_dir = path_to_data_dir
        self._path_to_csv = path_to_csv
        self.object_interaction_ratio = object_interaction_ratio

        self._crop_size = crop_size

        self._num_frames = num_frames
        self._num_sec = num_sec
        self._target_fps = target_fps
        self._fps=fps

        self.transform = transform

        self._enable_multi_thread_decode = enable_multi_thread_decode
        self._inverse_uniform_sampling = inverse_uniform_sampling
        self._use_offset_sampling = use_offset_sampling

        print(self)
        print(locals())
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        self._object_path_to_videos = []
        self._no_object_path_to_videos = []
        self._object_labels_times = []
        self._no_object_labels_times = []

        with pathmgr.open(self._path_to_csv, "r") as f:
            for curr_clip in f.read().splitlines():
                curr_values = curr_clip.split(',')
                if curr_values[0] != 'animal':
                    animal, ds_type, video_id, segment_id, start_time, end_time, total_time, interacting_object, video_path = curr_values
                    video_path = os.path.join(self._path_to_data_dir, video_path)
                    start_time, end_time, interacting_object = start_time.split(';'), end_time.split(';'), interacting_object.split(';')
                    start_time, end_time, total_time = [process_time(time) for time in start_time], [process_end_time(time, total_time) for time in end_time], process_time(total_time)
                    
                    assert len(start_time) == len(end_time) == len(interacting_object), "Error with csv on {curr_row}".format(curr_row=curr_values)
            
                    # Getting None Segments inbetween videos
                    start_time, end_time, interacting_object = get_none_segments(start_time, end_time, total_time, interacting_object)
                    
                    object_labels = [get_label(curr_interacting_object) for curr_interacting_object in interacting_object]
                    assert len(start_time) == len(end_time) == len(interacting_object), "Error with processing"
                    
                    for i in range(len(start_time)):
                        interaction = 0. if object_labels[i] == -1 else 1.
                        if object_labels[i] == -1:
                            object_labels[i] = 0 
                            
                        labels_times = (interaction, object_labels[i], start_time[i], end_time[i], total_time)
                        if interaction:
                            self._object_path_to_videos.append(video_path)
                            self._object_labels_times.append(labels_times)
                        else:
                            self._no_object_path_to_videos.append(video_path)
                            self._no_object_labels_times.append(labels_times)

        self._path_to_videos = self._object_path_to_videos + self._no_object_path_to_videos
        self._labels_times = self._object_labels_times + self._no_object_labels_times

        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Object Interaction from {}".format(
            self._path_to_csv
        )
        print(
            "Constructing Object Interaction dataloader (size: {}) from {}".format(
                len(self._path_to_videos), self._path_to_csv
            )
        )

    def __getitem__(self, index):
        """
        With probability self.object_interaction_ratio randomly choose a clip
        with an object interaction, otherwise randomly choose a clip without
        an object interaction. If the video cannot be fetched and decoded
        successfully, find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler. (not used)
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `num frames` x `channel` x `height` x `width`.
            interaction (int): whether there is an interaction in the vidoe. 
                0 for no interaction, 1 for an interaction.
            object_label (int): the label of the current video.
        """
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            if self.mode == 'train':
                # If training sample random video with object interaction 
                # with prob self.object_interaction_ratio
                if random.random() < self.object_interaction_ratio:
                    # Get sample with object interaction
                    index = random.randint(0, len(self._object_path_to_videos) - 1)
                    # Get a clip with an object interaction
                    video_path = self._object_path_to_videos[index]
                    label_time = self._object_labels_times[index]
                    interaction, object_label, start_time, end_time, total_time = label_time
                else:
                    # Get sample without object interaction
                    index = random.randint(0, len(self._no_object_path_to_videos) - 1)
                    # Get a clip without object interaction
                    video_path = self._no_object_path_to_videos[index]
                    label_time = self._no_object_labels_times[index]
                    interaction, object_label, start_time, end_time, total_time = label_time
            elif self.mode == 'test':
                # If test sample provided index
                video_path = self._path_to_videos[index]
                label_time = self._labels_times[index]
                interaction, object_label, start_time, end_time, total_time = label_time

            # Decode Video
            try:
                if self.mode == 'train':
                    # For training randomnly select start of clip
                    start_seek = min(max(int((start_time + end_time - self._num_sec) / 2), start_time), max(end_time - self._num_sec, 0))
                elif self.mode == 'test':
                    # Gets as to close to the center as possible of current clip
                    start_seek = min(max(int((start_time + end_time - self._num_sec) / 2), start_time), max(end_time - self._num_sec, 0))
                
                num_sec = min(self._num_sec, end_time-start_time)
                frames = decode_ffmpeg(video_path, start_seek=start_seek, num_sec=num_sec, num_frames=self._num_frames, fps=self._fps)
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
            
            start_idx, end_idx = get_start_end_idx(
                frames.shape[0], self._num_frames, 0, 1
            )
            frames = temporal_sampling(
                frames, start_idx, end_idx, self._num_frames
            )
            
            frames = frames.permute(0, 3, 1, 2) / 255.
            frames = torch.stack([self.transform(f) for f in frames])
            return frames, torch.tensor([interaction]), object_label
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
