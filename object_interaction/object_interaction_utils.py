import os
import torch

objects_dict = {
    'person': 0, 
    'ball': 1, 
    'bench': 2, 
    'bird': 3, 
    'dog': 4, 
    'cat': 5, 
    'other animal': 6, 
    'toy': 7, 
    'door': 8, 
    'floor': 9, 
    'food': 10, 
    'plant': 11, 
    'filament': 12, 
    'plastic': 13, 
    'water': 14, 
    'vehicle': 15, 
    'other': 16, 
}

# Decoding Helper Functions
def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """

    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames = transform.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames = transform.random_crop(frames, crop_size)
        else:
            transform_func = (
                transform.random_resized_crop_with_shift
                if motion_shift
                else transform.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames = transform.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale}) == 1
        frames = transform.random_short_side_scale_jitter(frames, min_scale, max_scale)
        frames = transform.uniform_crop(frames, crop_size, spatial_idx)
    return frames

def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    new_frames = torch.index_select(frames, 0, index)
    return new_frames

def get_start_end_idx(video_size, clip_size, clip_idx, num_clips, use_offset=False):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        if use_offset:
            if num_clips == 1:
                # Take the center clip if num_clips is 1.
                start_idx = math.floor(delta / 2)
            else:
                # Uniformly sample the clip with the given index.
                start_idx = clip_idx * math.floor(delta / (num_clips - 1))
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


# Helper Functions
def get_video_path(path_to_data_dir, animal, video_id, segment_id):
    video_name = 'edited_{video_id}_segment_{segment_id}.mp4'.format(video_id=video_id, segment_id=segment_id.zfill(6))
    video_path = os.path.join(path_to_data_dir, animal, video_name)
    return video_path

def process_time(time):
    # Assumes in format ##:##:## or ##:## or NONE to get time in seconds
    if time == 'NONE':
        return 0

    time_split = time.split(':')
    if len(time_split) == 1:
        time_sec = int(time_split[0])
    elif len(time_split) == 2:
        time_sec = int(time_split[0]) * 60 + int(time_split[1])
    elif len(time_split) == 3:
        time_sec = int(time_split[0]) * 3600 + int(time_split[1]) * 60 + int(time_split[2])
    else:
        raise ValueError('Invalid Time was {time} but expected in format ##:##:##, ##:##, or #'.format(time=time))
    return time_sec

def process_end_time(time, total_time):
    # Get end time for processing NONE times
    if time == 'NONE':
        return process_time(total_time)
    else:
        return process_time(time)

def get_none_segments(start_time, end_time, total_time, interacting_object):
    min_beg_end_length = 4 + 2 # the actual clip must be at least 4 seconds, 2 seconds from the next clip 
    min_mid_length = 2 + 4 + 2 # 2 seconds after the last clip, the actual clip must be at least 4 seconds, 2 clips before next clip
    
    new_start_time = []
    new_end_time = []
    new_interacting_object = []
    
    # Checking middle portions
    for i in range(len(start_time) - 1):
        idx = i+1
        if start_time[idx] - end_time[i] >= min_mid_length:
            new_start_time.append(end_time[i]+2)
            new_end_time.append(start_time[idx]-2)
            new_interacting_object.append('NONE')
    
    # Checking beginning
    if start_time[0] >= min_beg_end_length:
        new_start_time.append(0)
        new_end_time.append(start_time[0]-2)
        new_interacting_object.append('NONE')
    
    # Checking ending
    if total_time - end_time[-1] >= min_beg_end_length:
        new_start_time.append(end_time[-1]+2)
        new_end_time.append(total_time)
        new_interacting_object.append('NONE')
        
    start_time.extend(new_start_time)
    end_time.extend(new_end_time)
    interacting_object.extend(new_interacting_object)
    return start_time, end_time, interacting_object

def get_label(interacting_object):
    if interacting_object == 'NONE':
        return -1
    elif interacting_object in objects_dict:
        object_label_idx = objects_dict[interacting_object]
    else:
        raise ValueError("Invalid Object Type: {curr_object}".format(curr_object=interacting_object))

    return object_label_idx
 