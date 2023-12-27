import ffmpeg
import numpy as np
import torch

def decode_ffmpeg(video_path, start_seek, num_sec=2, num_frames=16, fps=5):

    probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='stream=width,height,duration,r_frame_rate')
    video_info = next((s for s in probe['streams'] if 'width' in s and 'height' in s), None)
    
    if video_info is None:
        raise ValueError("No video stream information found in the input video.")
    
    width = int(video_info['width'])
    height = int(video_info['height'])
    r_frame_rate = video_info['r_frame_rate'].split('/')
            
    if fps is None:
        fps = int(r_frame_rate[0]) / int(r_frame_rate[1])
    
    cmd = (
        ffmpeg
        .input(video_path, ss=start_seek, t=num_sec + 0.1)
        .filter('fps', fps=fps)
    )
    out, _ = (
        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, quiet=True)
    )
    
    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    video_copy = video.copy()
    video = torch.from_numpy(video_copy)    
    return video[:num_frames].type(torch.float32)
