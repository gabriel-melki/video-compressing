"""
TODO
"""
from typing import Union, List, Optional, Dict
import os
import uuid
import subprocess
import logging
from pathlib import Path
from tqdm import tqdm
import ffmpeg
from src.video_compressing.helper import get_video_bitrate, get_audio_bitrate

logger = logging.getLogger(__file__)

def _get_params_for_compression(
        input_file: str,
        reduction_factor: float
    ) -> Dict:
    # Get video format
    video_format = input_file.split('.')[-1]

    # Determine best params given video format
    if video_format == 'mov':
        ideal_params = {
            "vf":f'scale=iw*{reduction_factor}:ih*{reduction_factor}',
            "acodec":'aac',  # Specify AAC codec for audio
            "vcodec":'libx264',  # Specify H.264 codec for video
            "preset":'veryslow',  # Compression preset
            "crf":int(23 / reduction_factor) # Quality parameter for H.264
        }
    elif video_format == 'mp4':
        video_bitrate = get_video_bitrate(input_file)
        audio_bitrate = get_audio_bitrate(input_file)
        ideal_params = {
            "vf": f'scale=iw*{reduction_factor}:ih*{reduction_factor}',
            "acodec": 'aac',  # Specify AAC codec for audio
            "vcodec": 'libx264',  # Specify H.264 codec for video
            "video_bitrate": f'{int(video_bitrate * reduction_factor ** 2)}k',
            "audio_bitrate": f'{int(audio_bitrate * reduction_factor ** 2)}k',
            "preset": 'veryslow',  # Compression preset
            "crf": int(23 / reduction_factor) # Quality parameter for H.264
        }
    else:
        raise ValueError(f"Video format {video_format} is not supported")

    return ideal_params


def _validate_output_file(
        input_file: str,
        output_file: Optional[str]= None
    ) -> Path:
    # If no output file, create a valid one.
    if output_file is None:
        current_dir = Path(input_file).parent
        output_file = current_dir / f"{uuid.uuid1()}.mp4"
    # Else validate the extension and directory
    else:
        output_file = Path(output_file) if not isinstance(output_file, Path) else output_file
        # Force the file to have a .mp4 extension
        output_file = output_file.with_suffix('.mp4')
        if output_file.exists():
            logger.info("%s already exists. Overriding it.", output_file)
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    return output_file


def merge_videos_to_single_mp4(
        file_list: List[str],
        output_file: Optional[str] = None
    ) -> Path:
    """
    Merges multiple videos files into a single file.

    Args:
        file_list (List[str]): List of videos files to merge.
        output_file (str): Path to the output merged MP4 file.
    """
    # Validate output path
    validated_output_file = _validate_output_file(file_list[0], output_file)

    # Create a temporary file list for FFmpeg
    temp_file = f"{uuid.uuid1()}.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        for file in file_list:
            f.write(f"file '{os.path.abspath(file)}'\n")

    # Command to run FFmpeg for merging files
    command = [
        'ffmpeg',
        '-f', 'concat',              # Use concat demuxer
        '-safe', '0',                # Allow unsafe file paths
        '-i', temp_file,             # Input list file
        '-c', 'copy',                # Copy codec (no re-encoding)
        '-y', validated_output_file  # Override the outputfile
    ]

    try:
        # Run the FFmpeg command
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Merged files saved to {validated_output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during merging: {e}")
    finally:
        # Clean up the temporary file list
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return Path(validated_output_file)


def reduce_video_size(
        input_file: str,
        reduction_factor: Union[int, float],
        output_file: Optional[str] = None
    ) -> Path :
    """
    Reduces the size of a video file by adjusting its bitrate and scaling.

    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to save the output reduced-size video.
        reduction_factor (Union[int, float]): Factor by which to reduce the video size.
            A value of 1 means no reduction, 0.5 means reducing the size to half, and so on.
    """
    # Input validation
    if not 0 < reduction_factor <= 1:
        raise ValueError(f"Reduction_factor must be between 0 and 1. Got {reduction_factor}")

    # Validate output path
    validated_output_file = _validate_output_file(input_file, output_file)

    try:
        input_stream = ffmpeg.input(input_file)
        compression_params = _get_params_for_compression(
            input_file=input_file,
            reduction_factor=reduction_factor
        )
        output_stream = ffmpeg.output(
            input_stream,
            str(validated_output_file), y=None,
            **compression_params
        )
        ffmpeg.run(output_stream, quiet=True)
        return validated_output_file
    except ffmpeg.Error as e:
        raise RuntimeError('Error occurred:', e.stderr) from e


def reduce_and_merge_videos(
        input_files: List[str],
        reduction_factor: float,
        output_file: Optional[str] = None
    ) -> Path:
    """
    The following tasks are performed:
        - Reduce the size of a list of .MOV video files
        - Merge them
        - Save the merged video to the specified output file.

    Args:
        input_files (List[str]): List of input .MOV video file paths to be reduced and merged.
        output_file (str): Path to the output file to save the merged video.
        reduction_factor (float): Factor by which to reduce the video size.
        A value of 1 means no reduction.
        0.5 means reducing the size to half, and so on.

    Returns:
        None. The function reduces the size of the input video files, merges them,
        and saves the merged video to the specified output file.
    """
    # Save reduced video files in the temporary folder
    reduced_files = [
        reduce_video_size(file, reduction_factor)
        for file in tqdm(input_files, desc="Reducing Size")
    ]

    try:
        # Merge the reduced video files into a single video
        merged_output_file = merge_videos_to_single_mp4(reduced_files, output_file)
    except Exception as exc:
        raise ValueError(f'Error occurred during merging: {exc}') from exc
    finally:
        # Delete the temporary reduced files
        for file in tqdm(reduced_files, desc="Cleaning"):
            os.remove(file)

    return merged_output_file
