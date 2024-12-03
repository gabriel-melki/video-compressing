import os
from stat import ST_MTIME
from typing import List
import subprocess
import ffmpeg

def get_files_ordered_by_date(folder_path: str) -> List[str]:
    """
    Get a list of all files in the folder
    """
    files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path) 
        if os.path.isfile(os.path.join(folder_path, f)) &
        f.endswith('.mp4')
    ]
    
    # Sort files based on their modified date
    files.sort(key=lambda x: os.stat(x)[ST_MTIME])
    
    return files


def size_of_files(file_list: List[str]) -> float:
    """
    TODO
    """
    total_size = 0
    for filename in file_list:
        if os.path.isfile(filename):
            total_size += os.path.getsize(filename)
        else:
            print(f"Skipping {filename} as it is not a file.")
    return total_size


def get_media_duration(file: str) -> float:
    """
    Retrieve the duration of a media file using ffprobe
    """
    try:
        result = subprocess.run([
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            str(file)
        ], capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except subprocess.CalledProcessError as e:
        raise ValueError(f"FFprobe failed to analyze reduced video: {e}") from e
 

def get_files_with_full_path(directory: str) -> List[str]:
    """
    Get the full paths of all files in the specified directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        List[str]: List of full paths of files in the directory, sorted by alphabetical order.
    """
    file_paths = []
    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, item)):
            file_paths.append(os.path.join(directory, item))
    return sorted(file_paths)


def get_video_bitrate(input_file: str) -> int:
    """
    Returns the bitrate of the input video file.
    """
    try:
        # Probe the input file to get its information
        probe = ffmpeg.probe(input_file, v='error', select_streams='v:0', show_entries='stream=bit_rate')

        # Extract the bitrate from the probe result
        bitrate = int(probe['streams'][0]['bit_rate'])  # Bitrate is in bits per second
        
        return bitrate
    except ffmpeg.Error as e:
        raise Exception(f"Error getting bitrate for {input_file}: {e.stderr.decode()}") from e


def get_audio_bitrate(input_file: str) -> int:
    """
    Returns the bitrate of the input video file.
    """
    try:
        # Probe the input file to get its information
        probe = ffmpeg.probe(input_file, v='error', select_streams='a:0', show_entries='stream=bit_rate')

        # Ensure 'streams' and 'bit_rate' exist in the probe result
        if 'streams' in probe and len(probe['streams']) > 0 and 'bit_rate' in probe['streams'][0]:
            bitrate = int(probe['streams'][0]['bit_rate'])  # Bitrate is in bits per second
            return bitrate
        else:
            return 0
        
    except ffmpeg.Error as e:
        raise Exception(f"Error getting bitrate for {input_file}: {e.stderr.decode()}") from e
