# Video Compressing
A Python library for reducing the size of and merging multiple video files. This package is designed for simplicity and efficiency, leveraging FFmpeg for video compression and manipulation.

## Features
- Reduce Video Size: Scale down video dimensions and bitrate using a customizable reduction factor.
- Merge Multiple Videos: Combine multiple video files into a single output.
- Command-Line Interface: Use the package directly from the terminal for quick processing.

## Installation
Clone the Repository
```bash
git clone https://github.com/Gabriel-melki/video-compressing.git
cd video-compressing
```

### Install Dependencies
This project uses Poetry for dependency management. Install Poetry if you don’t have it:
```bash
pip install poetry
```
Install the dependencies and activate the vitual environment:
```bash
poetry install
poetry shell
```

Ensure FFmpeg is installed on your system:

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```
**MacOS (with Homebrew):**
```bash
brew install ffmpeg
```

**Windows:**

Download FFmpeg and add it to your PATH.

### Usage
Command-Line Tool
You can run the VideoCompressing tool directly from the terminal.

```bash
python -m src.video_compressing.tools input1.mp4 input2.mp4 -r 0.5 -o output.mp4
```
Parameters:
 - input_files: List of input video files (required)
 - -r, --reduction-factor: A float value to scale down videos (required).
 Example: 0.5 reduces the size by 50%.
 - o, --output-file: Name of the output file (optional).
 Default: Automatically generated if not provided.

### Example:
Reduce and merge video1.mp4 and video2.mp4 by 50% into output.mp4:

```bash
python -m src.video_compressing.tools video1.mp4 video2.mp4 -r 0.5 -o output.mp4
```

Without an output file:
```bash
python -m src.video_compressing.tools video1.mp4 video2.mp4 -r 0.5
```

Using as a Library
Import and call the reduce_and_merge_videos function directly in your Python code:

```python
from src.video_compressing.tools import reduce_and_merge_videos

reduce_and_merge_videos(
    input_files=["video1.mp4", "video2.mp4"],
    reduction_factor=0.5,
    output_file="output.mp4"
)
```

File Structure
```bash
VideoCompressing/
├── src/
│   └── video_compressing/
│       ├── __init__.py
│       └── tools.py  # Contains reduce_and_merge_videos and CLI logic
├── script/
├── test/
├── .venv/
├── .vscode/
├── poetry.lock
├── pyproject.toml  # Defines the library and dependencies
├── README.md
└── video_editing_environment.yaml
```