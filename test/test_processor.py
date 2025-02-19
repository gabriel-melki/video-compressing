"""
This module contains the tests for the processor functions.
"""

import os
import uuid
import pytest
import requests
import numpy as np
from video_compressing.utils import get_media_duration
from video_compressing.processor import (
    reduce_video_size,
    reduce_and_merge_videos,
    merge_videos_to_single_mp4,
)

TOLERANCE = 0.01

@pytest.fixture(scope="function", params=[
    # Testing .mp4 format
    [
        "https://getsamplefiles.com/download/mp4/sample-1.mp4",
        "https://getsamplefiles.com/download/mp4/sample-2.mp4"
    ],
    # Testing .mov format
    [
        "https://getsamplefiles.com/download/mov/sample-1.mov",
        "https://getsamplefiles.com/download/mov/sample-2.mov"
    ],
])
def test_video_files(tmp_path, request):
    """
    Fixture to download small test .MOV video files with cleanup
    """
    # Sample small video URLs (replace with reliable small video URLs)
    video_urls = request.param
    video_paths = []

    for video_url in video_urls:
        # Download video
        response = requests.get(video_url, stream=True, timeout=10)
        response.raise_for_status()
        video_name = video_url.split('/')[-1]
        video_path = tmp_path / video_name

        with open(video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify file size and content
        assert video_path.stat().st_size > 0, f"Downloaded file {video_path} is empty"
        video_paths.append(str(video_path))


    yield video_paths

    # Cleanup: remove all files in the temporary directory
    for path in tmp_path.iterdir():
        path.unlink()


@pytest.fixture(scope="function", params=[
    None,
    str(uuid.uuid1())
])
def optional_output_file(request) -> str:
    """
    Fixture to specify the name of the output_file, and ensure cleanup afterwards.
    """
    output_file = request.param

    # Yield the output file path to the test function
    yield output_file

    # Cleanup: delete the file if it exists
    if output_file and os.path.exists(output_file):
        os.remove(output_file)


def test_merge_videos_to_single_mp4(test_video_files, optional_output_file):
    """
    Test MP4 file merging
    """
    try:
        # Perform file merging
        output_file = merge_videos_to_single_mp4(test_video_files, optional_output_file)

        # Verifications
        assert output_file.exists(), f"Merged output file was not created at location {output_file}"

        # Verify video is valid
        duration = get_media_duration(output_file)
        assert duration > 0, "Reduced video has zero duration"

        # Check file size
        total_input_size = int(np.sum([
            os.path.getsize(input_file) for input_file in test_video_files
        ]))
        output_size = os.path.getsize(output_file)
        assert output_size < total_input_size * (1 + TOLERANCE), \
            f"Output size {output_size} should not be higher than the input size {total_input_size}"

    finally:
        # Ensure cleanup of output file
        if output_file and output_file.exists():
            output_file.unlink()

@pytest.mark.parametrize("reduction_factor", [0.2, 0.5, 1])
def test_reduce_video_size(test_video_files, optional_output_file, reduction_factor):
    """
    Test video size reduction
    """
    created_files = []
    try:
        for input_file in test_video_files:
            # Perform video size reduction
            output_file = reduce_video_size(
                input_file=input_file,
                reduction_factor=reduction_factor,
                output_file=optional_output_file
            )
            created_files.append(output_file)
            # Verifications
            assert output_file.exists(), f"Output file was not created for {input_file}"

            # Verify video is valid
            duration = get_media_duration(output_file)
            assert duration > 0, f"Reduced video has zero duration for {input_file}"

            # Check file size reduction with 1% tolerance
            input_size = os.path.getsize(input_file)
            output_size = os.path.getsize(output_file)
            target_size = input_size * reduction_factor
            assert output_size <= target_size * (1 + TOLERANCE), \
                f"Output size {output_size} and input size {input_size} are not " \
                f"respecting the reduction_factor of {reduction_factor} (with 1% tolerance)"
    finally:
        # Ensure cleanup of all created files
        for output_file in created_files:
            if output_file and output_file.exists():
                output_file.unlink()


@pytest.mark.parametrize("reduction_factor", [0.2, 0.5, 1])
def test_reduce_and_merge_videos(test_video_files, optional_output_file, reduction_factor):
    """
    Test video reduction and merging
    """
    try:
        input_files = test_video_files
        created_files = []
        # Perform video reduction and merging
        output_file = reduce_and_merge_videos(
            input_files=input_files,
            reduction_factor=reduction_factor,
            output_file=optional_output_file
        )
        created_files.append(output_file)
        # Verifications
        assert output_file.exists(), "Merged output file was not created"

        # Verify video is valid
        duration = get_media_duration(output_file)
        assert duration > 0, "Reduced video has zero duration"

        # Check file size with 1% tolerance
        total_input_size = int(np.sum([
            os.path.getsize(input_file) for input_file in input_files
        ]))
        output_size = os.path.getsize(output_file)
        target_size = total_input_size * reduction_factor
        assert output_size <= target_size * (1 + TOLERANCE), \
            f"Output size {output_size} and input size {total_input_size} are not " \
            f"respecting the reduction_factor of {reduction_factor} (with 1% tolerance)"
    finally:
        # Ensure cleanup of all created files
        for output_file in created_files:
            if output_file and output_file.exists():
                output_file.unlink()
