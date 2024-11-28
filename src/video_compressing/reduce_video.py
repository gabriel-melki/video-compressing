import argparse
from src.video_compressing.tools import reduce_and_merge_videos

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Reduce the size of and merge multiple video files."
    )

    # Add arguments for input files
    parser.add_argument(
        "input_files",
        metavar="input_files",
        nargs="+",
        help="List of input video files to be reduced and merged."
    )

    # Add argument for the reduction factor
    parser.add_argument(
        "-r", "--reduction-factor",
        type=float,
        required=True,
        help="Reduction factor to scale the videos. E.g., 0.5 for 50% size."
    )

    # Add optional argument for the output file
    parser.add_argument(
        "-o", "--output-file",
        type=str,
        default=None,
        help="Output file name. If not specified, a default name will be generated."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    reduce_and_merge_videos(
        input_files=args.input_files,
        reduction_factor=args.reduction_factor,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()
