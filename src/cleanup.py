import argparse
import shutil
from pathlib import Path
from datetime import datetime
from config import OUTPUT_DIR
from utils import get_timestamp


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse a timestamp string back to datetime object."""
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def find_timestamp_dirs(base_dir: Path, current_depth: int = 0, max_depth: int = 2):
    """
    Recursively find timestamp directories up to max_depth.
    Returns tuples of (timestamp_dir, parsed_timestamp)
    """
    timestamp_dirs = []

    if current_depth > max_depth:
        return timestamp_dirs

    for item in base_dir.iterdir():
        if not item.is_dir():
            continue

        # Try to parse as timestamp
        timestamp = parse_timestamp(item.name)
        if timestamp:
            timestamp_dirs.append((item, timestamp))
        else:
            # If not a timestamp, recurse into directory
            timestamp_dirs.extend(find_timestamp_dirs(
                item, current_depth + 1, max_depth))

    return timestamp_dirs


def delete_old_outputs():
    """Delete output directories with timestamps older than the current time."""
    current_time = parse_timestamp(get_timestamp())
    deleted_count = 0

    # Find all timestamp directories recursively (up to depth 2)
    timestamp_dirs = find_timestamp_dirs(OUTPUT_DIR)

    # Delete directories with timestamps older than current time
    for dir_path, dir_time in timestamp_dirs:
        if dir_time < current_time:
            print(f"Deleting: {dir_path.relative_to(OUTPUT_DIR)}")
            shutil.rmtree(dir_path)
            deleted_count += 1

    print(f"Cleanup completed. Deleted {deleted_count} timestamp directories.")


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup old rule generator outputs")
    parser.add_argument(
        "--delete-old-outputs", "-doo",
        action="store_true",
        help="Delete output directories with timestamps older than current time"
    )
    args = parser.parse_args()

    if args.delete_old_outputs:
        delete_old_outputs()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
