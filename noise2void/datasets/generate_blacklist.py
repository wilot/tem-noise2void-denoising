"""generate_blacklist.py

Scans a directory of raw .dm4 video files, reads the frame count of each one, and writes
a starter blacklist.toml that includes every frame of every video.

Edit the generated file afterwards to:
  - Restrict a video to valid frame ranges:  "0231a" = [[4, 815]]
  - Add multiple ranges to skip bad sections: "0231a" = [[0, 300], [320, 800]]
  - Exclude a video entirely:                 "0231a" = false

Usage (run from the project root):
    python -m noise2void.datasets.generate_blacklist
    python -m noise2void.datasets.generate_blacklist --data_dir data/MyExperiment/raw --output noise2void/datasets/my_blacklist.toml
"""

import re
import argparse
from pathlib import Path

import hyperspy.api as hs


# ── file format ───────────────────────────────────────────────────────────────
FILE_EXTENSION = ".dm3"   # Change to ".dm4", ".tif", etc. to match your data

# Regex pattern to parse each filename stem (without extension).
# Must define named groups:
#   channel — detector channel label (e.g. HAADF, BF)
#   id      — unique identifier for this video, used as the blacklist key
#   frames  — (optional) frame count encoded in the filename; if this group is
#             absent or doesn't match, the file is opened to count frames instead
#
# Current format:  "HAADF STACK(100)-50"  or  "BF STACK(100)-50"
FILENAME_PATTERN = r"^(?P<channel>\w+) STACK\((?P<frames>\d+)\)-(?P<id>\d+)$"

# Only files whose parsed channel matches this string are processed.
# Paired files with other channels (e.g. BF when PRIMARY_CHANNEL="HAADF") are skipped
# so each video group produces exactly one blacklist entry.
# Set to None to process every file regardless of channel.
PRIMARY_CHANNEL = "HAADF"

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_OUTPUT   = Path("data/blacklist.toml")


def _parse_stem(stem: str) -> tuple[str, str, int | None] | None:
    """Parses a filename stem using FILENAME_PATTERN.

    Returns (channel, video_id, frame_count) where frame_count is None if the
    pattern has no 'frames' group or the group did not match.
    Returns None if the stem does not match the pattern at all.
    """
    m = re.match(FILENAME_PATTERN, stem)
    if m is None:
        return None
    channel = m.group("channel")
    video_id = m.group("id")
    try:
        frames = int(m.group("frames"))
    except IndexError:
        frames = None
    return channel, video_id, frames


def _get_frame_count(fpath: Path) -> int:
    """Returns the number of frames in the file using a lazy HyperSpy load."""
    sig = hs.load(str(fpath), lazy=True)
    return int(sig.data.shape[0])


def _format_entry(video_id: str, frame_count: int, id_col_width: int) -> str:
    """Formats a single TOML entry, aligning the '=' signs for readability."""
    key = f'"{video_id}"'
    padding = " " * (id_col_width - len(key))
    return f'{key}{padding} = [[0, {frame_count}]]'


def generate(data_dir: Path, output_path: Path) -> None:
    """Scans data_dir for video files and writes a blacklist TOML to output_path."""

    all_files = sorted(data_dir.glob(f"*{FILE_EXTENSION}"))
    if not all_files:
        print(f"No {FILE_EXTENSION} files found in {data_dir}")
        return

    # Collect (video_id, frame_count) for every recognised video file
    videos: list[tuple[str, int]] = []
    for fpath in all_files:
        parsed = _parse_stem(fpath.stem)
        if parsed is None:
            print(f"  Skipping (unrecognised format): {fpath.name}")
            continue
        channel, video_id, frame_count = parsed
        if PRIMARY_CHANNEL is not None and channel != PRIMARY_CHANNEL:
            print(f"  Skipping (channel '{channel}' ≠ PRIMARY_CHANNEL '{PRIMARY_CHANNEL}'): {fpath.name}")
            continue
        if frame_count is None:  # Not in filename — load the file to count
            print(f"  Reading {fpath.name} …", end=" ", flush=True)
            frame_count = _get_frame_count(fpath)
            print(f"{frame_count} frames  →  id = \"{video_id}\"")
        else:
            print(f"  {fpath.name}  →  {frame_count} frames, id = \"{video_id}\"")
        videos.append((video_id, frame_count))

    if not videos:
        print("No valid video files found — nothing written.")
        return

    # Build aligned TOML entries
    id_col_width = max(len(f'"{vid}"') for vid, _ in videos)
    lines = [
        "[videos]  # Define valid regions of each video, using their IDs, with start-stop frame numbers",
        "# Each entry is a list of [start, stop] ranges (stop is exclusive).",
        "# Set a value to 'false' to exclude a video entirely.",
        "",
    ]
    for video_id, frame_count in videos:
        lines.append(_format_entry(video_id, frame_count, id_col_width))

    lines += [
        "",
        "[images]  # Excluded images by integer ID",
        "blacklist = []",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"\nBlacklist written to: {output_path}")
    print("Edit the file to restrict frame ranges or exclude videos before training.")

    # Print all filenames formatted for copy-paste into config.yaml video_filter
    all_filter_files = sorted(
        fpath.name
        for fpath in data_dir.glob(f"*{FILE_EXTENSION}")
        if _parse_stem(fpath.stem) is not None
    )
    filter_str = ", ".join(f'"{f}"' for f in all_filter_files)
    print(f"\n--- video_filter for config.yaml ---")
    print(f"video_filter: [{filter_str}]")
    print("------------------------------------")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a starter blacklist.toml from a directory of raw .dm4 video files."
    )
    parser.add_argument(
        "--data_dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Directory containing raw .dm4 files (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Path to write the generated blacklist TOML (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Data directory not found: {args.data_dir}")
        raise SystemExit(1)

    if args.output.exists():
        answer = input(f"{args.output} already exists. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            raise SystemExit(0)

    print(f"Scanning {args.data_dir} …\n")
    generate(args.data_dir, args.output)


if __name__ == "__main__":
    main()
