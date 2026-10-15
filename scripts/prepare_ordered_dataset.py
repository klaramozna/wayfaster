#!/usr/bin/env python3
"""Create chronological WayFASTER train and validation datasets.

The input must be one complete recording containing states.csv, images.csv,
and the referenced image and depth files. Existing output directories are
never overwritten unless --force is supplied.
"""

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


Row = Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild chronological WayFASTER dataset splits from one recording."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Complete recording directory containing states.csv and images.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for output split directories; defaults to source's parent.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of chronologically sorted states assigned to training (default: 0.8).",
    )
    parser.add_argument(
        "--train-name",
        default="data_train_ordered",
        help="Name of the training output directory.",
    )
    parser.add_argument(
        "--valid-name",
        default="data_valid_ordered",
        help="Name of the validation output directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace output directories if they already exist.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> Tuple[List[str], List[Row]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header: {}".format(path))
        return list(reader.fieldnames), list(reader)


def timestamp(row: Row, path: Path) -> float:
    try:
        return float(row["timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid timestamp in {}: {}".format(path, row)) from exc


def sorted_rows(rows: Sequence[Row], path: Path) -> List[Row]:
    if "timestamp" not in rows[0] if rows else True:
        raise ValueError("CSV is missing a timestamp column: {}".format(path))
    return sorted(rows, key=lambda row: timestamp(row, path))


def ensure_columns(rows: Sequence[Row], columns: Sequence[str], path: Path) -> None:
    if not rows:
        raise ValueError("CSV is empty: {}".format(path))
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError("{} is missing columns: {}".format(path, ", ".join(missing)))


def relative_media_path(value: str, csv_path: Path) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Media path must be relative and contained by the recording: {}".format(csv_path))
    return relative


def copy_media(rows: Sequence[Row], source: Path, destination: Path, csv_path: Path) -> None:
    copied = set()
    for row in rows:
        for column in ("image", "depth"):
            relative = relative_media_path(row[column], csv_path)
            source_file = source / relative
            destination_file = destination / relative
            if relative in copied:
                continue
            if not source_file.is_file():
                raise FileNotFoundError("Referenced media file does not exist: {}".format(source_file))
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination_file)
            copied.add(relative)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_sorted(rows: Sequence[Row], path: Path) -> None:
    values = [timestamp(row, path) for row in rows]
    if any(current < previous for previous, current in zip(values, values[1:])):
        raise ValueError("Output timestamps are not sorted: {}".format(path))


def build_split(
    name: str,
    states: Sequence[Row],
    images: Sequence[Row],
    state_fields: Sequence[str],
    image_fields: Sequence[str],
    source: Path,
    output_root: Path,
    recording_name: str,
) -> None:
    split_root = output_root / name
    recording = split_root / recording_name
    recording.mkdir(parents=True)
    (recording / "image").mkdir()
    (recording / "depth").mkdir()

    write_csv(recording / "states.csv", state_fields, states)
    write_csv(recording / "images.csv", image_fields, images)
    copy_media(images, source, recording, source / "images.csv")
    write_csv(
        split_root / "rosbags.csv",
        ["rosbags"],
        [{"rosbags": recording_name}],
    )

    validate_sorted(states, recording / "states.csv")
    validate_sorted(images, recording / "images.csv")

    print("{}: states={} images={}".format(name, len(states), len(images)))


def prepare_dataset(args: argparse.Namespace) -> None:
    source = args.source.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError("Source recording directory does not exist: {}".format(source))

    states_path = source / "states.csv"
    images_path = source / "images.csv"
    state_fields, states = read_csv(states_path)
    image_fields, images = read_csv(images_path)
    ensure_columns(states, ("timestamp",), states_path)
    ensure_columns(images, ("timestamp", "image", "depth"), images_path)

    states = sorted_rows(states, states_path)
    images = sorted_rows(images, images_path)
    if len(states) < 2:
        raise ValueError("At least two state rows are required to create a split.")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")

    split_index = math.ceil(len(states) * args.train_ratio)
    boundary = timestamp(states[split_index], states_path) if split_index < len(states) else None
    train_states = states[:split_index]
    valid_states = states[split_index:]

    if boundary is None:
        raise ValueError("The validation split has no timestamp boundary.")
    train_images = [row for row in images if timestamp(row, images_path) < boundary]
    valid_images = [row for row in images if timestamp(row, images_path) >= boundary]

    output_root = (args.output_root or source.parent).expanduser().resolve()
    destinations = [output_root / args.train_name, output_root / args.valid_name]
    if source in [destination.resolve() for destination in destinations if destination.exists()]:
        raise ValueError("An output directory cannot be the source recording.")
    existing = [path for path in destinations if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Output already exists; use --force to replace: {}".format(
                ", ".join(str(path) for path in existing)
            )
        )
    for path in existing:
        shutil.rmtree(path)

    output_root.mkdir(parents=True, exist_ok=True)
    build_split(
        args.train_name,
        train_states,
        train_images,
        state_fields,
        image_fields,
        source,
        output_root,
        source.name,
    )
    build_split(
        args.valid_name,
        valid_states,
        valid_images,
        state_fields,
        image_fields,
        source,
        output_root,
        source.name,
    )
    print("Boundary timestamp: {}".format(boundary))
    print("Output root: {}".format(output_root))


def main() -> None:
    args = parse_args()
    try:
        prepare_dataset(args)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit("error: {}".format(exc))


if __name__ == "__main__":
    main()
