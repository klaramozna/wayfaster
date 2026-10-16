#!/usr/bin/env python3
"""Render WayFASTER inference .npz files as local PNG images."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render predicted traversability maps saved by infer.py."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing trav_map_*.npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG files; defaults to input_dir/png.",
    )
    parser.add_argument(
        "--max-maps",
        type=int,
        default=-1,
        help="Maximum number of maps to render; <= 0 renders all.",
    )
    return parser.parse_args()


def render_map(input_path: Path, output_path: Path) -> None:
    with np.load(input_path) as data:
        trav = data["trav"]

    if trav.ndim != 3 or trav.shape[0] < 2:
        raise ValueError(
            "Expected trav with shape (2, X, Y), got {} in {}".format(
                trav.shape, input_path
            )
        )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for channel, title in enumerate(("mu", "nu")):
        image = axes[channel].imshow(
            trav[channel].T,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="equal",
        )
        axes[channel].set_title(title)
        axes[channel].set_xlabel("y cell")
        axes[channel].set_ylabel("x cell")
        figure.colorbar(image, ax=axes[channel], fraction=0.046, pad=0.04)

    figure.suptitle(input_path.stem)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir / "png"
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(args.input_dir.glob("trav_map_*.npz"))
    if args.max_maps > 0:
        inputs = inputs[: args.max_maps]
    if not inputs:
        raise SystemExit("No trav_map_*.npz files found in {}".format(args.input_dir))

    for input_path in inputs:
        output_path = output_dir / (input_path.stem + ".png")
        render_map(input_path, output_path)
        print("Saved {}".format(output_path))


if __name__ == "__main__":
    main()
