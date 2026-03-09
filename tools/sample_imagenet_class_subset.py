#!/usr/bin/env python3
"""Sample ImageNet classes without replacement and write them to a txt file."""

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample classes from imagenet_classes.txt without replacement (reproducible)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/prompt_class_lists/imagenet_classes.txt"),
        help="Path to source ImageNet classes txt (one class per line).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to sampled classes txt output. Default: data/prompt_class_lists/imagenet_classes_sample_<n>_seed<seed>.txt",
    )
    parser.add_argument(
        "--indices_output",
        type=Path,
        default=None,
        help="Optional output file for sampled 0-based class indices. Default: data/prompt_class_lists/imagenet_indices_sample_<n>_seed<seed>.txt",
    )
    parser.add_argument("--n", type=int, default=41, help="Number of classes to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser.parse_args()


def read_classes(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    classes = [line for line in lines if line]
    if not classes:
        raise ValueError(f"No classes found in input file: {path}")
    return classes


def main() -> None:
    args = parse_args()
    classes = read_classes(args.input)

    if args.n < 1:
        raise ValueError("--n must be >= 1")
    if args.n > len(classes):
        raise ValueError(f"Requested n={args.n}, but only {len(classes)} classes are available")

    rng = random.Random(args.seed)
    sampled_indices = rng.sample(range(len(classes)), args.n)
    sampled_classes = [classes[i] for i in sampled_indices]

    if args.output is None:
        args.output = Path(f"data/prompt_class_lists/imagenet_classes_sample_{args.n}_seed{args.seed}.txt")
    if args.indices_output is None:
        args.indices_output = Path(f"data/prompt_class_lists/imagenet_indices_sample_{args.n}_seed{args.seed}.txt")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sampled_classes) + "\n", encoding="utf-8")

    if args.indices_output:
        args.indices_output.parent.mkdir(parents=True, exist_ok=True)
        args.indices_output.write_text(",".join(str(i) for i in sampled_indices) + "\n", encoding="utf-8")

    print(f"Wrote {len(sampled_classes)} sampled classes to {args.output}")
    if args.indices_output:
        print(f"Wrote sampled indices to {args.indices_output}")
    print(f"seed={args.seed}")


if __name__ == "__main__":
    main()
