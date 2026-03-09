#!/usr/bin/env python3
"""Prepare first-N / remaining splits for sampled ImageNet class+index files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError(f"No non-empty lines in: {path}")
    return lines


def _read_indices(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No indices found in: {path}")
    out: list[int] = []
    for tok in text.replace("\n", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError(f"No parsed indices in: {path}")
    return out


def _write_lines(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + "\n", encoding="utf-8")


def _write_indices(path: Path, items: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(str(x) for x in items) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split sampled 41-class files into profile/eval subsets."
    )
    parser.add_argument(
        "--classes_file",
        type=Path,
        default=Path("data/prompt_class_lists/imagenet_classes_sample_41_seed0.txt"),
        help="Input class names (one per line).",
    )
    parser.add_argument(
        "--indices_file",
        type=Path,
        default=Path("data/prompt_class_lists/imagenet_indices_sample_41_seed0.txt"),
        help="Input class indices (comma-separated).",
    )
    parser.add_argument(
        "--profile_count",
        type=int,
        default=5,
        help="Number of first entries used for profile collection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = _read_lines(args.classes_file)
    indices = _read_indices(args.indices_file)

    if len(classes) != len(indices):
        raise ValueError(
            f"Count mismatch: {args.classes_file} has {len(classes)} lines, "
            f"{args.indices_file} has {len(indices)} indices"
        )
    if args.profile_count < 1 or args.profile_count >= len(classes):
        raise ValueError(
            f"--profile_count must be in [1, {len(classes)-1}], got {args.profile_count}"
        )

    stem_cls = args.classes_file.stem
    stem_idx = args.indices_file.stem
    out_dir = args.classes_file.parent

    first_cls_path = out_dir / f"{stem_cls}_first{args.profile_count}.txt"
    last_cls_path = out_dir / f"{stem_cls}_last{len(classes) - args.profile_count}.txt"
    first_idx_path = out_dir / f"{stem_idx}_first{args.profile_count}.txt"
    last_idx_path = out_dir / f"{stem_idx}_last{len(indices) - args.profile_count}.txt"
    meta_path = out_dir / f"{stem_cls}_split_meta.json"

    classes_first = classes[: args.profile_count]
    classes_last = classes[args.profile_count :]
    idx_first = indices[: args.profile_count]
    idx_last = indices[args.profile_count :]

    _write_lines(first_cls_path, classes_first)
    _write_lines(last_cls_path, classes_last)
    _write_indices(first_idx_path, idx_first)
    _write_indices(last_idx_path, idx_last)

    meta = {
        "classes_file": str(args.classes_file),
        "indices_file": str(args.indices_file),
        "total_count": len(classes),
        "profile_count": args.profile_count,
        "eval_count": len(classes) - args.profile_count,
        "classes_first_file": str(first_cls_path),
        "classes_last_file": str(last_cls_path),
        "indices_first_file": str(first_idx_path),
        "indices_last_file": str(last_idx_path),
        "classes_first_preview": classes_first[:3],
        "classes_last_preview": classes_last[:3],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {first_cls_path} ({len(classes_first)} lines)")
    print(f"Wrote {last_cls_path} ({len(classes_last)} lines)")
    print(f"Wrote {first_idx_path} ({len(idx_first)} indices)")
    print(f"Wrote {last_idx_path} ({len(idx_last)} indices)")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
