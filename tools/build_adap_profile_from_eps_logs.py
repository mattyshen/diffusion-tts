#!/usr/bin/env python3
"""Build per-(model,reward) adaptive profile from eps-greedy logs.

Statistic per timestep t:
    z_t = (E_u[best_local(u,t) - avg_global(u,t)]) / SD_all_candidates(score_hat - final_score)

where u indexes complete runs/samples only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ParsedLog:
    path: Path
    run_meta: dict[str, Any] | None
    run_end: dict[str, Any] | None
    candidates: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build adaptive z-profile from eps-greedy JSONL logs."
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        required=True,
        help="Directory containing JSONL log files (searched recursively).",
    )
    parser.add_argument(
        "--output_profile",
        type=Path,
        required=True,
        help="Output txt file (one z value per line) for adap-eps-greedy.",
    )
    parser.add_argument(
        "--output_curve_csv",
        type=Path,
        default=None,
        help="Optional CSV with per-timestep intermediate stats.",
    )
    parser.add_argument(
        "--output_meta_json",
        type=Path,
        default=None,
        help="Optional JSON summary metadata.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="eps_greedy",
        help="Method to keep from run_meta.",
    )
    parser.add_argument(
        "--expected_steps",
        type=int,
        default=None,
        help="Optional fixed timestep count. If omitted, inferred from logs.",
    )
    parser.add_argument(
        "--latest_only_by_key",
        action="store_true",
        help="If set, keep only latest JSONL per logical run key to avoid duplicates.",
    )
    return parser.parse_args()


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(y):
        return None
    return y


def _read_jsonl(path: Path) -> ParsedLog:
    run_meta = None
    run_end = None
    candidates: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            et = event.get("event_type")
            if et == "run_meta":
                run_meta = event.get("run_meta", {})
            elif et == "run_end":
                run_end = event
            elif et == "candidate_eval":
                candidates.append(event)
    return ParsedLog(path=path, run_meta=run_meta, run_end=run_end, candidates=candidates)


def _run_key(meta: dict[str, Any] | None, path: Path) -> tuple[Any, ...]:
    if meta is None:
        return ("unknown", str(path))
    backend = str(meta.get("backend", ""))
    method = str(meta.get("method", ""))
    scorer = str(meta.get("scorer", ""))
    if backend == "sd":
        return ("sd", method, scorer, int(meta.get("prompt_index", -1)))
    if backend == "edm":
        class_indices = meta.get("class_indices", [])
        if isinstance(class_indices, list):
            class_indices = tuple(class_indices)
        return ("edm", method, scorer, class_indices)
    return (backend, method, scorer, str(path))


def _choose_files(log_paths: list[Path], latest_only_by_key: bool) -> list[Path]:
    if not latest_only_by_key:
        return sorted(log_paths)
    key_to_path: dict[tuple[Any, ...], Path] = {}
    for p in sorted(log_paths):
        parsed = _read_jsonl(p)
        key = _run_key(parsed.run_meta, p)
        prev = key_to_path.get(key)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            key_to_path[key] = p
    return sorted(key_to_path.values())


def _final_score_by_sample(parsed: ParsedLog) -> dict[int, float]:
    out: dict[int, float] = {}
    if parsed.run_end is None:
        return out
    per_sample = parsed.run_end.get("per_sample_scores")
    if isinstance(per_sample, list) and len(per_sample) > 0:
        for i, val in enumerate(per_sample):
            fv = _safe_float(val)
            if fv is not None:
                out[i] = fv
        return out
    best_score = _safe_float(parsed.run_end.get("best_score"))
    if best_score is not None:
        out[0] = best_score
    return out


def _build_df(selected_paths: list[Path], method: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    backend_seen = set()
    scorer_seen = set()
    inferred_steps: list[int] = []
    kept_files = 0

    for path in selected_paths:
        parsed = _read_jsonl(path)
        if parsed.run_meta is None:
            continue
        run_method = str(parsed.run_meta.get("method", "")).lower().replace("-", "_")
        if run_method != method:
            continue
        if len(parsed.candidates) == 0:
            continue
        final_by_sample = _final_score_by_sample(parsed)
        if len(final_by_sample) == 0:
            continue
        kept_files += 1
        backend_seen.add(str(parsed.run_meta.get("backend", "")))
        scorer_seen.add(str(parsed.run_meta.get("scorer", "")))
        if parsed.run_meta.get("num_steps") is not None:
            try:
                inferred_steps.append(int(parsed.run_meta["num_steps"]))
            except Exception:
                pass

        for ev in parsed.candidates:
            sample_idx = int(ev.get("sample_idx", 0))
            final_score = final_by_sample.get(sample_idx)
            if final_score is None:
                continue
            score = _safe_float(ev.get("score"))
            timestep_idx = ev.get("timestep_idx")
            if score is None or timestep_idx is None:
                continue
            rows.append(
                {
                    "run_file": str(path),
                    "sample_idx": int(sample_idx),
                    "unit_id": f"{path}::s{sample_idx}",
                    "timestep_idx": int(timestep_idx),
                    "timestep_value": _safe_float(ev.get("timestep_value")),
                    "local_iter_idx": int(ev.get("local_iter_idx", -1)),
                    "candidate_idx": int(ev.get("candidate_idx", -1)),
                    "is_global_candidate": bool(ev.get("is_global_candidate", False)),
                    "score": float(score),
                    "final_score": float(final_score),
                }
            )

    df = pd.DataFrame(rows)
    meta = {
        "kept_files": kept_files,
        "backend_seen": sorted(backend_seen),
        "scorer_seen": sorted(scorer_seen),
        "num_rows": int(len(df)),
        "inferred_steps_from_meta": sorted(set(inferred_steps)),
    }
    return df, meta


def _infer_expected_steps(df: pd.DataFrame, expected_steps_arg: int | None, meta_steps: list[int]) -> int:
    if expected_steps_arg is not None:
        return int(expected_steps_arg)
    if len(meta_steps) > 0:
        return int(pd.Series(meta_steps).mode().iloc[0])
    counts = df.groupby("unit_id")["timestep_idx"].nunique()
    if len(counts) == 0:
        raise ValueError("No candidate rows after filtering.")
    return int(counts.mode().iloc[0])


def _unit_timestep_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grp_cols = ["unit_id", "run_file", "sample_idx", "timestep_idx"]
    for keys, g in df.groupby(grp_cols, dropna=False):
        unit_id, run_file, sample_idx, t_idx = keys
        g_global = g[g["is_global_candidate"]]
        g_local = g[~g["is_global_candidate"]]
        if len(g_global) == 0 or len(g_local) == 0:
            continue
        avg_global = float(g_global["score"].mean())
        best_local = float(g_local["score"].max())
        rows.append(
            {
                "unit_id": unit_id,
                "run_file": run_file,
                "sample_idx": int(sample_idx),
                "timestep_idx": int(t_idx),
                "timestep_value": float(g["timestep_value"].dropna().median())
                if g["timestep_value"].notna().any()
                else np.nan,
                "avg_global": avg_global,
                "best_local": best_local,
                "local_minus_avg_global": best_local - avg_global,
            }
        )
    return pd.DataFrame(rows)


def _build_curve(df_complete: pd.DataFrame, unit_metrics: pd.DataFrame, expected_steps: int) -> pd.DataFrame:
    reward_curve = (
        unit_metrics.groupby("timestep_idx", as_index=False)
        .agg(
            timestep_value=("timestep_value", "median"),
            avg_global_mean=("avg_global", "mean"),
            best_local_mean=("best_local", "mean"),
            gap_mean=("local_minus_avg_global", "mean"),
            n_units=("unit_id", "nunique"),
        )
        .sort_values("timestep_idx")
    )

    bias_df = df_complete.copy()
    bias_df["bias"] = bias_df["score"] - bias_df["final_score"]
    bias_curve = (
        bias_df.groupby("timestep_idx", as_index=False)
        .agg(
            sd_bias=("bias", "std"),
            n_bias=("bias", "count"),
        )
        .sort_values("timestep_idx")
    )

    curve = reward_curve.merge(bias_curve, on="timestep_idx", how="outer").sort_values("timestep_idx")
    curve["z_raw"] = curve["gap_mean"] / curve["sd_bias"].replace(0.0, np.nan)

    full = pd.DataFrame({"timestep_idx": np.arange(expected_steps, dtype=int)})
    curve = full.merge(curve, on="timestep_idx", how="left")
    curve["z_profile"] = (
        curve["z_raw"]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(limit_direction="both")
        .fillna(0.0)
    )
    return curve


def main() -> None:
    args = parse_args()
    method = args.method.lower().replace("-", "_")

    if not args.log_dir.exists():
        raise FileNotFoundError(f"Log directory does not exist: {args.log_dir}")
    log_paths = sorted(args.log_dir.rglob("*.jsonl"))
    if len(log_paths) == 0:
        raise ValueError(f"No jsonl files found under: {args.log_dir}")

    selected_paths = _choose_files(log_paths, latest_only_by_key=args.latest_only_by_key)
    df, meta = _build_df(selected_paths, method=method)
    if df.empty:
        raise ValueError(
            "No candidate rows after filtering. Check --log_dir, --method, and run_end/final score logging."
        )

    expected_steps = _infer_expected_steps(
        df,
        expected_steps_arg=args.expected_steps,
        meta_steps=meta.get("inferred_steps_from_meta", []),
    )
    timestep_counts = df.groupby("unit_id")["timestep_idx"].nunique()
    complete_units = set(timestep_counts[timestep_counts == expected_steps].index.tolist())
    df_complete = df[df["unit_id"].isin(complete_units)].copy()
    if df_complete.empty:
        raise ValueError(
            f"No complete units found with expected_steps={expected_steps}. "
            f"Available timestep counts: {sorted(set(timestep_counts.tolist()))}"
        )

    unit_metrics = _unit_timestep_metrics(df_complete)
    if unit_metrics.empty:
        raise ValueError("No unit/timestep metrics available (missing global/local candidates).")

    curve = _build_curve(df_complete, unit_metrics, expected_steps=expected_steps)
    profile = curve["z_profile"].to_numpy(dtype=float)
    if np.any(~np.isfinite(profile)):
        raise ValueError("Non-finite values in output profile after filling.")

    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_profile, profile, fmt="%.10f")

    if args.output_curve_csv is not None:
        args.output_curve_csv.parent.mkdir(parents=True, exist_ok=True)
        curve.to_csv(args.output_curve_csv, index=False)

    if args.output_meta_json is not None:
        args.output_meta_json.parent.mkdir(parents=True, exist_ok=True)
        meta_out = {
            "log_dir": str(args.log_dir),
            "selected_files": [str(p) for p in selected_paths],
            "num_jsonl_total": len(log_paths),
            "num_jsonl_selected": len(selected_paths),
            "method": method,
            "expected_steps": expected_steps,
            "num_complete_units": int(len(complete_units)),
            "num_candidate_rows_complete": int(len(df_complete)),
            "profile_path": str(args.output_profile),
            "curve_csv_path": None if args.output_curve_csv is None else str(args.output_curve_csv),
            "backend_seen": meta.get("backend_seen", []),
            "scorer_seen": meta.get("scorer_seen", []),
        }
        args.output_meta_json.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    print(f"Wrote profile: {args.output_profile} ({len(profile)} steps)")
    if args.output_curve_csv is not None:
        print(f"Wrote curve csv: {args.output_curve_csv}")
    if args.output_meta_json is not None:
        print(f"Wrote meta json: {args.output_meta_json}")
    print(
        f"Complete units: {len(complete_units)} | "
        f"Steps: {expected_steps} | "
        f"Selected logs: {len(selected_paths)}/{len(log_paths)}"
    )


if __name__ == "__main__":
    main()
