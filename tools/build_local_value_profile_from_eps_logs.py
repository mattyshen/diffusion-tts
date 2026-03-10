#!/usr/bin/env python3
"""Build local-search value diagnostics from eps-greedy JSONL logs.

This tool is meant for first-5 profile-data runs. It outputs per-(timestep, local_iter)
statistics that quantify whether local search is worth it relative to a global baseline.

Definitions for each complete unit u and timestep t:
  G_avg(u,t) = average global-candidate score across all local iterations at timestep t
  G_best(u,t) = best global-candidate score across all local iterations at timestep t
  L(u,t,k) = best local-candidate score at local iteration k (NaN if no local candidate at k)
  L_so_far(u,t,k) = max_{j<=k} L(u,t,j)

Derived metrics:
  gain_avg(u,t,k) = L_so_far(u,t,k) - G_avg(u,t)
  gain_best(u,t,k) = L_so_far(u,t,k) - G_best(u,t)
  marginal(u,t,k) = L_so_far(u,t,k) - L_so_far(u,t,k-1), k>=1

Bias normalization per timestep:
  sd_bias(t) = std over all candidates in complete units of (score - final_score)

Normalized variants divide by sd_bias(t).
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
        description="Build local-search value diagnostics from eps-greedy JSONL logs."
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        required=True,
        help="Directory containing JSONL log files (searched recursively).",
    )
    parser.add_argument(
        "--output_curve_csv",
        type=Path,
        required=True,
        help="Output CSV with per-(timestep_idx, local_iter_idx) aggregated stats.",
    )
    parser.add_argument(
        "--output_timestep_csv",
        type=Path,
        default=None,
        help="Optional output CSV with per-timestep summary and recommended local steps.",
    )
    parser.add_argument(
        "--output_k_profile",
        type=Path,
        default=None,
        help="Optional txt profile: recommended number of local iterations per timestep.",
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
    parser.add_argument(
        "--alpha_start_z",
        type=float,
        default=0.0,
        help="If z(gain_avg) at k=0 is below this, recommended local steps for t is 0.",
    )
    parser.add_argument(
        "--alpha_marginal_z",
        type=float,
        default=0.0,
        help="Continue local search while z(marginal gain) stays >= this threshold.",
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
    inferred_k: list[int] = []
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
        params = parsed.run_meta.get("params", {})
        if isinstance(params, dict) and params.get("K") is not None:
            try:
                inferred_k.append(int(params["K"]))
            except Exception:
                pass

        for ev in parsed.candidates:
            sample_idx = int(ev.get("sample_idx", 0))
            final_score = final_by_sample.get(sample_idx)
            if final_score is None:
                continue
            score = _safe_float(ev.get("score"))
            timestep_idx = ev.get("timestep_idx")
            local_iter_idx = ev.get("local_iter_idx")
            if score is None or timestep_idx is None or local_iter_idx is None:
                continue
            rows.append(
                {
                    "run_file": str(path),
                    "sample_idx": int(sample_idx),
                    "unit_id": f"{path}::s{sample_idx}",
                    "timestep_idx": int(timestep_idx),
                    "timestep_value": _safe_float(ev.get("timestep_value")),
                    "local_iter_idx": int(local_iter_idx),
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
        "inferred_k_from_meta": sorted(set(inferred_k)),
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


def _se(series: pd.Series) -> float:
    vals = series.astype(float).to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return float("nan")
    return float(np.std(vals, ddof=1) / np.sqrt(vals.size))


def _build_unit_iter_df(df_complete: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_cols = ["unit_id", "run_file", "sample_idx", "timestep_idx"]

    for keys, g in df_complete.groupby(base_cols, dropna=False):
        unit_id, run_file, sample_idx, t_idx = keys
        g_global_all = g[g["is_global_candidate"]]
        g_local_all = g[~g["is_global_candidate"]]
        if len(g_global_all) == 0 or len(g_local_all) == 0:
            continue

        avg_global_t = float(g_global_all["score"].mean())
        best_global_t = float(g_global_all["score"].max())
        timestep_value = (
            float(g["timestep_value"].dropna().median()) if g["timestep_value"].notna().any() else np.nan
        )

        k_df = (
            g.loc[~g["is_global_candidate"], ["local_iter_idx", "score"]]
            .groupby("local_iter_idx", as_index=False)["score"]
            .max()
            .rename(columns={"score": "best_local_k"})
            .sort_values("local_iter_idx")
            .reset_index(drop=True)
        )
        if k_df.empty:
            continue

        k_df["best_local_so_far"] = k_df["best_local_k"].cummax()
        k_df["gain_vs_avg_global"] = k_df["best_local_so_far"] - avg_global_t
        k_df["gain_vs_best_global"] = k_df["best_local_so_far"] - best_global_t
        k_df["marginal_gain"] = k_df["best_local_so_far"].diff()

        for _, row in k_df.iterrows():
            rows.append(
                {
                    "unit_id": unit_id,
                    "run_file": run_file,
                    "sample_idx": int(sample_idx),
                    "timestep_idx": int(t_idx),
                    "timestep_value": timestep_value,
                    "local_iter_idx": int(row["local_iter_idx"]),
                    "avg_global_t": avg_global_t,
                    "best_global_t": best_global_t,
                    "best_local_k": _safe_float(row["best_local_k"]),
                    "best_local_so_far": _safe_float(row["best_local_so_far"]),
                    "gain_vs_avg_global": _safe_float(row["gain_vs_avg_global"]),
                    "gain_vs_best_global": _safe_float(row["gain_vs_best_global"]),
                    "marginal_gain": _safe_float(row["marginal_gain"]),
                }
            )

    return pd.DataFrame(rows)


def _build_curve(unit_iter_df: pd.DataFrame, sd_bias_by_t: pd.Series) -> pd.DataFrame:
    if unit_iter_df.empty:
        return pd.DataFrame()

    work = unit_iter_df.copy()
    work["sd_bias"] = work["timestep_idx"].map(sd_bias_by_t)

    work["z_gain_vs_avg_global"] = work["gain_vs_avg_global"] / work["sd_bias"].replace(0.0, np.nan)
    work["z_gain_vs_best_global"] = work["gain_vs_best_global"] / work["sd_bias"].replace(0.0, np.nan)
    work["z_marginal_gain"] = work["marginal_gain"] / work["sd_bias"].replace(0.0, np.nan)

    group_cols = ["timestep_idx", "local_iter_idx"]
    curve = (
        work.groupby(group_cols, as_index=False)
        .agg(
            timestep_value=("timestep_value", "median"),
            n_units=("unit_id", "nunique"),
            sd_bias=("sd_bias", "median"),
            avg_global_mean=("avg_global_t", "mean"),
            best_global_mean=("best_global_t", "mean"),
            best_local_k_mean=("best_local_k", "mean"),
            best_local_so_far_mean=("best_local_so_far", "mean"),
            gain_vs_avg_global_mean=("gain_vs_avg_global", "mean"),
            gain_vs_best_global_mean=("gain_vs_best_global", "mean"),
            marginal_gain_mean=("marginal_gain", "mean"),
            z_gain_vs_avg_global_mean=("z_gain_vs_avg_global", "mean"),
            z_gain_vs_best_global_mean=("z_gain_vs_best_global", "mean"),
            z_marginal_gain_mean=("z_marginal_gain", "mean"),
            p_gain_vs_avg_global_pos=("gain_vs_avg_global", lambda s: float(np.nanmean(np.asarray(s) > 0.0))),
            p_marginal_pos=("marginal_gain", lambda s: float(np.nanmean(np.asarray(s) > 0.0))),
        )
        .sort_values(["timestep_idx", "local_iter_idx"])
        .reset_index(drop=True)
    )

    se_df = (
        work.groupby(group_cols, as_index=False)
        .agg(
            gain_vs_avg_global_se=("gain_vs_avg_global", _se),
            gain_vs_best_global_se=("gain_vs_best_global", _se),
            marginal_gain_se=("marginal_gain", _se),
            z_gain_vs_avg_global_se=("z_gain_vs_avg_global", _se),
            z_gain_vs_best_global_se=("z_gain_vs_best_global", _se),
            z_marginal_gain_se=("z_marginal_gain", _se),
        )
        .sort_values(["timestep_idx", "local_iter_idx"])
        .reset_index(drop=True)
    )

    return curve.merge(se_df, on=["timestep_idx", "local_iter_idx"], how="left")


def _build_timestep_summary(
    curve: pd.DataFrame,
    alpha_start_z: float,
    alpha_marginal_z: float,
) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for t_idx, g in curve.groupby("timestep_idx", dropna=False):
        g = g.sort_values("local_iter_idx").reset_index(drop=True)
        k_vals = g["local_iter_idx"].astype(int).tolist()
        z_start = _safe_float(g.loc[g["local_iter_idx"] == min(k_vals), "z_gain_vs_avg_global_mean"].iloc[0])

        rec_steps = 0
        if z_start is not None and z_start >= alpha_start_z:
            rec_steps = 1
            for _, row in g.iterrows():
                k = int(row["local_iter_idx"])
                if k == min(k_vals):
                    continue
                z_m = _safe_float(row["z_marginal_gain_mean"])
                if z_m is None or z_m < alpha_marginal_z:
                    break
                rec_steps = k + 1

        final_gain = _safe_float(g["gain_vs_avg_global_mean"].iloc[-1])
        k_90pct = np.nan
        if final_gain is not None and np.isfinite(final_gain) and final_gain > 0:
            target = 0.9 * final_gain
            reached = g[g["gain_vs_avg_global_mean"] >= target]
            if len(reached) > 0:
                k_90pct = int(reached.iloc[0]["local_iter_idx"]) + 1

        rows.append(
            {
                "timestep_idx": int(t_idx),
                "timestep_value": _safe_float(g["timestep_value"].median()),
                "n_units": int(g["n_units"].max()),
                "max_local_iter_idx": int(g["local_iter_idx"].max()),
                "z_start_gain_vs_avg_global": z_start,
                "z_end_gain_vs_avg_global": _safe_float(g["z_gain_vs_avg_global_mean"].iloc[-1]),
                "z_end_gain_vs_best_global": _safe_float(g["z_gain_vs_best_global_mean"].iloc[-1]),
                "z_end_marginal": _safe_float(g["z_marginal_gain_mean"].iloc[-1]),
                "recommended_local_steps": int(rec_steps),
                "k_90pct_gain_steps": k_90pct,
                "gain_vs_avg_global_end": final_gain,
            }
        )

    return pd.DataFrame(rows).sort_values("timestep_idx").reset_index(drop=True)


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

    df_complete["bias"] = df_complete["score"] - df_complete["final_score"]
    sd_bias_by_t = df_complete.groupby("timestep_idx")["bias"].std()

    unit_iter_df = _build_unit_iter_df(df_complete)
    if unit_iter_df.empty:
        raise ValueError("No unit-level local/global metrics available.")

    curve = _build_curve(unit_iter_df, sd_bias_by_t=sd_bias_by_t)
    if curve.empty:
        raise ValueError("No aggregated curve rows available.")

    ts_summary = _build_timestep_summary(
        curve,
        alpha_start_z=args.alpha_start_z,
        alpha_marginal_z=args.alpha_marginal_z,
    )

    args.output_curve_csv.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(args.output_curve_csv, index=False)

    if args.output_timestep_csv is not None:
        args.output_timestep_csv.parent.mkdir(parents=True, exist_ok=True)
        ts_summary.to_csv(args.output_timestep_csv, index=False)

    if args.output_k_profile is not None:
        if ts_summary.empty:
            raise ValueError("Cannot write k profile: timestep summary is empty.")
        full = pd.DataFrame({"timestep_idx": np.arange(expected_steps, dtype=int)})
        merged = full.merge(ts_summary[["timestep_idx", "recommended_local_steps"]], on="timestep_idx", how="left")
        k_profile = merged["recommended_local_steps"].fillna(0).astype(int).to_numpy()
        args.output_k_profile.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.output_k_profile, k_profile, fmt="%d")

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
            "num_curve_rows": int(len(curve)),
            "curve_csv_path": str(args.output_curve_csv),
            "timestep_csv_path": None if args.output_timestep_csv is None else str(args.output_timestep_csv),
            "k_profile_path": None if args.output_k_profile is None else str(args.output_k_profile),
            "backend_seen": meta.get("backend_seen", []),
            "scorer_seen": meta.get("scorer_seen", []),
            "alpha_start_z": float(args.alpha_start_z),
            "alpha_marginal_z": float(args.alpha_marginal_z),
        }
        args.output_meta_json.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    print(f"Wrote local-value curve csv: {args.output_curve_csv}")
    if args.output_timestep_csv is not None:
        print(f"Wrote local-value timestep csv: {args.output_timestep_csv}")
    if args.output_k_profile is not None:
        print(f"Wrote recommended-k profile txt: {args.output_k_profile}")
    if args.output_meta_json is not None:
        print(f"Wrote meta json: {args.output_meta_json}")
    print(
        f"Complete units: {len(complete_units)} | Steps: {expected_steps} | "
        f"Selected logs: {len(selected_paths)}/{len(log_paths)}"
    )


if __name__ == "__main__":
    main()
