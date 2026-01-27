#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""summarize_ablation.py

Aggregate per-run GT metrics from the ablation suite and produce
report-friendly tables + plots (effects vs baseline).

Expected folder structure (from experiments/ablation_suite.py):
  experiments_out/<condition>/seed_XXX/causal_out/metrics_summary.json

Usage:
  python -m experiments.summarize_ablation --experiments_out experiments_out

Outputs (written to experiments_out/summary):
  - metrics_all_runs.csv
  - effects_vs_baseline.csv
  - effects_forest.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


KEY_METRICS = [
    "min_interdrone_distance_min",
    "collision_episodes",
    "collision_steps_frac",
    "swarm_formation_error_mean",
    "swarm_formation_error_p95",
    "swarm_formation_error_max",
]

def _collect_runs(root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for cond_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cond = cond_dir.name
        for seed_dir in sorted(cond_dir.glob("seed_*")):
            seed = seed_dir.name.replace("seed_", "")
            metrics_path = seed_dir / "causal_out" / "metrics_summary.json"
            if not metrics_path.exists():
                continue
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            row = {"condition": cond, "seed": seed}
            for k in KEY_METRICS:
                row[k] = metrics.get(k, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)

def _effects_vs_baseline(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    if df.empty:
        return df
    out_rows = []
    base = df[df["condition"] == baseline]
    if base.empty:
        raise ValueError(f"Baseline condition '{baseline}' not found in data. Available: {sorted(df['condition'].unique())}")
    base_mean = base[KEY_METRICS].mean(numeric_only=True)
    base_std = base[KEY_METRICS].std(numeric_only=True)

    for cond in sorted(df["condition"].unique()):
        g = df[df["condition"] == cond]
        mean = g[KEY_METRICS].mean(numeric_only=True)
        std = g[KEY_METRICS].std(numeric_only=True)
        for m in KEY_METRICS:
            out_rows.append({
                "metric": m,
                "condition": cond,
                "mean": float(mean[m]) if m in mean else np.nan,
                "std": float(std[m]) if m in std else np.nan,
                "diff_vs_baseline": float(mean[m] - base_mean[m]),
                "baseline_mean": float(base_mean[m]),
                "baseline_std": float(base_std[m]),
                "n_runs": int(len(g)),
            })
    return pd.DataFrame(out_rows)

def _plot_forest(eff: pd.DataFrame, out_path: Path, baseline: str) -> None:
    # One figure with multiple panels (one per metric), but keep it readable: max 6 metrics.
    metrics = KEY_METRICS[:6]
    conditions = [c for c in sorted(eff["condition"].unique()) if c != baseline]
    if not conditions:
        return

    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, max(3.5, 1.6 * len(metrics))), sharex=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        sub = eff[(eff["metric"] == m) & (eff["condition"] != baseline)].copy()
        sub = sub.set_index("condition").loc[conditions].reset_index()
        y = np.arange(len(conditions))
        x = sub["diff_vs_baseline"].to_numpy(dtype=float)
        # naive CI using std/sqrt(n), good enough for quick reporting
        se = (sub["std"].to_numpy(dtype=float) / np.sqrt(np.maximum(sub["n_runs"].to_numpy(dtype=float), 1.0)))
        ax.errorbar(x, y, xerr=1.96 * se, fmt="o")
        ax.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(conditions, fontsize=9)
        ax.set_title(f"Effect vs {baseline}  —  {m}", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)

def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize ablation suite results (effects vs baseline).")
    ap.add_argument("--experiments_out", type=str, default="experiments_out", help="Root folder from ablation_suite.")
    ap.add_argument("--baseline", type=str, default="baseline", help="Baseline condition name.")
    args = ap.parse_args()

    root = Path(args.experiments_out)
    df = _collect_runs(root)
    out_dir = root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "metrics_all_runs.csv", index=False)

    if df.empty:
        print("[summarize] No runs found. Did you run --run_causal_analysis with --mode lite?")
        return 1

    eff = _effects_vs_baseline(df, baseline=args.baseline)
    eff.to_csv(out_dir / "effects_vs_baseline.csv", index=False)

    _plot_forest(eff, out_dir / "effects_forest.png", baseline=args.baseline)

    print(f"[summarize] Wrote: {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
