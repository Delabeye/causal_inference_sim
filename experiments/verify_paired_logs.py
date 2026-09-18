#!/usr/bin/env python3
"""Verify that two seeded UAV runs are identical before an intervention.

The comparison is intentionally strict by default. ``run_id`` is the only
ignored CSV field; every recorded numeric and categorical value must otherwise
match at the same logged simulation time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_IGNORED_COLUMNS = ("run_id",)


def _drone_paths(log_dir: Path, run_id: int) -> dict[str, Path]:
    prefix = f"run_{int(run_id)}_"
    paths = {}
    for path in sorted(log_dir.glob(f"{prefix}drone_*.csv")):
        drone_name = path.stem[len(prefix) :]
        paths[drone_name] = path
    return paths


def _first_difference(
    left: pd.Series,
    right: pd.Series,
    *,
    atol: float,
) -> tuple[int, Any, Any] | None:
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        left_values = left.to_numpy(dtype=float)
        right_values = right.to_numpy(dtype=float)
        same = np.isclose(
            left_values,
            right_values,
            rtol=0.0,
            atol=float(atol),
            equal_nan=True,
        )
    else:
        sentinel = "<PAIRED_LOG_NA>"
        left_values = left.fillna(sentinel).astype(str).to_numpy()
        right_values = right.fillna(sentinel).astype(str).to_numpy()
        same = left_values == right_values
    differing = np.flatnonzero(~same)
    if differing.size == 0:
        return None
    index = int(differing[0])
    return index, left_values[index], right_values[index]


def _compare_ground_truth_npz(
    root: Path,
    baseline_run_id: int,
    comparison_run_id: int,
    *,
    before_time: float | None,
    atol: float,
) -> dict[str, Any]:
    left_path = root / f"run_{int(baseline_run_id)}_ground_truth_matrices.npz"
    right_path = root / f"run_{int(comparison_run_id)}_ground_truth_matrices.npz"
    if not left_path.exists() and not right_path.exists():
        return {"checked": False, "passed": True, "reason": "both files absent"}
    if not left_path.exists() or not right_path.exists():
        return {
            "checked": True,
            "passed": False,
            "error": "ground-truth NPZ missing in one run",
            "baseline_exists": left_path.exists(),
            "comparison_exists": right_path.exists(),
        }

    with np.load(left_path) as left, np.load(right_path) as right:
        if set(left.files) != set(right.files):
            return {
                "checked": True,
                "passed": False,
                "error": "ground-truth channel mismatch",
                "baseline_channels": sorted(left.files),
                "comparison_channels": sorted(right.files),
            }
        left_times = np.asarray(left["times"], dtype=float)
        right_times = np.asarray(right["times"], dtype=float)
        left_mask = (
            np.ones(left_times.shape, dtype=bool)
            if before_time is None
            else left_times < float(before_time)
        )
        right_mask = (
            np.ones(right_times.shape, dtype=bool)
            if before_time is None
            else right_times < float(before_time)
        )
        result: dict[str, Any] = {
            "checked": True,
            "passed": True,
            "baseline_rows": int(left_mask.sum()),
            "comparison_rows": int(right_mask.sum()),
        }
        for channel in left.files:
            left_values = left[channel]
            right_values = right[channel]
            if left_values.ndim and left_values.shape[0] == left_times.size:
                left_values = left_values[left_mask]
            if right_values.ndim and right_values.shape[0] == right_times.size:
                right_values = right_values[right_mask]
            if left_values.shape != right_values.shape:
                result.update(
                    passed=False,
                    error="ground-truth shape mismatch",
                    channel=channel,
                    baseline_shape=list(left_values.shape),
                    comparison_shape=list(right_values.shape),
                )
                return result
            if np.issubdtype(left_values.dtype, np.number):
                same = np.isclose(
                    left_values,
                    right_values,
                    rtol=0.0,
                    atol=float(atol),
                    equal_nan=True,
                )
            else:
                same = left_values == right_values
            if bool(np.all(same)):
                continue
            flat_index = int(np.flatnonzero(~same)[0])
            index = [
                int(value)
                for value in np.unravel_index(flat_index, left_values.shape)
            ]
            left_value = np.ravel(left_values)[flat_index]
            right_value = np.ravel(right_values)[flat_index]
            result.update(
                passed=False,
                error="ground-truth value mismatch",
                first_difference={
                    "channel": channel,
                    "index": index,
                    "baseline": left_value.item()
                    if isinstance(left_value, np.generic)
                    else left_value,
                    "comparison": right_value.item()
                    if isinstance(right_value, np.generic)
                    else right_value,
                },
            )
            return result
        return result


def compare_run_pair(
    log_dir: Path | str,
    baseline_run_id: int,
    comparison_run_id: int,
    *,
    before_time: float | None = None,
    atol: float = 0.0,
    ignored_columns: tuple[str, ...] = DEFAULT_IGNORED_COLUMNS,
) -> dict[str, Any]:
    """Return an auditable equality report for all matching UAV CSV logs."""
    root = Path(log_dir)
    baseline_paths = _drone_paths(root, baseline_run_id)
    comparison_paths = _drone_paths(root, comparison_run_id)
    names = sorted(set(baseline_paths) | set(comparison_paths))
    report: dict[str, Any] = {
        "baseline_run_id": int(baseline_run_id),
        "comparison_run_id": int(comparison_run_id),
        "before_time": None if before_time is None else float(before_time),
        "atol": float(atol),
        "ignored_columns": list(ignored_columns),
        "passed": True,
        "drones": {},
    }
    if not names:
        report["passed"] = False
        report["error"] = f"No UAV CSV logs found in {root}"
        return report

    for name in names:
        drone_report: dict[str, Any] = {"passed": True}
        report["drones"][name] = drone_report
        if name not in baseline_paths or name not in comparison_paths:
            drone_report.update(
                passed=False,
                error="missing log in one run",
                baseline_exists=name in baseline_paths,
                comparison_exists=name in comparison_paths,
            )
            report["passed"] = False
            continue

        baseline = pd.read_csv(baseline_paths[name])
        comparison = pd.read_csv(comparison_paths[name])
        if before_time is not None:
            baseline = baseline[baseline["time"] < float(before_time)].reset_index(drop=True)
            comparison = comparison[comparison["time"] < float(before_time)].reset_index(drop=True)

        drone_report["baseline_rows"] = int(len(baseline))
        drone_report["comparison_rows"] = int(len(comparison))
        if len(baseline) != len(comparison):
            drone_report.update(
                passed=False,
                error="row count mismatch",
            )
            report["passed"] = False
            continue

        baseline_columns = [
            column for column in baseline.columns if column not in ignored_columns
        ]
        comparison_columns = [
            column for column in comparison.columns if column not in ignored_columns
        ]
        if baseline_columns != comparison_columns:
            drone_report.update(
                passed=False,
                error="column mismatch",
                baseline_columns=baseline_columns,
                comparison_columns=comparison_columns,
            )
            report["passed"] = False
            continue

        for column in baseline_columns:
            difference = _first_difference(
                baseline[column], comparison[column], atol=atol
            )
            if difference is None:
                continue
            row_index, baseline_value, comparison_value = difference
            drone_report.update(
                passed=False,
                error="value mismatch",
                first_difference={
                    "row": row_index,
                    "time": float(baseline.iloc[row_index]["time"]),
                    "column": column,
                    "baseline": baseline_value.item()
                    if isinstance(baseline_value, np.generic)
                    else baseline_value,
                    "comparison": comparison_value.item()
                    if isinstance(comparison_value, np.generic)
                    else comparison_value,
                },
            )
            report["passed"] = False
            break

    ground_truth = _compare_ground_truth_npz(
        root,
        baseline_run_id,
        comparison_run_id,
        before_time=before_time,
        atol=atol,
    )
    report["ground_truth_matrices"] = ground_truth
    report["passed"] = bool(report["passed"] and ground_truth["passed"])

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that paired UAV logs match before an intervention."
    )
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--baseline-run", type=int, required=True)
    parser.add_argument("--comparison-run", type=int, required=True)
    parser.add_argument("--before-time", type=float, default=None)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = compare_run_pair(
        args.logs_dir,
        args.baseline_run,
        args.comparison_run,
        before_time=args.before_time,
        atol=args.atol,
    )
    rendered = json.dumps(report, indent=2, allow_nan=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
