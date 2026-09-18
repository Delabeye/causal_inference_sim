"""High-level orchestration and audit helpers for simulator datasets."""

from __future__ import annotations

import hashlib
import importlib.metadata
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utilities.config import load_config


MANIFEST_NAME = "dataset_manifest.json"
VERIFICATION_NAME = "dataset_verification.json"


def _read_plan(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _enabled(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _optional_run_id(value: object) -> int | None:
    text = str(value or "").strip().lower()
    if text in {"", "nan", "none"}:
        return None
    return int(float(text))


def _resolve_artifact(value: str | Path, dataset_dir: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    return dataset_dir / path.name


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def summarize_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fork_rows = [row for row in rows if _enabled(row.get("counterfactual_forks_enabled", 0))]
    expected_forks = 0
    for row in fork_rows:
        raw = str(row.get("snapshot_times_s", "")).strip()
        if raw and raw.lower() not in {"nan", "none"}:
            expected_forks += len(json.loads(raw))
    return {
        "planned_runs": len(rows),
        "planned_baseline_runs": len(rows) - len(fork_rows),
        "planned_fork_runs": len(fork_rows),
        "planned_counterfactual_branches": expected_forks,
        "run_ids": [int(row["run_id"]) for row in rows],
        "formations": sorted({str(row.get("formation", "")) for row in rows}),
        "splits": {
            split: sum(str(row.get("split", "")) == split for row in rows)
            for split in ("train", "val", "test")
        },
    }


def write_dataset_manifest(
    dataset_dir: Path,
    *,
    status: str,
    config_path: Path,
    plan_path: Path,
    rows: list[dict[str, Any]],
    execution: dict[str, Any] | None = None,
    verification: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Write one self-contained provenance record for the dataset."""

    dataset_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir.resolve()),
        "command": list(sys.argv),
        "config": {
            "path": str(config_path.resolve()),
            "sha256": _sha256(config_path),
        },
        "plan": {
            "path": str(plan_path.resolve()),
            "sha256": _sha256(plan_path),
            **summarize_plan(rows),
        },
        "software": {
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "pybullet": _package_version("pybullet"),
            "torch": _package_version("torch"),
        },
    }
    if execution is not None:
        payload["execution"] = execution
    if verification is not None:
        payload["verification"] = verification
    if error is not None:
        payload["error"] = error
    path = dataset_dir / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def verify_dataset(
    dataset_dir: Path,
    *,
    plan_path: Path | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    """Verify the artifacts required by the interventional dataset loader."""

    dataset_dir = Path(dataset_dir)
    plan_path = Path(plan_path) if plan_path is not None else dataset_dir / "plan.csv"
    rows = _read_plan(plan_path)
    errors: list[str] = []
    warnings: list[str] = []
    branch_count = 0
    completed_runs = 0
    checked_pairs = 0

    for row in rows:
        run_id = int(row["run_id"])
        summary_path = dataset_dir / f"run_{run_id}_summary.json"
        trace_path = dataset_dir / f"run_{run_id}_learning_trace.npz"
        ground_truth_path = dataset_dir / f"run_{run_id}_ground_truth_matrices.npz"
        if not summary_path.exists():
            errors.append(f"run {run_id}: missing {summary_path.name}")
        else:
            completed_runs += 1
        if not trace_path.exists():
            errors.append(f"run {run_id}: missing {trace_path.name}")
        if not ground_truth_path.exists():
            warnings.append(f"run {run_id}: missing {ground_truth_path.name}")

        paired_baseline = _optional_run_id(row.get("paired_baseline_run_id"))
        if paired_baseline is not None:
            check_path = dataset_dir / f"run_{run_id}_preintervention_check.json"
            if not check_path.exists():
                errors.append(f"run {run_id}: missing {check_path.name}")
            else:
                checked_pairs += 1
                try:
                    pair_report = json.loads(check_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"run {run_id}: invalid pair report: {exc}")
                else:
                    if not bool(pair_report.get("passed", False)):
                        errors.append(f"run {run_id}: pre-intervention check failed")

        if not _enabled(row.get("counterfactual_forks_enabled", 0)):
            continue
        fork_manifest_path = dataset_dir / f"run_{run_id}_counterfactual_forks.json"
        if not fork_manifest_path.exists():
            errors.append(f"run {run_id}: missing {fork_manifest_path.name}")
            continue
        try:
            fork_manifest = json.loads(
                fork_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"run {run_id}: invalid fork manifest: {exc}")
            continue
        forks = list(fork_manifest.get("forks", []))
        raw_times = str(row.get("snapshot_times_s", "")).strip()
        expected = len(json.loads(raw_times)) if raw_times else 0
        if len(forks) != expected:
            errors.append(
                f"run {run_id}: expected {expected} forks, found {len(forks)}"
            )
        branch_count += len(forks)
        for fork in forks:
            for field in ("baseline_trace", "intervention_trace", "event_catalog"):
                value = fork.get(field)
                if not value:
                    errors.append(f"run {run_id}: fork missing '{field}'")
                    continue
                artifact = _resolve_artifact(value, dataset_dir)
                if not artifact.exists():
                    errors.append(f"run {run_id}: missing fork artifact {artifact.name}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir.resolve()),
        "plan_path": str(plan_path.resolve()),
        "passed": not errors,
        "planned_runs": len(rows),
        "completed_runs": completed_runs,
        "checked_pairs": checked_pairs,
        "counterfactual_branches": branch_count,
        "errors": errors,
        "warnings": warnings,
    }
    if write_report:
        (dataset_dir / VERIFICATION_NAME).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def generate_dataset(
    *,
    config_path: Path,
    dataset_dir: Path,
    plan_path: Path | None = None,
    dry_run: bool = False,
    resume: bool = False,
    verify_after: bool = True,
) -> dict[str, Any]:
    """Execute an existing plan and maintain its provenance manifest."""

    dataset_dir = Path(dataset_dir)
    plan_path = Path(plan_path) if plan_path is not None else dataset_dir / "plan.csv"
    config_path = Path(config_path)
    if not plan_path.exists():
        raise FileNotFoundError(
            f"Plan not found: {plan_path}. Run 'python main.py dataset plan' first."
        )
    rows = _read_plan(plan_path)
    base_config = load_config(str(config_path))
    write_dataset_manifest(
        dataset_dir,
        status="validating" if dry_run else "running",
        config_path=config_path,
        plan_path=plan_path,
        rows=rows,
    )
    try:
        # Keep the top-level CLI lightweight: importing the simulation manager
        # loads PyBullet and controller dependencies, so do it only when the
        # user actually asks to validate or execute a plan.
        from experiments.run_seeded_batch import execute_plan_rows

        execution = execute_plan_rows(
            base_config,
            rows,
            logs_dir=dataset_dir,
            plan_path=plan_path,
            dry_run=dry_run,
            resume=resume,
        )
        verification = None
        status = "validated" if dry_run else "complete"
        if verify_after and not dry_run:
            verification = verify_dataset(dataset_dir, plan_path=plan_path)
            if not verification["passed"]:
                status = "verification_failed"
        payload = write_dataset_manifest(
            dataset_dir,
            status=status,
            config_path=config_path,
            plan_path=plan_path,
            rows=rows,
            execution=execution,
            verification=verification,
        )
    except Exception as exc:
        write_dataset_manifest(
            dataset_dir,
            status="failed",
            config_path=config_path,
            plan_path=plan_path,
            rows=rows,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    return payload


__all__ = [
    "generate_dataset",
    "summarize_plan",
    "verify_dataset",
    "write_dataset_manifest",
]
