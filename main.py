"""Unified command-line entry point for simulation and dataset generation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from experiments.dataset_pipeline import (
    generate_dataset,
    verify_dataset,
    write_dataset_manifest,
)
from experiments.make_paired_perturbation_plan import (
    add_plan_arguments,
    build_plan,
    next_available_run_id,
    write_plan,
)
from utilities.config import load_config


def _command_simulate(args: argparse.Namespace) -> int:
    """Run one or more ordinary simulations without an experimental plan."""

    from simulator.simulator_manager import SimulationManager

    base_config = load_config(args.config)
    if args.runs < 1:
        raise ValueError("--runs must be at least one.")
    mode = args.mode or ("gui" if args.runs == 1 else "direct")
    if args.runs > 1 and mode == "gui":
        raise ValueError("Batch simulation requires --mode direct.")

    completed = []
    for index in range(args.runs):
        config = copy.deepcopy(base_config)
        simulation = config.setdefault("simulation", {})
        simulation["connect_mode"] = mode
        if args.logs_dir is not None:
            simulation["log_dir"] = str(args.logs_dir)
        if args.max_sim_time is not None:
            simulation["max_sim_time"] = float(args.max_sim_time)
        if args.seed is not None:
            simulation["seed"] = int(args.seed)
            simulation["seed_add_run_id"] = args.runs > 1
        if args.start_run_id is not None:
            simulation["run_config"] = int(args.start_run_id) + index
            simulation["fixed_run_config"] = True

        manager = SimulationManager(config)
        try:
            manager.run()
            completed.append(int(manager.config["simulation"]["run_config"]))
        finally:
            manager.stop()

    print(json.dumps({"completed_run_ids": completed, "mode": mode}, indent=2))
    return 0


def _command_dataset_plan(args: argparse.Namespace) -> int:
    """Create a reproducible paired counterfactual plan."""

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if args.out_plan is None:
        args.out_plan = args.logs_dir / "plan.csv"
    if args.out_plan.exists() and not args.force:
        raise FileExistsError(
            f"Plan already exists: {args.out_plan}. Use --force to replace it."
        )
    if args.start_run_id is None:
        args.start_run_id = next_available_run_id(args.logs_dir)
    rows = build_plan(args)
    write_plan(args.out_plan, rows)
    manifest = write_dataset_manifest(
        args.logs_dir,
        status="planned",
        config_path=config_path,
        plan_path=args.out_plan,
        rows=rows,
    )
    print(
        json.dumps(
            {
                "plan": str(args.out_plan),
                "manifest": str(args.logs_dir / "dataset_manifest.json"),
                **manifest["plan"],
            },
            indent=2,
        )
    )
    return 0


def _command_dataset_generate(args: argparse.Namespace) -> int:
    """Validate or execute an existing dataset plan."""

    payload = generate_dataset(
        config_path=Path(args.config),
        dataset_dir=args.dataset,
        plan_path=args.plan,
        dry_run=args.dry_run,
        resume=args.resume,
        verify_after=not args.no_verify,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "dataset": payload["dataset_dir"],
                "manifest": str(args.dataset / "dataset_manifest.json"),
            },
            indent=2,
        )
    )
    if payload["status"] == "verification_failed":
        return 1
    return 0


def _command_dataset_verify(args: argparse.Namespace) -> int:
    """Check that all plan, trace, fork and pairing artifacts are present."""

    report = verify_dataset(args.dataset, plan_path=args.plan)
    manifest_path = args.dataset / "dataset_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["status"] = "complete" if report["passed"] else "verification_failed"
        payload["verification"] = report
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UAV simulator and counterfactual dataset pipeline."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    simulate = commands.add_parser(
        "simulate", help="Run one or more ordinary simulations."
    )
    simulate.add_argument("--config", default="config.yaml")
    simulate.add_argument("--runs", type=int, default=1)
    simulate.add_argument("--mode", choices=["gui", "direct"], default=None)
    simulate.add_argument("--logs-dir", type=Path, default=None)
    simulate.add_argument("--start-run-id", type=int, default=None)
    simulate.add_argument("--max-sim-time", type=float, default=None)
    simulate.add_argument("--seed", type=int, default=None)
    simulate.set_defaults(handler=_command_simulate)

    dataset = commands.add_parser(
        "dataset", help="Plan, generate and verify counterfactual datasets."
    )
    dataset_commands = dataset.add_subparsers(
        dest="dataset_command", required=True
    )

    plan = dataset_commands.add_parser(
        "plan", help="Create a paired plan.csv without simulating."
    )
    add_plan_arguments(plan)
    plan.set_defaults(out_plan=None)
    plan.add_argument(
        "--dataset",
        dest="logs_dir",
        type=Path,
        default=argparse.SUPPRESS,
        help="Alias for --logs-dir.",
    )
    plan.add_argument(
        "--config",
        default="config.yaml",
        help="Base simulator config recorded in the dataset manifest.",
    )
    plan.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing plan.csv.",
    )
    plan.set_defaults(handler=_command_dataset_plan)

    generate = dataset_commands.add_parser(
        "generate", help="Execute or resume an existing plan.csv."
    )
    generate.add_argument("--config", default="config.yaml")
    generate.add_argument("--dataset", type=Path, required=True)
    generate.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="Plan path; defaults to <dataset>/plan.csv.",
    )
    generate.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate every effective run config without starting PyBullet.",
    )
    generate.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already have a summary JSON.",
    )
    generate.add_argument(
        "--no-verify",
        action="store_true",
        help="Do not run the final artifact verification.",
    )
    generate.set_defaults(handler=_command_dataset_generate)

    verify = dataset_commands.add_parser(
        "verify", help="Audit all artifacts declared by a dataset plan."
    )
    verify.add_argument("--dataset", type=Path, required=True)
    verify.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="Plan path; defaults to <dataset>/plan.csv.",
    )
    verify.set_defaults(handler=_command_dataset_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
