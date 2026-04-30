"""Unified checkpoint evaluation for paper-aligned experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from batched_evaluation import evaluate_actor_batched, evaluate_conditioned_actor_batched, evaluate_zero_policy_batched
from configs import EnvConfig
from evaluation import evaluate_actor, evaluate_zero_policy
from networks import ActorNetwork
from objectives import SUPPORTED_OBJECTIVES, make_objective
from utils import ensure_dir, resolve_device, save_metrics_csv


def load_best_v_star(run_dir: Path) -> float:
    record_path = run_dir / "best_record.json"
    if not record_path.exists():
        return 0.0
    with record_path.open("r", encoding="utf-8") as f:
        record = json.load(f)
    return float(record["train_or_validation_record"].get("v_star", 0.0))


def load_last_v_star(run_dir: Path) -> float:
    metrics_path = run_dir / "training_metrics.csv"
    if not metrics_path.exists():
        return 0.0
    with metrics_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0.0
    return float(rows[-1].get("v_star", 0.0))


def load_conditioned_v_star(run_dir: Path, checkpoint_name: str, risk_lambda: float, alpha: float) -> float:
    table_name = "best_v_star_table.json" if checkpoint_name.startswith("best") else "v_star_table.json"
    table_path = run_dir / table_name
    if not table_path.exists():
        table_path = run_dir / "v_star_table.json"
    if not table_path.exists():
        return 0.0
    with table_path.open("r", encoding="utf-8") as f:
        table = json.load(f)
    return float(table.get(f"{risk_lambda:g},{alpha:g}", 0.0))


def parse_objective_dir(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use OBJECTIVE=DIR, for example onestep_es08=outputs_onestep.")
    objective_name, run_dir = value.split("=", 1)
    objective_name = objective_name.strip()
    run_dir = run_dir.strip()
    if objective_name not in SUPPORTED_OBJECTIVES:
        choices = ", ".join(SUPPORTED_OBJECTIVES)
        raise argparse.ArgumentTypeError(f"Unsupported objective '{objective_name}'. Choices: {choices}.")
    if not run_dir:
        raise argparse.ArgumentTypeError("Run directory cannot be empty.")
    return objective_name, run_dir


def evaluate_checkpoint(
    run_dir: Path,
    checkpoint_name: str,
    objective_name: str,
    alpha: float,
    risk_lambda: float,
    v_star: float,
    episodes: int,
    seed: int,
    device: torch.device,
    vectorized: bool,
    eval_batch_size: int,
) -> dict[str, float]:
    env_config = EnvConfig()
    if objective_name == "conditioned_composite_es":
        if not vectorized:
            raise ValueError("conditioned_composite_es checkpoint evaluation currently requires --vectorized.")
        actor = ActorNetwork(input_dim=7, action_scale=env_config.a_max).to(device)
        actor.load_state_dict(torch.load(run_dir / checkpoint_name, map_location=device))
        theta_v_star = load_conditioned_v_star(run_dir, checkpoint_name, risk_lambda, alpha)
        return evaluate_conditioned_actor_batched(
            actor,
            env_config,
            risk_lambda,
            alpha,
            theta_v_star,
            episodes,
            device,
            seed,
            batch_size=eval_batch_size,
        )

    objective = make_objective(objective_name, alpha=alpha, risk_lambda=risk_lambda)
    actor = ActorNetwork(action_scale=env_config.a_max).to(device)
    actor.load_state_dict(torch.load(run_dir / checkpoint_name, map_location=device))
    if vectorized:
        return evaluate_actor_batched(
            actor,
            objective,
            env_config,
            v_star,
            episodes,
            device,
            seed,
            batch_size=eval_batch_size,
        )
    return evaluate_actor(actor, objective, env_config, v_star, episodes, device, seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--es-dir", type=str)
    parser.add_argument("--es06-dir", type=str)
    parser.add_argument("--mean-dir", type=str)
    parser.add_argument("--var-dir", type=str)
    parser.add_argument("--avar08-dir", type=str)
    parser.add_argument("--mean-var-dir", type=str)
    parser.add_argument("--composite-es-dir", type=str)
    parser.add_argument("--onestep-es08-dir", type=str)
    parser.add_argument("--onestep-es06-dir", type=str)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        type=parse_objective_dir,
        metavar="OBJECTIVE=DIR",
        help="Add any supported objective run directory, e.g. --run onestep_es08=outputs_onestep.",
    )
    parser.add_argument("--eval-episodes", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--risk-alpha", type=float, default=None)
    parser.add_argument("--risk-lambda", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="comparison_outputs")
    parser.add_argument("--vectorized", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=65_536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.risk_alpha is None:
        args.risk_alpha = args.alpha
    else:
        args.alpha = args.risk_alpha

    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    env_config = EnvConfig()
    rows: list[dict[str, object]] = []

    if args.vectorized:
        zero = evaluate_zero_policy_batched(
            env_config,
            args.eval_episodes,
            alpha=args.alpha,
            device=device,
            seed=args.seed,
            batch_size=args.eval_batch_size,
        )
    else:
        zero = evaluate_zero_policy(env_config, args.eval_episodes, alpha=args.alpha, seed=args.seed)
    rows.append({"model": "always_zero", **zero})

    objective_dirs = [
        ("es", args.es_dir),
        ("es06", args.es06_dir),
        ("mean", args.mean_dir),
        ("var", args.var_dir),
        ("avar08", args.avar08_dir),
        ("mean_var", args.mean_var_dir),
        ("composite_es", args.composite_es_dir),
        ("onestep_es08", args.onestep_es08_dir),
        ("onestep_es06", args.onestep_es06_dir),
        *args.run,
    ]
    for objective_name, dir_arg in objective_dirs:
        if not dir_arg:
            continue
        run_dir = Path(dir_arg)
        last_actor = run_dir / "last_actor.pt"
        if not last_actor.exists():
            last_actor = run_dir / "actor.pt"
        if last_actor.exists():
            metrics = evaluate_checkpoint(
                run_dir,
                last_actor.name,
                objective_name,
                args.alpha,
                args.risk_lambda,
                load_last_v_star(run_dir),
                args.eval_episodes,
                args.seed,
                device,
                args.vectorized,
                args.eval_batch_size,
            )
            rows.append({"model": f"{objective_name}_last", **metrics})
        best_actor = run_dir / "best_actor.pt"
        if best_actor.exists():
            metrics = evaluate_checkpoint(
                run_dir,
                best_actor.name,
                objective_name,
                args.alpha,
                args.risk_lambda,
                load_best_v_star(run_dir),
                args.eval_episodes,
                args.seed,
                device,
                args.vectorized,
                args.eval_batch_size,
            )
            rows.append({"model": f"{objective_name}_best", **metrics})

    save_metrics_csv(output_dir / "comparison_table.csv", rows)
    for row in rows:
        print(
            f"{row['model']:>12} mean={float(row['mean_cost']):.6f} "
            f"var={float(row['variance']):.6f} VaR={float(row['VaR']):.6f} ES={float(row['ES']):.6f}"
        )


if __name__ == "__main__":
    main()
