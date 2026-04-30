"""Run multi-seed paper-aligned experiments and aggregate comparisons."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import torch

from batched_evaluation import evaluate_zero_policy_batched
from compare import evaluate_checkpoint, load_best_v_star, load_last_v_star
from configs import EnvConfig, TrainConfig
from evaluation import evaluate_zero_policy
from objectives import SUPPORTED_OBJECTIVES
from trainer import run_training
from utils import ensure_dir, resolve_device, save_metrics_csv


def parse_seed_list(value: str) -> list[int]:
    seeds: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise argparse.ArgumentTypeError("Provide at least one seed.")
    return seeds


def completed_training_dir(path: Path) -> bool:
    return (path / "training_metrics.csv").exists() and (path / "best_actor.pt").exists()


def format_float_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def objective_run_label(objective: str, risk_lambda: float) -> str:
    if objective == "composite_es":
        return f"{objective}_l{format_float_tag(risk_lambda)}"
    return objective


def comparison_scope_label(args: argparse.Namespace) -> str:
    if "composite_es" not in args.objectives:
        return ""
    if len(args.objectives) == 1:
        return objective_run_label("composite_es", args.risk_lambda)
    return f"risk_l{format_float_tag(args.risk_lambda)}"


def run_dir(output_root: Path, objective: str, seed: int, risk_lambda: float = 0.5) -> Path:
    return output_root / f"{objective_run_label(objective, risk_lambda)}_seed{seed}"


def build_train_config(args: argparse.Namespace, objective: str, seed: int, output_dir: Path) -> TrainConfig:
    return TrainConfig(
        objective=objective,
        iterations=args.iterations,
        num_episodes=args.num_episodes,
        eval_episodes=args.train_eval_episodes,
        critic_updates=args.critic_updates,
        actor_updates=args.actor_updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        alpha=args.alpha,
        risk_alpha=args.risk_alpha,
        risk_lambda=args.risk_lambda,
        initial_v_star=args.initial_v_star,
        sigma_v=args.sigma_v,
        sigma_v_decay=args.sigma_v_decay,
        sigma_v_decay_every=args.sigma_v_decay_every,
        min_sigma_v=args.min_sigma_v,
        seed=seed,
        log_interval=args.log_interval,
        output_dir=str(output_dir),
        device=args.train_device,
        make_heatmap=args.heatmap,
        rollout_mode=args.rollout_mode,
        eval_vectorized=args.train_eval_vectorized,
        eval_batch_size=args.train_eval_batch_size,
        best_selection=args.best_selection,
        validation_interval=args.validation_interval,
        validation_episodes=args.validation_episodes,
        onestep_mc_samples=args.onestep_mc_samples,
        conditioned_lambdas=args.conditioned_lambdas,
        conditioned_alphas=args.conditioned_alphas,
        conditioned_eval_lambdas=args.conditioned_eval_lambdas,
        conditioned_calibration_rounds=args.conditioned_calibration_rounds,
    )


def train_runs(args: argparse.Namespace, output_root: Path) -> None:
    force_train = args.force or args.force_train
    for seed in args.seeds:
        for objective in args.objectives:
            path = run_dir(output_root, objective, seed, args.risk_lambda)
            if completed_training_dir(path) and not force_train:
                print(f"[skip] {path} already has training outputs")
                continue

            label = objective_run_label(objective, args.risk_lambda)
            print(f"\n[train] objective={label} seed={seed} output={path}")
            config = build_train_config(args, objective, seed, path)
            run_training(config)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def comparison_rows_for_seed(args: argparse.Namespace, output_root: Path, seed: int) -> list[dict[str, object]]:
    device = resolve_device(args.compare_device)
    env_config = EnvConfig()
    rows: list[dict[str, object]] = []

    if args.vectorized:
        zero_metrics = evaluate_zero_policy_batched(
            env_config,
            args.compare_eval_episodes,
            alpha=args.alpha,
            device=device,
            seed=seed + args.compare_seed_offset,
            batch_size=args.compare_batch_size,
        )
    else:
        zero_metrics = evaluate_zero_policy(
            env_config,
            args.compare_eval_episodes,
            alpha=args.alpha,
            seed=seed + args.compare_seed_offset,
        )
    rows.append({"seed": seed, "model": "always_zero", **zero_metrics})

    for objective in args.objectives:
        path = run_dir(output_root, objective, seed, args.risk_lambda)
        label = objective_run_label(objective, args.risk_lambda)
        if not completed_training_dir(path):
            print(f"[warn] missing completed training outputs: {path}")
            continue

        last_actor = path / "last_actor.pt"
        if not last_actor.exists():
            last_actor = path / "actor.pt"
        if last_actor.exists():
            metrics = evaluate_checkpoint(
                path,
                last_actor.name,
                objective,
                args.alpha,
                args.risk_lambda,
                load_last_v_star(path),
                args.compare_eval_episodes,
                seed + args.compare_seed_offset,
                device,
                args.vectorized,
                args.compare_batch_size,
            )
            rows.append({"seed": seed, "model": f"{label}_last", **metrics})

        best_actor = path / "best_actor.pt"
        if best_actor.exists():
            metrics = evaluate_checkpoint(
                path,
                best_actor.name,
                objective,
                args.alpha,
                args.risk_lambda,
                load_best_v_star(path),
                args.compare_eval_episodes,
                seed + args.compare_seed_offset,
                device,
                args.vectorized,
                args.compare_batch_size,
            )
            rows.append({"seed": seed, "model": f"{label}_best", **metrics})

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return rows


def aggregate_by_model(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)

    summary: list[dict[str, object]] = []
    metric_names = [
        "mean_cost",
        "variance",
        "std_cost",
        "VaR",
        "ES",
        "ES_0.8",
        "ES_0.6",
        "AVar_0.8",
        "mean_var_utility",
    ]
    for model in sorted(grouped):
        model_rows = grouped[model]
        out: dict[str, object] = {"model": model, "num_seeds": len(model_rows)}
        for metric in metric_names:
            if metric not in model_rows[0]:
                continue
            values = [float(row[metric]) for row in model_rows]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        summary.append(out)
    return summary


def read_metrics_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_comparisons(args: argparse.Namespace, output_root: Path) -> None:
    all_rows: list[dict[str, object]] = []
    scope_label = comparison_scope_label(args)
    comparisons_dir = output_root / "comparisons"
    if scope_label:
        comparisons_dir = comparisons_dir / scope_label
    comparisons_dir = ensure_dir(comparisons_dir)
    force_compare = args.force or args.force_compare

    for seed in args.seeds:
        seed_dir = ensure_dir(comparisons_dir / f"seed{seed}")
        comparison_path = seed_dir / "comparison_table.csv"
        if comparison_path.exists() and not force_compare:
            print(f"[skip] {comparison_path} already exists")
            all_rows.extend(read_metrics_csv(comparison_path))
            continue

        print(f"\n[compare] seed={seed} episodes={args.compare_eval_episodes} device={args.compare_device}")
        rows = comparison_rows_for_seed(args, output_root, seed)
        save_metrics_csv(comparison_path, rows)
        all_rows.extend(rows)
        for row in rows:
            print(
                f"{str(row['model']):>12} seed={seed} "
                f"mean={float(row['mean_cost']):.6f} "
                f"VaR={float(row['VaR']):.6f} ES={float(row['ES']):.6f}"
            )

    if all_rows:
        if scope_label:
            save_metrics_csv(output_root / f"all_comparisons_{scope_label}.csv", all_rows)
            save_metrics_csv(output_root / f"summary_by_model_{scope_label}.csv", aggregate_by_model(all_rows))
        else:
            save_metrics_csv(output_root / "all_comparisons.csv", all_rows)
            save_metrics_csv(output_root / "summary_by_model.csv", aggregate_by_model(all_rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ES/Mean experiments across multiple random seeds.")
    parser.add_argument("--seeds", type=parse_seed_list, default=parse_seed_list("7,17,31"))
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=SUPPORTED_OBJECTIVES,
        default=["es", "mean"],
    )
    parser.add_argument("--output-root", type=str, default="multirun")
    parser.add_argument("--force", action="store_true", help="Rerun training/comparison even when outputs already exist.")
    parser.add_argument("--force-train", action="store_true", help="Only force training reruns.")
    parser.add_argument("--force-compare", action="store_true", help="Only force comparison reruns.")

    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--num-episodes", type=int, default=256)
    parser.add_argument("--train-eval-episodes", type=int, default=10_000)
    parser.add_argument("--critic-updates", type=int, default=5)
    parser.add_argument("--actor-updates", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--risk-alpha", type=float, default=None)
    parser.add_argument("--risk-lambda", type=float, default=0.5)
    parser.add_argument("--initial-v-star", type=float, default=0.0)
    parser.add_argument("--sigma-v", type=float, default=1.0)
    parser.add_argument("--sigma-v-decay", type=float, default=0.98)
    parser.add_argument("--sigma-v-decay-every", type=int, default=10)
    parser.add_argument("--min-sigma-v", type=float, default=0.05)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--train-device", type=str, default="auto")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--rollout-mode", choices=["batched", "scalar"], default="batched")
    parser.add_argument("--train-eval-batch-size", type=int, default=65_536)
    parser.add_argument("--no-vectorized-train-eval", dest="train_eval_vectorized", action="store_false")
    parser.add_argument("--best-selection", choices=["rollout", "validation"], default="validation")
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--validation-episodes", type=int, default=1_000)
    parser.add_argument("--onestep-mc-samples", type=int, default=64)
    parser.add_argument("--conditioned-lambdas", type=str, default="0,0.25,0.5,0.75,1")
    parser.add_argument("--conditioned-alphas", type=str, default="0.8")
    parser.add_argument("--conditioned-eval-lambdas", type=str, default="0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1")
    parser.add_argument("--conditioned-calibration-rounds", type=int, default=2)

    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--compare-device", type=str, default="auto")
    parser.add_argument("--compare-eval-episodes", type=int, default=1_000_000)
    parser.add_argument("--compare-batch-size", type=int, default=262_144)
    parser.add_argument("--compare-seed-offset", type=int, default=100_000)
    parser.add_argument("--no-vectorized", dest="vectorized", action="store_false")
    parser.set_defaults(vectorized=True, train_eval_vectorized=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.risk_alpha is None:
        args.risk_alpha = args.alpha
    else:
        args.alpha = args.risk_alpha

    output_root = ensure_dir(args.output_root)
    print(f"[config] seeds={args.seeds} objectives={args.objectives} output_root={output_root}")
    print(
        f"[config] train_device={args.train_device} rollout_mode={args.rollout_mode} "
        f"compare_device={args.compare_device} risk_alpha={args.risk_alpha} risk_lambda={args.risk_lambda}"
    )

    train_runs(args, output_root)
    if not args.no_compare:
        run_comparisons(args, output_root)


if __name__ == "__main__":
    main()
