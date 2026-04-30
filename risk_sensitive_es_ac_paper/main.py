"""Entry point for paper-aligned experiments."""

from __future__ import annotations

import argparse

from configs import TrainConfig
from objectives import SUPPORTED_OBJECTIVES
from trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objective",
        choices=SUPPORTED_OBJECTIVES,
        default="es",
    )
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--num-episodes", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=10_000)
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-heatmap", action="store_true")
    parser.add_argument("--rollout-mode", choices=["batched", "scalar"], default="batched")
    parser.add_argument("--no-vectorized-eval", dest="eval_vectorized", action="store_false")
    parser.add_argument("--eval-batch-size", type=int, default=65_536)
    parser.add_argument("--best-selection", choices=["rollout", "validation"], default="rollout")
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--validation-episodes", type=int, default=1_000)
    parser.add_argument("--onestep-mc-samples", type=int, default=64)
    parser.add_argument("--conditioned-lambdas", type=str, default="0,0.25,0.5,0.75,1")
    parser.add_argument("--conditioned-alphas", type=str, default="0.8")
    parser.add_argument("--conditioned-eval-lambdas", type=str, default="0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1")
    parser.add_argument("--conditioned-calibration-rounds", type=int, default=2)
    parser.set_defaults(eval_vectorized=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.risk_alpha is None:
        args.risk_alpha = args.alpha
    else:
        args.alpha = args.risk_alpha

    output_dir = args.output_dir
    if args.objective == "mean" and args.output_dir == "outputs":
        output_dir = "outputs_mean"

    config = TrainConfig(
        objective=args.objective,
        iterations=args.iterations,
        num_episodes=args.num_episodes,
        eval_episodes=args.eval_episodes,
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
        seed=args.seed,
        log_interval=args.log_interval,
        output_dir=output_dir,
        device=args.device,
        make_heatmap=not args.no_heatmap,
        rollout_mode=args.rollout_mode,
        eval_vectorized=args.eval_vectorized,
        eval_batch_size=args.eval_batch_size,
        best_selection=args.best_selection,
        validation_interval=args.validation_interval,
        validation_episodes=args.validation_episodes,
        onestep_mc_samples=args.onestep_mc_samples,
        conditioned_lambdas=args.conditioned_lambdas,
        conditioned_alphas=args.conditioned_alphas,
        conditioned_eval_lambdas=args.conditioned_eval_lambdas,
        conditioned_calibration_rounds=args.conditioned_calibration_rounds,
    )
    result = run_training(config)
    metrics = result["eval_metrics"]
    objective_alpha = float(result["objective_alpha"])

    if args.objective == "conditioned_composite_es":
        best_metrics = result["best_eval_metrics"]
        print("\nFinal deterministic evaluation (conditioned_composite_es average over eval grid)")
        print(f"mean cost : {metrics['mean_cost']:.6f}")
        print(f"variance  : {metrics['variance']:.6f}")
        print(f"ES avg    : {metrics['ES']:.6f}")
        print(f"J avg     : {metrics['average_composite_metric']:.6f}")
        print(f"outputs   : {result['output_dir']}")
        print("\nBest checkpoint")
        print(f"selection : {result['best_record']['selection_mode']} / {result['best_record']['selection_metric']}")
        print(f"mean cost : {best_metrics['mean_cost']:.6f}")
        print(f"variance  : {best_metrics['variance']:.6f}")
        print(f"ES avg    : {best_metrics['ES']:.6f}")
        print(f"J avg     : {best_metrics['average_composite_metric']:.6f}")
        print("v_star    : saved in v_star_table.json and best_v_star_table.json")
        return

    if args.objective == "composite_es":
        print(f"\nFinal deterministic evaluation ({args.objective}, lambda={args.risk_lambda:.3g})")
    else:
        print(f"\nFinal deterministic evaluation ({args.objective})")
    print(f"mean cost : {metrics['mean_cost']:.6f}")
    print(f"variance  : {metrics['variance']:.6f}")
    print(f"VaR_{objective_alpha:.1f}   : {metrics['VaR']:.6f}")
    print(f"ES_{objective_alpha:.1f}    : {metrics['ES']:.6f}")
    print(f"v_star    : {result['v_star']:.6f}")
    print(f"outputs   : {result['output_dir']}")

    best_record = result["best_record"]["train_or_validation_record"]
    best_metrics = result["best_eval_metrics"]
    print("\nBest checkpoint")
    print(f"iteration : {int(best_record['iteration'])}")
    print(f"selection : {result['best_record']['selection_mode']} / {result['best_record']['selection_metric']}")
    print(f"record mean: {best_record['mean_cost']:.6f}")
    print(f"record ES  : {best_record['ES']:.6f}")
    print(f"mean cost  : {best_metrics['mean_cost']:.6f}")
    print(f"variance   : {best_metrics['variance']:.6f}")
    print(f"VaR_{objective_alpha:.1f}    : {best_metrics['VaR']:.6f}")
    print(f"ES_{objective_alpha:.1f}     : {best_metrics['ES']:.6f}")
    print(f"v_star     : {result['best_v_star']:.6f}")

    zero = result["zero_baseline_metrics"]
    print("\nAlways-zero baseline")
    print(f"mean cost : {zero['mean_cost']:.6f}")
    print(f"variance  : {zero['variance']:.6f}")
    print(f"VaR_{objective_alpha:.1f}   : {zero['VaR']:.6f}")
    print(f"ES_{objective_alpha:.1f}    : {zero['ES']:.6f}")


if __name__ == "__main__":
    main()
