"""Paper-aligned training orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from batched_evaluation import (
    estimate_conditioned_v_star_batched,
    evaluate_actor_batched,
    evaluate_conditioned_actor_batched,
    evaluate_zero_policy_batched,
)
from batched_rollout import collect_conditioned_rollouts_batched, collect_rollouts_batched
from configs import EnvConfig, TrainConfig
from evaluation import evaluate_actor, evaluate_zero_policy, save_line_plot, save_policy_heatmap
from networks import ActorNetwork, CriticNetwork
from objectives import Objective, make_objective
from rollout import collect_rollouts
from utils import ensure_dir, metric_summary, resolve_device, sample_var, save_metrics_csv, set_global_seed


def comparison_row(model: str, metrics: dict[str, float]) -> dict[str, object]:
    row: dict[str, object] = {"model": model}
    for key, value in metrics.items():
        row[key] = value
    return row


def update_critic(
    critic: CriticNetwork,
    objective: Objective,
    optimizer: optim.Optimizer,
    tensors: dict[str, torch.Tensor],
    env_config: EnvConfig,
    updates: int,
    batch_size: int,
    onestep_mc_samples: int,
) -> float:
    n = tensors["inputs"].shape[0]
    last_loss = 0.0
    critic.train()
    for _ in range(updates):
        idx = torch.randint(0, n, (min(batch_size, n),), device=tensors["inputs"].device)
        batch = {k: v[idx] for k, v in tensors.items()}
        pred = critic(batch["inputs"])
        target = objective.critic_target(
            critic,
            batch,
            env_config=env_config,
            mc_samples=onestep_mc_samples,
        )
        loss = F.mse_loss(pred, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    return last_loss


def update_actor(
    actor: ActorNetwork,
    critic: CriticNetwork,
    objective: Objective,
    optimizer: optim.Optimizer,
    tensors: dict[str, torch.Tensor],
    env_config: EnvConfig,
    updates: int,
    batch_size: int,
    onestep_mc_samples: int,
) -> float:
    n = tensors["inputs"].shape[0]
    last_loss = 0.0
    actor.train()
    critic.eval()
    for _ in range(updates):
        idx = torch.randint(0, n, (min(batch_size, n),), device=tensors["inputs"].device)
        batch = {k: v[idx] for k, v in tensors.items()}

        with torch.no_grad():
            value_signal = objective.critic_target(
                critic,
                batch,
                env_config=env_config,
                mc_samples=onestep_mc_samples,
            )
            advantage = value_signal - value_signal.mean()
            advantage = advantage / (advantage.std(unbiased=False) + 1e-6)

        log_prob = actor.log_prob_from_action(batch["inputs"], batch["actions"])
        loss = (log_prob * advantage).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    return last_loss


def save_checkpoint(
    actor: ActorNetwork,
    critic: CriticNetwork,
    output_dir: Path,
    prefix: str,
) -> tuple[Path, Path]:
    actor_path = output_dir / f"{prefix}_actor.pt"
    critic_path = output_dir / f"{prefix}_critic.pt"
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    return actor_path, critic_path


def evaluate_actor_for_training(
    actor: ActorNetwork,
    objective: Objective,
    env_config: EnvConfig,
    v_star: float,
    episodes: int,
    device: torch.device,
    seed: int,
    vectorized: bool,
    batch_size: int,
) -> dict[str, float]:
    if vectorized:
        return evaluate_actor_batched(
            actor,
            objective,
            env_config,
            v_star,
            episodes,
            device,
            seed,
            batch_size=batch_size,
        )
    return evaluate_actor(actor, objective, env_config, v_star, episodes, device, seed)


def evaluate_zero_for_training(
    env_config: EnvConfig,
    episodes: int,
    alpha: float,
    device: torch.device,
    seed: int,
    vectorized: bool,
    batch_size: int,
) -> dict[str, float]:
    if vectorized:
        return evaluate_zero_policy_batched(
            env_config,
            episodes,
            alpha,
            device,
            seed,
            batch_size=batch_size,
    )
    return evaluate_zero_policy(env_config, episodes, alpha, seed)


ConditionedTheta = tuple[float, float]


def parse_float_grid(value: str, name: str) -> list[float]:
    values: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    return values


def theta_key(theta: ConditionedTheta) -> ConditionedTheta:
    return (round(float(theta[0]), 10), round(float(theta[1]), 10))


def theta_json_key(theta: ConditionedTheta) -> str:
    risk_lambda, alpha = theta_key(theta)
    return f"{risk_lambda:g},{alpha:g}"


def validate_conditioned_grid(theta_grid: list[ConditionedTheta]) -> None:
    for risk_lambda, alpha in theta_grid:
        if not 0.0 <= risk_lambda <= 1.0:
            raise ValueError(f"conditioned lambda must be in [0, 1], got {risk_lambda}.")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"conditioned alpha must be in (0, 1), got {alpha}.")


def make_conditioned_grid(lambdas: list[float], alphas: list[float]) -> list[ConditionedTheta]:
    theta_grid = [theta_key((risk_lambda, alpha)) for alpha in alphas for risk_lambda in lambdas]
    validate_conditioned_grid(theta_grid)
    return theta_grid


def conditioned_composite_metric(metrics: dict[str, float], risk_lambda: float) -> float:
    return (1.0 - float(risk_lambda)) * metrics["mean_cost"] + float(risk_lambda) * metrics["ES"]


def v_star_table_to_json(v_star_table: dict[ConditionedTheta, float]) -> dict[str, float]:
    return {theta_json_key(theta): float(value) for theta, value in sorted(v_star_table.items())}


def lookup_conditioned_v_star(
    v_star_table: dict[ConditionedTheta, float],
    train_theta_grid: list[ConditionedTheta],
    theta: ConditionedTheta,
) -> tuple[float, bool]:
    key = theta_key(theta)
    if key in v_star_table:
        return float(v_star_table[key]), True

    risk_lambda, alpha = key
    if risk_lambda <= 0.0:
        return 0.0, True

    same_alpha = sorted(
        (candidate for candidate in train_theta_grid if candidate[1] == alpha and candidate[0] > 0.0),
        key=lambda item: item[0],
    )
    if not same_alpha:
        return 0.0, False
    if risk_lambda <= same_alpha[0][0]:
        return float(v_star_table.get(same_alpha[0], 0.0)), False
    if risk_lambda >= same_alpha[-1][0]:
        return float(v_star_table.get(same_alpha[-1], 0.0)), False

    for left, right in zip(same_alpha[:-1], same_alpha[1:]):
        left_lambda, _ = left
        right_lambda, _ = right
        if left_lambda <= risk_lambda <= right_lambda:
            weight = (risk_lambda - left_lambda) / (right_lambda - left_lambda)
            left_v = float(v_star_table.get(left, 0.0))
            right_v = float(v_star_table.get(right, 0.0))
            return (1.0 - weight) * left_v + weight * right_v, False
    return float(v_star_table.get(same_alpha[-1], 0.0)), False


def summarize_conditioned_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {}
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
        "composite_metric",
    ]
    summary: dict[str, float] = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in rows if metric in row]
        if values:
            summary[metric] = float(sum(values) / len(values))
    summary["average_composite_metric"] = summary.get("composite_metric", float("inf"))
    return summary


def evaluate_conditioned_grid_for_training(
    actor: ActorNetwork,
    env_config: EnvConfig,
    theta_grid: list[ConditionedTheta],
    train_theta_grid: list[ConditionedTheta],
    v_star_table: dict[ConditionedTheta, float],
    episodes: int,
    device: torch.device,
    seed: int,
    batch_size: int,
    calibration_rounds: int,
    calibrate_unseen: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, theta in enumerate(theta_grid):
        risk_lambda, alpha = theta
        eval_v_star, is_trained_theta = lookup_conditioned_v_star(v_star_table, train_theta_grid, theta)
        used_calibration_rounds = 0
        if calibrate_unseen and not is_trained_theta and risk_lambda > 0.0:
            for round_index in range(max(0, calibration_rounds)):
                eval_v_star = estimate_conditioned_v_star_batched(
                    actor,
                    env_config,
                    risk_lambda,
                    alpha,
                    eval_v_star,
                    episodes,
                    device,
                    seed + 10_000 + 97 * index + round_index,
                    batch_size=batch_size,
                )
                used_calibration_rounds += 1

        metrics = evaluate_conditioned_actor_batched(
            actor,
            env_config,
            risk_lambda,
            alpha,
            eval_v_star,
            episodes,
            device,
            seed + 20_000 + 97 * index,
            batch_size=batch_size,
        )
        row: dict[str, object] = {
            "lambda": risk_lambda,
            "alpha": alpha,
            "v_star": float(eval_v_star),
            "trained_theta": float(is_trained_theta),
            "calibration_rounds": float(used_calibration_rounds),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def run_conditioned_training(config: TrainConfig) -> dict[str, object]:
    if config.best_selection not in {"rollout", "validation"}:
        raise ValueError(f"Unsupported best_selection: {config.best_selection}")
    if config.rollout_mode != "batched":
        raise ValueError("conditioned_composite_es currently supports only batched rollout.")

    set_global_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = ensure_dir(config.output_dir)

    env_config = EnvConfig()
    objective_alpha = config.risk_alpha if config.risk_alpha is not None else config.alpha
    train_lambdas = parse_float_grid(config.conditioned_lambdas, "conditioned_lambdas")
    train_alphas = parse_float_grid(config.conditioned_alphas, "conditioned_alphas")
    eval_lambdas = parse_float_grid(config.conditioned_eval_lambdas, "conditioned_eval_lambdas")
    train_theta_grid = make_conditioned_grid(train_lambdas, train_alphas)
    eval_theta_grid = make_conditioned_grid(eval_lambdas, train_alphas)

    objective = make_objective("conditioned_composite_es", objective_alpha, risk_lambda=config.risk_lambda)
    actor = ActorNetwork(input_dim=7, action_scale=env_config.a_max).to(device)
    critic = CriticNetwork(input_dim=7).to(device)
    actor_lr = config.learning_rate if config.learning_rate is not None else config.actor_learning_rate
    critic_lr = config.learning_rate if config.learning_rate is not None else config.critic_learning_rate
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    v_star_table = {theta: (0.0 if theta[0] <= 0.0 else float(config.initial_v_star)) for theta in train_theta_grid}
    best_v_star_table = dict(v_star_table)
    sigma_v = float(config.sigma_v)
    rows: list[dict[str, object]] = []

    theta_rng = np.random.default_rng(config.seed + 1234)
    theta_order: list[int] = []

    def next_theta() -> ConditionedTheta:
        nonlocal theta_order
        if not theta_order:
            theta_order = [int(item) for item in theta_rng.permutation(len(train_theta_grid))]
        return train_theta_grid[theta_order.pop(0)]

    best_metric = float("inf")
    best_train_record: dict[str, object] = {}
    best_actor_path = output_dir / "best_actor.pt"
    best_critic_path = output_dir / "best_critic.pt"

    for iteration in range(1, config.iterations + 1):
        active_theta = next_theta()
        risk_lambda, alpha = active_theta
        active_v_star = float(v_star_table[active_theta])
        tensors, total_costs = collect_conditioned_rollouts_batched(
            actor=actor,
            objective=objective,
            env_config=env_config,
            num_episodes=config.num_episodes,
            risk_lambda=risk_lambda,
            alpha=alpha,
            v_star=active_v_star,
            sigma_v=sigma_v,
            device=device,
            seed=config.seed + iteration,
        )

        metrics = metric_summary(total_costs, alpha)
        metrics["composite_metric"] = conditioned_composite_metric(metrics, risk_lambda)
        if risk_lambda <= 0.0:
            v_star_table[active_theta] = 0.0
        else:
            v_star_table[active_theta] = sample_var(total_costs, alpha)

        rollout_metric = metrics["composite_metric"]
        is_best = False
        validation_metrics: dict[str, float] | None = None

        if config.best_selection == "rollout" and rollout_metric < best_metric:
            is_best = True
            best_metric = rollout_metric
            best_v_star_table = dict(v_star_table)
            best_train_record = {
                "iteration": float(iteration),
                "selection_source": "rollout",
                "selection_metric": "active_theta_composite_metric",
                "lambda": risk_lambda,
                "alpha": alpha,
                **metrics,
                "v_star": float(v_star_table[active_theta]),
            }
            best_actor_path, best_critic_path = save_checkpoint(actor, critic, output_dir, "best")

        critic_loss = update_critic(
            critic=critic,
            objective=objective,
            optimizer=critic_optimizer,
            tensors=tensors,
            env_config=env_config,
            updates=config.critic_updates,
            batch_size=config.batch_size,
            onestep_mc_samples=config.onestep_mc_samples,
        )
        actor_loss = update_actor(
            actor=actor,
            critic=critic,
            objective=objective,
            optimizer=actor_optimizer,
            tensors=tensors,
            env_config=env_config,
            updates=config.actor_updates,
            batch_size=config.batch_size,
            onestep_mc_samples=config.onestep_mc_samples,
        )

        if (
            config.best_selection == "validation"
            and config.validation_interval > 0
            and (iteration == 1 or iteration % config.validation_interval == 0 or iteration == config.iterations)
        ):
            validation_rows = evaluate_conditioned_grid_for_training(
                actor,
                env_config,
                train_theta_grid,
                train_theta_grid,
                v_star_table,
                config.validation_episodes,
                device,
                config.seed + 50_000 + iteration,
                config.eval_batch_size,
                calibration_rounds=0,
                calibrate_unseen=False,
            )
            validation_metrics = summarize_conditioned_rows(validation_rows)
            validation_metric = validation_metrics["average_composite_metric"]
            if validation_metric < best_metric:
                is_best = True
                best_metric = validation_metric
                best_v_star_table = dict(v_star_table)
                best_train_record = {
                    "iteration": float(iteration),
                    "selection_source": "validation",
                    "selection_metric": "average_composite_metric",
                    **validation_metrics,
                    "v_star_table": v_star_table_to_json(v_star_table),
                }
                best_actor_path, best_critic_path = save_checkpoint(actor, critic, output_dir, "best")

        if any(theta[0] > 0.0 for theta in train_theta_grid) and iteration % config.sigma_v_decay_every == 0:
            sigma_v = max(config.min_sigma_v, sigma_v * config.sigma_v_decay)

        row = {
            "iteration": float(iteration),
            "lambda": risk_lambda,
            "alpha": alpha,
            "mean_cost": metrics["mean_cost"],
            "variance": metrics["variance"],
            "std_cost": metrics["std_cost"],
            "VaR": metrics["VaR"],
            "ES": metrics["ES"],
            "ES_0.8": metrics["ES_0.8"],
            "ES_0.6": metrics["ES_0.6"],
            "composite_metric": metrics["composite_metric"],
            "v_star": float(v_star_table[active_theta]),
            "sigma_v": float(sigma_v),
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "best_metric": best_metric,
            "is_best": float(is_best),
        }
        if validation_metrics is not None:
            row.update(
                {
                    "validation_mean_cost": validation_metrics["mean_cost"],
                    "validation_ES": validation_metrics["ES"],
                    "validation_ES_0.8": validation_metrics["ES_0.8"],
                    "validation_ES_0.6": validation_metrics["ES_0.6"],
                    "validation_composite_metric": validation_metrics["composite_metric"],
                    "validation_average_composite_metric": validation_metrics["average_composite_metric"],
                }
            )
        rows.append(row)

        if iteration == 1 or iteration % config.log_interval == 0:
            print(
                f"iter={iteration:04d} lambda={risk_lambda:.3g} alpha={alpha:.3g} "
                f"mean={metrics['mean_cost']:.4f} ES={metrics['ES']:.4f} "
                f"J={metrics['composite_metric']:.4f} v_theta={v_star_table[active_theta]:.4f} "
                f"best={best_metric:.4f} critic_loss={critic_loss:.4f} actor_loss={actor_loss:.4f}"
            )

    if not best_train_record:
        best_v_star_table = dict(v_star_table)
        final_summary = summarize_conditioned_rows(
            evaluate_conditioned_grid_for_training(
                actor,
                env_config,
                train_theta_grid,
                train_theta_grid,
                v_star_table,
                config.validation_episodes,
                device,
                config.seed + 60_000,
                config.eval_batch_size,
                calibration_rounds=0,
                calibrate_unseen=False,
            )
        )
        best_train_record = {
            "iteration": float(config.iterations),
            "selection_source": "fallback_final",
            "selection_metric": "average_composite_metric",
            **final_summary,
            "v_star_table": v_star_table_to_json(v_star_table),
        }
        best_actor_path, best_critic_path = save_checkpoint(actor, critic, output_dir, "best")

    save_metrics_csv(output_dir / "training_metrics.csv", rows)
    save_line_plot([float(r["mean_cost"]) for r in rows], "Mean total cost", "Training mean cost", output_dir / "training_mean_cost.png")
    save_line_plot([float(r["ES"]) for r in rows], f"ES_{objective_alpha}", f"Training ES_{objective_alpha}", output_dir / "training_es.png")
    save_line_plot(
        [float(r["composite_metric"]) for r in rows],
        "Composite metric",
        "Training conditioned composite metric",
        output_dir / "training_composite_metric.png",
    )
    save_line_plot([float(r["v_star"]) for r in rows], "active v_star", "Active auxiliary variable estimate", output_dir / "training_v_star.png")

    save_checkpoint(actor, critic, output_dir, "last")
    torch.save(actor.state_dict(), output_dir / "actor.pt")
    torch.save(critic.state_dict(), output_dir / "critic.pt")
    with (output_dir / "v_star_table.json").open("w", encoding="utf-8") as f:
        json.dump(v_star_table_to_json(v_star_table), f, indent=2)
    with (output_dir / "best_v_star_table.json").open("w", encoding="utf-8") as f:
        json.dump(v_star_table_to_json(best_v_star_table), f, indent=2)

    eval_seed = config.seed + 10_000
    eval_rows = evaluate_conditioned_grid_for_training(
        actor,
        env_config,
        eval_theta_grid,
        train_theta_grid,
        v_star_table,
        config.eval_episodes,
        device,
        eval_seed,
        config.eval_batch_size,
        calibration_rounds=config.conditioned_calibration_rounds,
        calibrate_unseen=True,
    )
    for row in eval_rows:
        row["model"] = "conditioned_composite_es_last"
    eval_metrics = summarize_conditioned_rows(eval_rows)

    best_actor = ActorNetwork(input_dim=7, action_scale=env_config.a_max).to(device)
    best_actor.load_state_dict(torch.load(best_actor_path, map_location=device))
    best_eval_rows = evaluate_conditioned_grid_for_training(
        best_actor,
        env_config,
        eval_theta_grid,
        train_theta_grid,
        best_v_star_table,
        config.eval_episodes,
        device,
        eval_seed,
        config.eval_batch_size,
        calibration_rounds=config.conditioned_calibration_rounds,
        calibrate_unseen=True,
    )
    for row in best_eval_rows:
        row["model"] = "conditioned_composite_es_best"
    best_eval_metrics = summarize_conditioned_rows(best_eval_rows)

    zero_baseline_metrics = evaluate_zero_for_training(
        env_config,
        config.eval_episodes,
        objective_alpha,
        device,
        eval_seed,
        config.eval_vectorized,
        config.eval_batch_size,
    )

    save_metrics_csv(output_dir / "conditioned_eval_by_theta.csv", eval_rows + best_eval_rows)
    save_metrics_csv(
        output_dir / "comparison_summary.csv",
        [
            comparison_row("always_zero", zero_baseline_metrics),
            comparison_row("conditioned_composite_es_last_average", eval_metrics),
            comparison_row("conditioned_composite_es_best_average", best_eval_metrics),
        ],
    )

    best_record = {
        "objective": objective.name,
        "risk_alpha": objective_alpha,
        "preference_grid": [{"lambda": theta[0], "alpha": theta[1]} for theta in train_theta_grid],
        "eval_grid": [{"lambda": theta[0], "alpha": theta[1]} for theta in eval_theta_grid],
        "selection_metric": "average_composite_metric",
        "selection_mode": config.best_selection,
        "train_or_validation_record": best_train_record,
        "deterministic_eval": best_eval_metrics,
        "v_star_table": v_star_table_to_json(best_v_star_table),
        "checkpoint_files": {
            "actor": best_actor_path.name,
            "critic": best_critic_path.name,
        },
    }
    with (output_dir / "best_record.json").open("w", encoding="utf-8") as f:
        json.dump(best_record, f, indent=2)
    with (output_dir / "zero_baseline.json").open("w", encoding="utf-8") as f:
        json.dump(zero_baseline_metrics, f, indent=2)

    return {
        "objective": objective.name,
        "objective_alpha": objective_alpha,
        "risk_lambda": config.risk_lambda,
        "actor": actor,
        "critic": critic,
        "v_star": 0.0,
        "v_star_table": v_star_table,
        "best_v_star": 0.0,
        "best_v_star_table": best_v_star_table,
        "best_record": best_record,
        "rows": rows,
        "eval_metrics": eval_metrics,
        "best_eval_metrics": best_eval_metrics,
        "zero_baseline_metrics": zero_baseline_metrics,
        "output_dir": str(output_dir),
    }


def run_training(config: TrainConfig) -> dict[str, object]:
    if config.objective == "conditioned_composite_es":
        return run_conditioned_training(config)

    if config.best_selection not in {"rollout", "validation"}:
        raise ValueError(f"Unsupported best_selection: {config.best_selection}")
    if config.rollout_mode not in {"batched", "scalar"}:
        raise ValueError(f"Unsupported rollout_mode: {config.rollout_mode}")

    set_global_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = ensure_dir(config.output_dir)

    env_config = EnvConfig()
    objective_alpha = config.risk_alpha if config.risk_alpha is not None else config.alpha
    objective = make_objective(config.objective, objective_alpha, risk_lambda=config.risk_lambda)
    actor = ActorNetwork(action_scale=env_config.a_max).to(device)
    critic = CriticNetwork().to(device)
    actor_lr = config.learning_rate if config.learning_rate is not None else config.actor_learning_rate
    critic_lr = config.learning_rate if config.learning_rate is not None else config.critic_learning_rate
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    v_star = float(config.initial_v_star)
    sigma_v = float(config.sigma_v)
    rows: list[dict[str, object]] = []

    best_metric = float("inf")
    best_train_record: dict[str, object] = {}
    best_v_star = v_star
    best_actor_path = output_dir / "best_actor.pt"
    best_critic_path = output_dir / "best_critic.pt"

    for iteration in range(1, config.iterations + 1):
        if config.rollout_mode == "batched":
            tensors, total_costs = collect_rollouts_batched(
                actor=actor,
                objective=objective,
                env_config=env_config,
                num_episodes=config.num_episodes,
                v_star=v_star,
                sigma_v=sigma_v,
                device=device,
                seed=config.seed + iteration,
            )
        else:
            buffer, total_costs = collect_rollouts(
                actor=actor,
                objective=objective,
                env_config=env_config,
                num_episodes=config.num_episodes,
                v_star=v_star,
                sigma_v=sigma_v,
                device=device,
                seed=config.seed + iteration,
            )
            tensors = buffer.as_tensors(env_config.T, device, objective)

        metrics = metric_summary(total_costs, objective.alpha)
        v_star = objective.update_v_star(total_costs, v_star)

        rollout_metric = objective.select_metric(metrics)
        is_best = False
        validation_metrics: dict[str, float] | None = None

        if config.best_selection == "rollout" and rollout_metric < best_metric:
            is_best = True
            best_metric = rollout_metric
            best_v_star = float(v_star)
            best_train_record = {
                "iteration": float(iteration),
                "selection_source": "rollout",
                "selection_metric": objective.selection_metric,
                **metrics,
                "v_star": float(v_star),
            }
            best_actor_path, best_critic_path = save_checkpoint(actor, critic, output_dir, "best")

        critic_loss = update_critic(
            critic=critic,
            objective=objective,
            optimizer=critic_optimizer,
            tensors=tensors,
            env_config=env_config,
            updates=config.critic_updates,
            batch_size=config.batch_size,
            onestep_mc_samples=config.onestep_mc_samples,
        )
        actor_loss = update_actor(
            actor=actor,
            critic=critic,
            objective=objective,
            optimizer=actor_optimizer,
            tensors=tensors,
            env_config=env_config,
            updates=config.actor_updates,
            batch_size=config.batch_size,
            onestep_mc_samples=config.onestep_mc_samples,
        )

        if (
            config.best_selection == "validation"
            and config.validation_interval > 0
            and (iteration == 1 or iteration % config.validation_interval == 0 or iteration == config.iterations)
        ):
            validation_metrics = evaluate_actor_for_training(
                actor=actor,
                objective=objective,
                env_config=env_config,
                v_star=v_star,
                episodes=config.validation_episodes,
                device=device,
                seed=config.seed + 50_000,
                vectorized=config.eval_vectorized,
                batch_size=config.eval_batch_size,
            )
            validation_metric = objective.select_metric(validation_metrics)
            if validation_metric < best_metric:
                is_best = True
                best_metric = validation_metric
                best_v_star = float(v_star)
                best_train_record = {
                    "iteration": float(iteration),
                    "selection_source": "validation",
                    "selection_metric": objective.selection_metric,
                    **validation_metrics,
                    "v_star": float(v_star),
                }
                best_actor_path, best_critic_path = save_checkpoint(actor, critic, output_dir, "best")

        if objective.uses_auxiliary_v and iteration % config.sigma_v_decay_every == 0:
            sigma_v = max(config.min_sigma_v, sigma_v * config.sigma_v_decay)

        row = {
            "iteration": float(iteration),
            "mean_cost": metrics["mean_cost"],
            "variance": metrics["variance"],
            "std_cost": metrics["std_cost"],
            "VaR": metrics["VaR"],
            "ES": metrics["ES"],
            "v_star": float(v_star),
            "sigma_v": float(sigma_v),
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "best_metric": best_metric,
            "is_best": float(is_best),
        }
        if validation_metrics is not None:
            row.update(
                {
                    "validation_mean_cost": validation_metrics["mean_cost"],
                    "validation_VaR": validation_metrics["VaR"],
                    "validation_ES": validation_metrics["ES"],
                    "validation_ES_0.8": validation_metrics["ES_0.8"],
                    "validation_ES_0.6": validation_metrics["ES_0.6"],
                    "validation_AVar_0.8": validation_metrics["AVar_0.8"],
                    "validation_mean_var_utility": validation_metrics["mean_var_utility"],
                }
            )
        rows.append(row)

        if iteration == 1 or iteration % config.log_interval == 0:
            print(
                f"iter={iteration:04d} mean={metrics['mean_cost']:.4f} "
                f"ES={metrics['ES']:.4f} VaR={metrics['VaR']:.4f} "
                f"v_star={v_star:.4f} best_{objective.selection_metric}={best_metric:.4f} "
                f"critic_loss={critic_loss:.4f} actor_loss={actor_loss:.4f}"
            )

    if not best_train_record:
        best_v_star = float(v_star)
        best_train_record = {
            "iteration": float(config.iterations),
            "selection_source": "fallback_final",
            "selection_metric": objective.selection_metric,
            **metric_summary(total_costs, objective.alpha),
            "v_star": float(v_star),
        }
        best_actor_path, best_critic_path = save_checkpoint(actor, critic, output_dir, "best")

    save_metrics_csv(output_dir / "training_metrics.csv", rows)
    save_line_plot([float(r["mean_cost"]) for r in rows], "Mean total cost", "Training mean cost", output_dir / "training_mean_cost.png")
    save_line_plot(
        [float(r["ES"]) for r in rows],
        f"ES_{objective.alpha}",
        f"Training ES_{objective.alpha}",
        output_dir / "training_es.png",
    )
    save_line_plot([float(r["v_star"]) for r in rows], "v_star", "Auxiliary variable estimate", output_dir / "training_v_star.png")

    save_checkpoint(actor, critic, output_dir, "last")
    torch.save(actor.state_dict(), output_dir / "actor.pt")
    torch.save(critic.state_dict(), output_dir / "critic.pt")

    eval_seed = config.seed + 10_000
    eval_metrics = evaluate_actor_for_training(
        actor,
        objective,
        env_config,
        v_star,
        config.eval_episodes,
        device,
        eval_seed,
        config.eval_vectorized,
        config.eval_batch_size,
    )

    best_actor = ActorNetwork(action_scale=env_config.a_max).to(device)
    best_actor.load_state_dict(torch.load(best_actor_path, map_location=device))
    best_eval_metrics = evaluate_actor_for_training(
        best_actor,
        objective,
        env_config,
        best_v_star,
        config.eval_episodes,
        device,
        eval_seed,
        config.eval_vectorized,
        config.eval_batch_size,
    )
    zero_baseline_metrics = evaluate_zero_for_training(
        env_config,
        config.eval_episodes,
        objective.alpha,
        device,
        eval_seed,
        config.eval_vectorized,
        config.eval_batch_size,
    )

    if config.make_heatmap:
        save_policy_heatmap(actor, objective, env_config, v_star, output_dir / "policy_heatmap.png", device)
        save_policy_heatmap(best_actor, objective, env_config, best_v_star, output_dir / "best_policy_heatmap.png", device)

    best_record = {
        "objective": objective.name,
        "risk_alpha": objective.alpha,
        "risk_lambda": config.risk_lambda,
        "selection_metric": objective.selection_metric,
        "selection_mode": config.best_selection,
        "train_or_validation_record": best_train_record,
        "deterministic_eval": best_eval_metrics,
        "checkpoint_files": {
            "actor": best_actor_path.name,
            "critic": best_critic_path.name,
        },
    }
    with (output_dir / "best_record.json").open("w", encoding="utf-8") as f:
        json.dump(best_record, f, indent=2)
    with (output_dir / "zero_baseline.json").open("w", encoding="utf-8") as f:
        json.dump(zero_baseline_metrics, f, indent=2)

    save_metrics_csv(
        output_dir / "comparison_summary.csv",
        [
            comparison_row("always_zero", zero_baseline_metrics),
            comparison_row(f"{objective.name}_last", eval_metrics),
            comparison_row(f"{objective.name}_best", best_eval_metrics),
        ],
    )

    return {
        "objective": objective.name,
        "objective_alpha": objective.alpha,
        "risk_lambda": config.risk_lambda,
        "actor": actor,
        "critic": critic,
        "v_star": v_star,
        "best_v_star": best_v_star,
        "best_record": best_record,
        "rows": rows,
        "eval_metrics": eval_metrics,
        "best_eval_metrics": best_eval_metrics,
        "zero_baseline_metrics": zero_baseline_metrics,
        "output_dir": str(output_dir),
    }
