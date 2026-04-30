"""Vectorized deterministic evaluation.

This module evaluates many independent OU trading episodes in batches. It is
statistically equivalent to the scalar evaluator but not pathwise identical,
because it uses torch RNG instead of the environment's NumPy RNG.
"""

from __future__ import annotations

import math

import torch

from configs import EnvConfig
from networks import ActorNetwork
from objectives import Objective
from utils import build_conditioned_inputs


def _expectile_tensor(costs: torch.Tensor, alpha: float, iterations: int = 80) -> torch.Tensor:
    lo = costs.min()
    hi = costs.max()
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        left = (1.0 - alpha) * torch.relu(mid - costs).mean()
        right = alpha * torch.relu(costs - mid).mean()
        if bool((left < right).item()):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _es_tensor(costs: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    var_alpha = torch.quantile(costs, float(alpha))
    es = var_alpha + torch.relu(costs - var_alpha).mean() / (1.0 - alpha)
    return var_alpha, es


def _asymmetric_variance_tensor(costs: torch.Tensor, alpha: float) -> torch.Tensor:
    expectile = _expectile_tensor(costs, alpha)
    diff = costs - expectile
    score = alpha * torch.relu(diff).pow(2)
    score = score + (1.0 - alpha) * torch.relu(-diff).pow(2)
    return score.mean()


def _metric_summary_tensor(costs: torch.Tensor, alpha: float) -> dict[str, float]:
    costs = costs.reshape(-1)
    mean_cost = costs.mean()
    variance = costs.var(unbiased=False)
    var_alpha, es = _es_tensor(costs, alpha)
    var_08, es_08 = _es_tensor(costs, 0.8)
    var_06, es_06 = _es_tensor(costs, 0.6)
    return {
        "mean_cost": float(mean_cost.item()),
        "variance": float(variance.item()),
        "std_cost": float(torch.sqrt(variance).item()),
        "VaR": float(var_alpha.item()),
        "ES": float(es.item()),
        "VaR_0.8": float(var_08.item()),
        "ES_0.8": float(es_08.item()),
        "VaR_0.6": float(var_06.item()),
        "ES_0.6": float(es_06.item()),
        "AVar_0.8": float(_asymmetric_variance_tensor(costs, 0.8).item()),
        "mean_var_utility": float((mean_cost + variance).item()),
    }


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _randn(
    shape: tuple[int, int],
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.float32, device=device, generator=generator)


def _rand(
    shape: tuple[int, int],
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.rand(shape, dtype=torch.float32, device=device, generator=generator)


def _initial_state(
    batch_size: int,
    env_config: EnvConfig,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = env_config
    p0_std = math.sqrt(8.0 * cfg.sigma * cfg.sigma / cfg.kappa)
    p = cfg.mu + p0_std * _randn((batch_size, 1), device, generator)
    p = torch.clamp(p, cfg.price_min, cfg.price_max)
    q = (2.0 * _rand((batch_size, 1), device, generator) - 1.0) * cfg.q_max
    x = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    y = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    return p, q, x, y


def _evaluate_batch(
    actor: ActorNetwork | None,
    objective: Objective | None,
    env_config: EnvConfig,
    v_star: float,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    cfg = env_config
    p, q, x, y = _initial_state(batch_size, cfg, device, generator)
    total_cost = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    sqrt_dt = math.sqrt(cfg.dt)
    zero_action = torch.zeros_like(total_cost) if actor is None else None
    raw_v = torch.empty_like(total_cost) if actor is not None else None
    t_col = torch.empty_like(total_cost) if actor is not None else None

    with torch.inference_mode():
        for t in range(cfg.T):
            if actor is None:
                action = zero_action
            else:
                raw_v.fill_(float(v_star))
                input_v, input_y = objective.tensor_inputs(raw_v, y)  # type: ignore[union-attr]
                t_col.fill_(float(t) / float(cfg.T))
                obs = torch.cat([t_col, input_v, p, q, input_y], dim=1)
                action = actor.deterministic(obs).clamp(-cfg.a_max, cfg.a_max)

            eps = _randn((batch_size, 1), device, generator)
            p_next = p + cfg.kappa * (cfg.mu - p) * cfg.dt
            p_next = p_next + cfg.sigma * sqrt_dt * eps
            p_next = torch.clamp(p_next, cfg.price_min, cfg.price_max)

            q_next = torch.clamp(q + action, -cfg.q_max, cfg.q_max)
            x_next = x + q * (p_next - p) - cfg.phi * action.pow(2)
            if t == cfg.T - 1:
                x_next = x_next - cfg.psi * q_next.pow(2)

            cost = x - x_next
            total_cost = total_cost + cost
            y = y - cost
            p, q, x = p_next, q_next, x_next

    return total_cost.squeeze(1)


def _evaluate_costs(
    actor: ActorNetwork | None,
    objective: Objective | None,
    env_config: EnvConfig,
    v_star: float,
    episodes: int,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> torch.Tensor:
    generator = _make_generator(device, seed)
    costs = torch.empty((episodes,), dtype=torch.float32, device=device)
    offset = 0
    while offset < episodes:
        current = min(batch_size, episodes - offset)
        costs[offset : offset + current] = _evaluate_batch(
            actor,
            objective,
            env_config,
            v_star,
            current,
            device,
            generator,
        )
        offset += current
    return costs


def evaluate_actor_batched(
    actor: ActorNetwork,
    objective: Objective,
    env_config: EnvConfig,
    v_star: float,
    episodes: int,
    device: torch.device,
    seed: int,
    batch_size: int = 65_536,
) -> dict[str, float]:
    was_training = actor.training
    actor.eval()
    costs = _evaluate_costs(actor, objective, env_config, v_star, episodes, device, seed, batch_size)
    actor.train(was_training)
    return _metric_summary_tensor(costs, objective.alpha)


def evaluate_zero_policy_batched(
    env_config: EnvConfig,
    episodes: int,
    alpha: float,
    device: torch.device,
    seed: int,
    batch_size: int = 65_536,
) -> dict[str, float]:
    costs = _evaluate_costs(None, None, env_config, 0.0, episodes, device, seed, batch_size)
    return _metric_summary_tensor(costs, alpha)


def _conditioned_network_inputs(
    t: int,
    v: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    y: torch.Tensor,
    risk_lambda: float,
    alpha: float,
    T: int,
) -> torch.Tensor:
    if risk_lambda <= 0.0:
        input_v = torch.zeros_like(v)
        input_y = torch.zeros_like(y)
    else:
        input_v = v
        input_y = y
    t_col = torch.full_like(v, float(t))
    return build_conditioned_inputs(t_col, input_v, p, q, input_y, risk_lambda, alpha, T)


def _evaluate_conditioned_batch(
    actor: ActorNetwork,
    env_config: EnvConfig,
    risk_lambda: float,
    alpha: float,
    v_star: float,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    cfg = env_config
    p, q, x, y = _initial_state(batch_size, cfg, device, generator)
    total_cost = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    sqrt_dt = math.sqrt(cfg.dt)
    raw_v = torch.full_like(total_cost, float(v_star))

    with torch.inference_mode():
        for t in range(cfg.T):
            obs = _conditioned_network_inputs(t, raw_v, p, q, y, risk_lambda, alpha, cfg.T)
            action = actor.deterministic(obs).clamp(-cfg.a_max, cfg.a_max)

            eps = _randn((batch_size, 1), device, generator)
            p_next = p + cfg.kappa * (cfg.mu - p) * cfg.dt
            p_next = p_next + cfg.sigma * sqrt_dt * eps
            p_next = torch.clamp(p_next, cfg.price_min, cfg.price_max)

            q_next = torch.clamp(q + action, -cfg.q_max, cfg.q_max)
            x_next = x + q * (p_next - p) - cfg.phi * action.pow(2)
            if t == cfg.T - 1:
                x_next = x_next - cfg.psi * q_next.pow(2)

            cost = x - x_next
            total_cost = total_cost + cost
            y = y - cost
            p, q, x = p_next, q_next, x_next

    return total_cost.squeeze(1)


def _evaluate_conditioned_costs(
    actor: ActorNetwork,
    env_config: EnvConfig,
    risk_lambda: float,
    alpha: float,
    v_star: float,
    episodes: int,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> torch.Tensor:
    generator = _make_generator(device, seed)
    costs = torch.empty((episodes,), dtype=torch.float32, device=device)
    offset = 0
    while offset < episodes:
        current = min(batch_size, episodes - offset)
        costs[offset : offset + current] = _evaluate_conditioned_batch(
            actor,
            env_config,
            risk_lambda,
            alpha,
            v_star,
            current,
            device,
            generator,
        )
        offset += current
    return costs


def estimate_conditioned_v_star_batched(
    actor: ActorNetwork,
    env_config: EnvConfig,
    risk_lambda: float,
    alpha: float,
    v_star: float,
    episodes: int,
    device: torch.device,
    seed: int,
    batch_size: int = 65_536,
) -> float:
    if risk_lambda <= 0.0:
        return 0.0
    was_training = actor.training
    actor.eval()
    costs = _evaluate_conditioned_costs(
        actor,
        env_config,
        risk_lambda,
        alpha,
        v_star,
        episodes,
        device,
        seed,
        batch_size,
    )
    actor.train(was_training)
    return float(torch.quantile(costs, float(alpha)).item())


def evaluate_conditioned_actor_batched(
    actor: ActorNetwork,
    env_config: EnvConfig,
    risk_lambda: float,
    alpha: float,
    v_star: float,
    episodes: int,
    device: torch.device,
    seed: int,
    batch_size: int = 65_536,
) -> dict[str, float]:
    was_training = actor.training
    actor.eval()
    costs = _evaluate_conditioned_costs(
        actor,
        env_config,
        risk_lambda,
        alpha,
        v_star,
        episodes,
        device,
        seed,
        batch_size,
    )
    actor.train(was_training)
    metrics = _metric_summary_tensor(costs, alpha)
    metrics["composite_metric"] = (1.0 - float(risk_lambda)) * metrics["mean_cost"] + float(risk_lambda) * metrics["ES"]
    return metrics
