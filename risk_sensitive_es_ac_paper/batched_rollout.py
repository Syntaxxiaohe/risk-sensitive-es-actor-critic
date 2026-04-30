"""批量化训练 rollout。"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from configs import EnvConfig
from networks import ActorNetwork
from objectives import Objective
from utils import build_conditioned_inputs, build_inputs


def _make_generator(device: torch.device, seed: Optional[int]) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def _randn(shape: tuple[int, int], device: torch.device, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.float32, device=device, generator=generator)


def _rand(shape: tuple[int, int], device: torch.device, generator: torch.Generator) -> torch.Tensor:
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


def _sample_actions(
    actor: ActorNetwork,
    inputs: torch.Tensor,
    env_config: EnvConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    mean, log_std = actor(inputs)
    noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
    raw_action = mean + log_std.exp() * noise
    action = actor.action_scale * torch.tanh(raw_action)
    return action.clamp(-env_config.a_max, env_config.a_max)


def _append_step(
    storage: dict[str, list[torch.Tensor]],
    *,
    inputs: torch.Tensor,
    next_inputs: torch.Tensor,
    t: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    y: torch.Tensor,
    action: torch.Tensor,
    cost: torch.Tensor,
    done: torch.Tensor,
) -> None:
    storage["inputs"].append(inputs)
    storage["next_inputs"].append(next_inputs)
    storage["t"].append(t)
    storage["v"].append(v)
    storage["p"].append(p)
    storage["q"].append(q)
    storage["y"].append(y)
    storage["actions"].append(action)
    storage["costs"].append(cost)
    storage["done"].append(done)


def collect_rollouts_batched(
    actor: ActorNetwork,
    objective: Objective,
    env_config: EnvConfig,
    num_episodes: int,
    v_star: float,
    sigma_v: float,
    device: torch.device,
    seed: Optional[int] = None,
) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    """一次并行采样一批 episode，避免单步 CPU/GPU 往返。"""

    cfg = env_config
    generator = _make_generator(device, seed)
    p, q, x, y = _initial_state(num_episodes, cfg, device, generator)
    v = objective.sample_auxiliary_v_tensor(num_episodes, v_star, sigma_v, device, generator)
    total_cost = torch.zeros((num_episodes, 1), dtype=torch.float32, device=device)
    sqrt_dt = math.sqrt(cfg.dt)

    storage: dict[str, list[torch.Tensor]] = {
        "inputs": [],
        "next_inputs": [],
        "t": [],
        "v": [],
        "p": [],
        "q": [],
        "y": [],
        "actions": [],
        "costs": [],
        "done": [],
    }

    was_training = actor.training
    actor.eval()
    with torch.no_grad():
        for step in range(cfg.T):
            t = torch.full((num_episodes, 1), float(step), dtype=torch.float32, device=device)
            input_v, input_y = objective.tensor_inputs(v, y)
            inputs = build_inputs(t, input_v, p, q, input_y, cfg.T)
            action = _sample_actions(actor, inputs, cfg, generator)

            eps = _randn((num_episodes, 1), device, generator)
            p_next = p + cfg.kappa * (cfg.mu - p) * cfg.dt
            p_next = p_next + cfg.sigma * sqrt_dt * eps
            p_next = torch.clamp(p_next, cfg.price_min, cfg.price_max)

            q_next = torch.clamp(q + action, -cfg.q_max, cfg.q_max)
            x_next = x + q * (p_next - p) - cfg.phi * action.pow(2)
            is_terminal = step == cfg.T - 1
            if is_terminal:
                x_next = x_next - cfg.psi * q_next.pow(2)

            cost = x - x_next
            y_next = y - cost
            next_t = torch.full((num_episodes, 1), float(step + 1), dtype=torch.float32, device=device)
            next_input_v, next_input_y = objective.tensor_inputs(v, y_next)
            next_inputs = build_inputs(next_t, next_input_v, p_next, q_next, next_input_y, cfg.T)
            done = torch.full((num_episodes, 1), is_terminal, dtype=torch.bool, device=device)

            _append_step(
                storage,
                inputs=inputs,
                next_inputs=next_inputs,
                t=t,
                v=v,
                p=p,
                q=q,
                y=y,
                action=action,
                cost=cost,
                done=done,
            )

            total_cost = total_cost + cost
            p, q, x, y = p_next, q_next, x_next, y_next
    actor.train(was_training)

    tensors = {key: torch.cat(values, dim=0) for key, values in storage.items()}
    total_costs = total_cost.squeeze(1).detach().cpu().numpy().astype(np.float64)
    return tensors, total_costs


def _conditioned_network_inputs(
    t: torch.Tensor,
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
    return build_conditioned_inputs(t, input_v, p, q, input_y, risk_lambda, alpha, T)


def collect_conditioned_rollouts_batched(
    actor: ActorNetwork,
    objective: Objective,
    env_config: EnvConfig,
    num_episodes: int,
    risk_lambda: float,
    alpha: float,
    v_star: float,
    sigma_v: float,
    device: torch.device,
    seed: Optional[int] = None,
) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    """Collect one batched rollout for a single preference theta."""

    cfg = env_config
    generator = _make_generator(device, seed)
    p, q, x, y = _initial_state(num_episodes, cfg, device, generator)
    if risk_lambda <= 0.0:
        v = torch.zeros((num_episodes, 1), dtype=torch.float32, device=device)
    else:
        v = objective.sample_auxiliary_v_tensor(num_episodes, v_star, sigma_v, device, generator)
    total_cost = torch.zeros((num_episodes, 1), dtype=torch.float32, device=device)
    sqrt_dt = math.sqrt(cfg.dt)
    lambda_col = torch.full((num_episodes, 1), float(risk_lambda), dtype=torch.float32, device=device)
    alpha_col = torch.full((num_episodes, 1), float(alpha), dtype=torch.float32, device=device)

    storage: dict[str, list[torch.Tensor]] = {
        "inputs": [],
        "next_inputs": [],
        "t": [],
        "v": [],
        "p": [],
        "q": [],
        "y": [],
        "risk_lambda": [],
        "alpha": [],
        "actions": [],
        "costs": [],
        "done": [],
    }

    was_training = actor.training
    actor.eval()
    with torch.no_grad():
        for step in range(cfg.T):
            t = torch.full((num_episodes, 1), float(step), dtype=torch.float32, device=device)
            inputs = _conditioned_network_inputs(t, v, p, q, y, risk_lambda, alpha, cfg.T)
            action = _sample_actions(actor, inputs, cfg, generator)

            eps = _randn((num_episodes, 1), device, generator)
            p_next = p + cfg.kappa * (cfg.mu - p) * cfg.dt
            p_next = p_next + cfg.sigma * sqrt_dt * eps
            p_next = torch.clamp(p_next, cfg.price_min, cfg.price_max)

            q_next = torch.clamp(q + action, -cfg.q_max, cfg.q_max)
            x_next = x + q * (p_next - p) - cfg.phi * action.pow(2)
            is_terminal = step == cfg.T - 1
            if is_terminal:
                x_next = x_next - cfg.psi * q_next.pow(2)

            cost = x - x_next
            y_next = y - cost
            next_t = torch.full((num_episodes, 1), float(step + 1), dtype=torch.float32, device=device)
            next_inputs = _conditioned_network_inputs(next_t, v, p_next, q_next, y_next, risk_lambda, alpha, cfg.T)
            done = torch.full((num_episodes, 1), is_terminal, dtype=torch.bool, device=device)

            storage["inputs"].append(inputs)
            storage["next_inputs"].append(next_inputs)
            storage["t"].append(t)
            storage["v"].append(v)
            storage["p"].append(p)
            storage["q"].append(q)
            storage["y"].append(y)
            storage["risk_lambda"].append(lambda_col)
            storage["alpha"].append(alpha_col)
            storage["actions"].append(action)
            storage["costs"].append(cost)
            storage["done"].append(done)

            total_cost = total_cost + cost
            p, q, x, y = p_next, q_next, x_next, y_next
    actor.train(was_training)

    tensors = {key: torch.cat(values, dim=0) for key, values in storage.items()}
    total_costs = total_cost.squeeze(1).detach().cpu().numpy().astype(np.float64)
    return tensors, total_costs
