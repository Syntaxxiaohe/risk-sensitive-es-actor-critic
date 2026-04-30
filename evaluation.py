"""Deterministic evaluation and plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from configs import EnvConfig
from envs import OUStatArbEnv
from networks import ActorNetwork
from objectives import Objective
from utils import build_single_input, metric_summary


def evaluate_actor(
    actor: ActorNetwork,
    objective: Objective,
    env_config: EnvConfig,
    v_star: float,
    episodes: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    env = OUStatArbEnv(env_config, seed=seed)
    total_costs: list[float] = []

    actor.eval()
    for _ in range(episodes):
        t, p, q, y = env.reset()
        total_cost = 0.0
        done = False
        while not done:
            input_v, input_y = objective.scalar_inputs(v_star, y)
            obs = build_single_input(t, input_v, p, q, input_y, env_config.T, device)
            with torch.no_grad():
                action = float(actor.deterministic(obs).cpu().numpy()[0, 0])
            (t, p, q, y), cost, done, _ = env.step(action)
            total_cost += cost
        total_costs.append(total_cost)

    actor.train()
    return metric_summary(total_costs, objective.alpha)


def evaluate_zero_policy(
    env_config: EnvConfig,
    episodes: int,
    alpha: float,
    seed: int,
) -> dict[str, float]:
    env = OUStatArbEnv(env_config, seed=seed)
    total_costs: list[float] = []
    for _ in range(episodes):
        env.reset()
        total_cost = 0.0
        done = False
        while not done:
            _, cost, done, _ = env.step(0.0)
            total_cost += cost
        total_costs.append(total_cost)
    return metric_summary(total_costs, alpha)


def save_line_plot(values: list[float], ylabel: str, title: str, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(values) + 1), values, linewidth=1.8)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_policy_heatmap(
    actor: ActorNetwork,
    objective: Objective,
    env_config: EnvConfig,
    v_star: float,
    output_path: Path,
    device: torch.device,
    grid_size: int = 80,
) -> None:
    p_grid = np.linspace(env_config.price_min, env_config.price_max, grid_size)
    q_grid = np.linspace(-env_config.q_max, env_config.q_max, grid_size)
    pp, qq = np.meshgrid(p_grid, q_grid, indexing="ij")
    input_v, input_y = objective.scalar_inputs(v_star, 0.0)
    obs = np.stack(
        [
            np.zeros_like(pp).reshape(-1),
            np.full_like(pp, input_v).reshape(-1),
            pp.reshape(-1),
            qq.reshape(-1),
            np.full_like(pp, input_y).reshape(-1),
        ],
        axis=1,
    ).astype(np.float32)

    with torch.no_grad():
        actions = actor.deterministic(torch.as_tensor(obs, device=device)).cpu().numpy()
    actions = actions.reshape(grid_size, grid_size)

    plt.figure(figsize=(7, 5))
    im = plt.imshow(
        actions,
        origin="lower",
        aspect="auto",
        extent=[-env_config.q_max, env_config.q_max, env_config.price_min, env_config.price_max],
        cmap="coolwarm",
        vmin=-env_config.a_max,
        vmax=env_config.a_max,
    )
    plt.xlabel("Inventory Q")
    plt.ylabel("Price P")
    plt.title(f"{objective.name} deterministic action at t=0, y=0")
    plt.colorbar(im, label="action")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

