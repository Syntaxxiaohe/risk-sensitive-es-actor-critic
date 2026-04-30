"""Trajectory collection."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from buffers import RolloutBuffer
from configs import EnvConfig
from envs import OUStatArbEnv
from networks import ActorNetwork
from objectives import Objective
from utils import build_single_input


def collect_rollouts(
    actor: ActorNetwork,
    objective: Objective,
    env_config: EnvConfig,
    num_episodes: int,
    v_star: float,
    sigma_v: float,
    device: torch.device,
    seed: Optional[int] = None,
) -> tuple[RolloutBuffer, np.ndarray]:
    rng = np.random.default_rng(seed)
    env = OUStatArbEnv(env_config, seed=seed)
    buffer = RolloutBuffer()
    total_costs: list[float] = []

    actor.eval()
    for _ in range(num_episodes):
        t, p, q, y = env.reset()
        v = objective.sample_auxiliary_v(rng, v_star, sigma_v)
        total_cost = 0.0
        done = False
        while not done:
            input_v, input_y = objective.scalar_inputs(v, y)
            obs = build_single_input(t, input_v, p, q, input_y, env_config.T, device)
            with torch.no_grad():
                action_tensor, _ = actor.sample(obs)
            action = float(action_tensor.cpu().numpy()[0, 0])

            (t_next, p_next, q_next, y_next), cost, done, info = env.step(action)
            buffer.add(
                t=t,
                p=p,
                q=q,
                y=y,
                v=v,
                action=info["action"],
                cost=cost,
                p_next=p_next,
                q_next=q_next,
                y_next=y_next,
                done=done,
            )
            total_cost += cost
            t, p, q, y = t_next, p_next, q_next, y_next

        total_costs.append(total_cost)

    actor.train()
    return buffer, np.asarray(total_costs, dtype=np.float64)

