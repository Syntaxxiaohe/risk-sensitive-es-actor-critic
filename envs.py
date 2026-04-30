"""OU statistical arbitrage environment used in the paper experiments."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from configs import EnvConfig


class OUStatArbEnv:
    """Finite-horizon OU statistical arbitrage simulator."""

    def __init__(self, config: Optional[EnvConfig] = None, seed: Optional[int] = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.P = 0.0
        self.Q = 0.0
        self.X = 0.0
        self.y = 0.0

    def reset(self, seed: Optional[int] = None) -> Tuple[int, float, float, float]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        cfg = self.config
        # implementation assumption: the paper writes N(mu, 8*sigma^2/kappa);
        # numpy receives the square root because its second argument is std.
        p0_std = math.sqrt(8.0 * cfg.sigma * cfg.sigma / cfg.kappa)
        self.P = float(np.clip(self.rng.normal(cfg.mu, p0_std), cfg.price_min, cfg.price_max))
        self.Q = float(self.rng.uniform(-cfg.q_max, cfg.q_max))
        self.X = 0.0
        self.y = 0.0
        self.t = 0
        return self.state

    @property
    def state(self) -> Tuple[int, float, float, float]:
        return self.t, self.P, self.Q, self.y

    def step(self, action: float):
        cfg = self.config
        if self.t >= cfg.T:
            raise RuntimeError("step() called after episode is done")

        t = self.t
        p_t = self.P
        q_t = self.Q
        x_t = self.X
        y_t = self.y

        a_t = float(np.clip(action, -cfg.a_max, cfg.a_max))
        eps = self.rng.normal(0.0, 1.0)
        p_next = p_t + cfg.kappa * (cfg.mu - p_t) * cfg.dt
        p_next += cfg.sigma * math.sqrt(cfg.dt) * eps
        p_next = float(np.clip(p_next, cfg.price_min, cfg.price_max))

        # implementation assumption: additive inventory dynamics with hard
        # bounds, matching the simplified reproduction requirement.
        q_next = float(np.clip(q_t + a_t, -cfg.q_max, cfg.q_max))
        x_next = x_t + q_t * (p_next - p_t) - cfg.phi * a_t * a_t

        done = (t + 1) >= cfg.T
        terminal_penalty = cfg.psi * q_next * q_next if done else 0.0
        if done:
            x_next -= terminal_penalty

        cost_t = x_t - x_next
        y_next = y_t - cost_t

        self.P = p_next
        self.Q = q_next
        self.X = x_next
        self.y = y_next
        self.t = t + 1

        info = {
            "action": a_t,
            "cost": float(cost_t),
            "terminal_penalty": float(terminal_penalty),
            "wealth": float(self.X),
        }
        return self.state, float(cost_t), done, info

