"""Rollout buffer."""

from __future__ import annotations

import torch

from objectives import Objective
from utils import build_inputs


class RolloutBuffer:
    def __init__(self) -> None:
        self.t: list[float] = []
        self.p: list[float] = []
        self.q: list[float] = []
        self.y: list[float] = []
        self.v: list[float] = []
        self.action: list[float] = []
        self.cost: list[float] = []
        self.p_next: list[float] = []
        self.q_next: list[float] = []
        self.y_next: list[float] = []
        self.done: list[float] = []

    def add(
        self,
        t: int,
        p: float,
        q: float,
        y: float,
        v: float,
        action: float,
        cost: float,
        p_next: float,
        q_next: float,
        y_next: float,
        done: bool,
    ) -> None:
        self.t.append(float(t))
        self.p.append(float(p))
        self.q.append(float(q))
        self.y.append(float(y))
        self.v.append(float(v))
        self.action.append(float(action))
        self.cost.append(float(cost))
        self.p_next.append(float(p_next))
        self.q_next.append(float(q_next))
        self.y_next.append(float(y_next))
        self.done.append(float(done))

    def __len__(self) -> int:
        return len(self.t)

    def as_tensors(self, T: int, device: torch.device, objective: Objective) -> dict[str, torch.Tensor]:
        def col(values: list[float]) -> torch.Tensor:
            return torch.as_tensor(values, dtype=torch.float32, device=device).view(-1, 1)

        t = col(self.t)
        v = col(self.v)
        p = col(self.p)
        q = col(self.q)
        y = col(self.y)
        y_next = col(self.y_next)
        input_v, input_y = objective.tensor_inputs(v, y)
        next_input_v, next_input_y = objective.tensor_inputs(v, y_next)
        next_t = t + 1.0

        return {
            "inputs": build_inputs(t, input_v, p, q, input_y, T),
            "next_inputs": build_inputs(next_t, next_input_v, col(self.p_next), col(self.q_next), next_input_y, T),
            "t": t,
            "v": v,
            "p": p,
            "q": q,
            "y": y,
            "actions": col(self.action),
            "costs": col(self.cost),
            "done": col(self.done).bool(),
        }

