"""Actor and critic networks."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Type

import torch
from torch import nn
from torch.distributions import Normal


def make_mlp(
    input_dim: int,
    hidden_sizes: Iterable[int],
    output_dim: int,
    activation: Type[nn.Module] = nn.Tanh,
    output_activation: Optional[Type[nn.Module]] = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    """Tanh-squashed Gaussian policy.

    implementation assumption: the paper does not specify exact NN architecture,
    so this branch keeps the 2x64 Tanh MLP used by the prototype.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_sizes: Tuple[int, int] = (64, 64),
        action_scale: float = 2.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.action_scale = float(action_scale)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.body = make_mlp(input_dim, hidden_sizes, hidden_sizes[-1], nn.Tanh, nn.Tanh)
        self.mean_head = nn.Linear(hidden_sizes[-1], 1)
        self.log_std_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        dist = Normal(mean, log_std.exp())
        raw_action = dist.sample()
        squashed = torch.tanh(raw_action)
        action = self.action_scale * squashed
        log_prob = self._squashed_log_prob(dist, raw_action, squashed)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return self.action_scale * torch.tanh(mean)

    def log_prob_from_action(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        scaled = (action / self.action_scale).clamp(-1.0 + eps, 1.0 - eps)
        raw_action = 0.5 * (torch.log1p(scaled) - torch.log1p(-scaled))
        mean, log_std = self.forward(obs)
        dist = Normal(mean, log_std.exp())
        return self._squashed_log_prob(dist, raw_action, scaled)

    def _squashed_log_prob(
        self,
        dist: Normal,
        raw_action: torch.Tensor,
        squashed_action: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-6
        log_prob = dist.log_prob(raw_action)
        correction = torch.log(self.action_scale * (1.0 - squashed_action.pow(2)) + eps)
        return (log_prob - correction).sum(dim=-1, keepdim=True)


class CriticNetwork(nn.Module):
    """Risk-sensitive or risk-neutral value estimator."""

    def __init__(self, input_dim: int = 5, hidden_sizes: Tuple[int, int] = (64, 64)):
        super().__init__()
        self.net = make_mlp(input_dim, hidden_sizes, 1, nn.Tanh, None)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

