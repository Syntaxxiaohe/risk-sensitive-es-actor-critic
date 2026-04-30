"""Objective definitions.

The paper branch keeps objective-specific logic isolated here. Adding Var,
AVar, Mean-Var, or OneStepES should mostly extend this module plus tests,
instead of rewriting the trainer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from configs import EnvConfig
from networks import CriticNetwork
from utils import build_inputs, sample_expectile, sample_var


SUPPORTED_OBJECTIVES = (
    "es",
    "es08",
    "es06",
    "composite_es",
    "conditioned_composite_es",
    "mean",
    "var",
    "avar08",
    "mean_var",
    "onestep_es08",
    "onestep_es06",
)


def es_score_torch(total_cost: torch.Tensor, v: torch.Tensor, alpha: float) -> torch.Tensor:
    return v + torch.relu(total_cost - v) / (1.0 - alpha)


def composite_es_score_torch(
    total_cost: torch.Tensor,
    v: torch.Tensor,
    alpha: float,
    risk_lambda: float,
) -> torch.Tensor:
    return (1.0 - risk_lambda) * total_cost + risk_lambda * es_score_torch(total_cost, v, alpha)


def variance_score_torch(total_cost: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (total_cost - v).pow(2)


def asymmetric_variance_score_torch(total_cost: torch.Tensor, v: torch.Tensor, alpha: float) -> torch.Tensor:
    diff = total_cost - v
    return alpha * torch.relu(diff).pow(2) + (1.0 - alpha) * torch.relu(-diff).pow(2)


def mean_variance_score_torch(total_cost: torch.Tensor, v: torch.Tensor, lambda_var: float) -> torch.Tensor:
    return total_cost + float(lambda_var) * variance_score_torch(total_cost, v)


class Objective(ABC):
    name: str
    selection_metric: str
    uses_auxiliary_v = True

    def __init__(self, alpha: float):
        self.alpha = alpha

    @abstractmethod
    def tensor_inputs(
        self,
        v: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return network-facing v and y columns."""

    @abstractmethod
    def scalar_inputs(self, v: float, y: float) -> tuple[float, float]:
        """Return network-facing scalar v and y."""

    @abstractmethod
    def sample_auxiliary_v(self, rng: np.random.Generator, v_star: float, sigma_v: float) -> float:
        """Sample the trajectory-level auxiliary variable."""

    @abstractmethod
    def sample_auxiliary_v_tensor(
        self,
        batch_size: int,
        v_star: float,
        sigma_v: float,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """批量采样每条轨迹对应的辅助变量 v。"""

    @abstractmethod
    def update_v_star(self, total_costs: np.ndarray, previous_v_star: float) -> float:
        """Update the outer auxiliary variable estimate."""

    @abstractmethod
    def critic_target(
        self,
        critic: CriticNetwork,
        batch: dict[str, torch.Tensor],
        *,
        env_config: EnvConfig | None = None,
        mc_samples: int = 1,
    ) -> torch.Tensor:
        """Return Bellman target for critic and actor value signal."""

    @abstractmethod
    def select_metric(self, metrics: dict[str, float]) -> float:
        """Metric minimized when selecting best checkpoints."""


class ExpectedShortfallObjective(Objective):
    selection_metric = "ES"

    def __init__(self, alpha: float, name: str = "es"):
        super().__init__(alpha)
        self.name = name

    def tensor_inputs(
        self,
        v: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return v, y

    def scalar_inputs(self, v: float, y: float) -> tuple[float, float]:
        return v, y

    def sample_auxiliary_v(self, rng: np.random.Generator, v_star: float, sigma_v: float) -> float:
        return float(rng.normal(v_star, sigma_v))

    def sample_auxiliary_v_tensor(
        self,
        batch_size: int,
        v_star: float,
        sigma_v: float,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        noise = torch.randn((batch_size, 1), dtype=torch.float32, device=device, generator=generator)
        return float(v_star) + float(sigma_v) * noise

    def update_v_star(self, total_costs: np.ndarray, previous_v_star: float) -> float:
        return sample_var(total_costs, self.alpha)

    def critic_target(
        self,
        critic: CriticNetwork,
        batch: dict[str, torch.Tensor],
        *,
        env_config: EnvConfig | None = None,
        mc_samples: int = 1,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_value = critic(batch["next_inputs"])
            terminal_total_cost = batch["costs"] - batch["y"]
            terminal_score = es_score_torch(terminal_total_cost, batch["v"], self.alpha)
            return torch.where(batch["done"], terminal_score, next_value)

    def select_metric(self, metrics: dict[str, float]) -> float:
        return metrics["ES"]


class CompositeExpectedShortfallObjective(ExpectedShortfallObjective):
    """Fixed composite objective: (1-lambda) E[C] + lambda ES_alpha(C).

    The ES term follows the same total-cost scoring-function implementation as
    ``es`` and ``es06``. The mean term is added at the same terminal total-cost
    level, so this is distinct from the recursive OneStepES baseline.
    """

    selection_metric = "composite_es"

    def __init__(self, alpha: float, risk_lambda: float = 0.5, name: str = "composite_es"):
        if not 0.0 <= risk_lambda <= 1.0:
            raise ValueError(f"risk_lambda must be in [0, 1], got {risk_lambda}.")
        super().__init__(alpha, name=name)
        self.risk_lambda = float(risk_lambda)
        self.uses_auxiliary_v = self.risk_lambda > 0.0

    def tensor_inputs(
        self,
        v: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.risk_lambda <= 0.0:
            return torch.zeros_like(v), torch.zeros_like(y)
        return v, y

    def scalar_inputs(self, v: float, y: float) -> tuple[float, float]:
        if self.risk_lambda <= 0.0:
            return 0.0, 0.0
        return v, y

    def sample_auxiliary_v(self, rng: np.random.Generator, v_star: float, sigma_v: float) -> float:
        if self.risk_lambda <= 0.0:
            return 0.0
        return super().sample_auxiliary_v(rng, v_star, sigma_v)

    def sample_auxiliary_v_tensor(
        self,
        batch_size: int,
        v_star: float,
        sigma_v: float,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if self.risk_lambda <= 0.0:
            return torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        return super().sample_auxiliary_v_tensor(batch_size, v_star, sigma_v, device, generator)

    def update_v_star(self, total_costs: np.ndarray, previous_v_star: float) -> float:
        if self.risk_lambda <= 0.0:
            return 0.0
        return sample_var(total_costs, self.alpha)

    def critic_target(
        self,
        critic: CriticNetwork,
        batch: dict[str, torch.Tensor],
        *,
        env_config: EnvConfig | None = None,
        mc_samples: int = 1,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_value = critic(batch["next_inputs"])
            if self.risk_lambda <= 0.0:
                nonterminal = (~batch["done"]).float()
                return batch["costs"] + nonterminal * next_value
            terminal_total_cost = batch["costs"] - batch["y"]
            terminal_score = composite_es_score_torch(
                terminal_total_cost,
                batch["v"],
                self.alpha,
                self.risk_lambda,
            )
            return torch.where(batch["done"], terminal_score, next_value)

    def select_metric(self, metrics: dict[str, float]) -> float:
        return (1.0 - self.risk_lambda) * metrics["mean_cost"] + self.risk_lambda * metrics["ES"]


class ConditionedCompositeExpectedShortfallObjective(Objective):
    """Preference-conditioned composite ES objective.

    This objective expects rollout batches to contain ``risk_lambda`` and
    ``alpha`` columns. The first prototype uses one theta per rollout batch, but
    the target is written row-wise so mini-batches remain valid after shuffling.
    """

    name = "conditioned_composite_es"
    selection_metric = "average_composite_metric"

    def tensor_inputs(
        self,
        v: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return v, y

    def scalar_inputs(self, v: float, y: float) -> tuple[float, float]:
        return v, y

    def sample_auxiliary_v(self, rng: np.random.Generator, v_star: float, sigma_v: float) -> float:
        return float(rng.normal(v_star, sigma_v))

    def sample_auxiliary_v_tensor(
        self,
        batch_size: int,
        v_star: float,
        sigma_v: float,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        noise = torch.randn((batch_size, 1), dtype=torch.float32, device=device, generator=generator)
        return float(v_star) + float(sigma_v) * noise

    def update_v_star(self, total_costs: np.ndarray, previous_v_star: float) -> float:
        return sample_var(total_costs, self.alpha)

    def critic_target(
        self,
        critic: CriticNetwork,
        batch: dict[str, torch.Tensor],
        *,
        env_config: EnvConfig | None = None,
        mc_samples: int = 1,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_value = critic(batch["next_inputs"])
            nonterminal = (~batch["done"]).float()
            mean_target = batch["costs"] + nonterminal * next_value

            risk_lambda = batch["risk_lambda"]
            alpha = batch["alpha"].clamp(max=1.0 - 1e-6)
            terminal_total_cost = batch["costs"] - batch["y"]
            es_score = batch["v"] + torch.relu(terminal_total_cost - batch["v"]) / (1.0 - alpha)
            terminal_score = (1.0 - risk_lambda) * terminal_total_cost + risk_lambda * es_score
            scoring_target = torch.where(batch["done"], terminal_score, next_value)

            return torch.where(risk_lambda <= 0.0, mean_target, scoring_target)

    def select_metric(self, metrics: dict[str, float]) -> float:
        return metrics.get("composite_metric", metrics["ES"])


class MeanObjective(Objective):
    name = "mean"
    selection_metric = "mean_cost"
    uses_auxiliary_v = False

    def tensor_inputs(
        self,
        v: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros_like(v), torch.zeros_like(y)

    def scalar_inputs(self, v: float, y: float) -> tuple[float, float]:
        return 0.0, 0.0

    def sample_auxiliary_v(self, rng: np.random.Generator, v_star: float, sigma_v: float) -> float:
        return 0.0

    def sample_auxiliary_v_tensor(
        self,
        batch_size: int,
        v_star: float,
        sigma_v: float,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        return torch.zeros((batch_size, 1), dtype=torch.float32, device=device)

    def update_v_star(self, total_costs: np.ndarray, previous_v_star: float) -> float:
        return 0.0

    def critic_target(
        self,
        critic: CriticNetwork,
        batch: dict[str, torch.Tensor],
        *,
        env_config: EnvConfig | None = None,
        mc_samples: int = 1,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_value = critic(batch["next_inputs"])
            nonterminal = (~batch["done"]).float()
            return batch["costs"] + nonterminal * next_value

    def select_metric(self, metrics: dict[str, float]) -> float:
        return metrics["mean_cost"]


class VarianceObjective(Objective):
    name = "var"
    selection_metric = "variance"

    def tensor_inputs(
        self,
        v: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return v, y

    def scalar_inputs(self, v: float, y: float) -> tuple[float, float]:
        return v, y

    def sample_auxiliary_v(self, rng: np.random.Generator, v_star: float, sigma_v: float) -> float:
        return float(rng.normal(v_star, sigma_v))

    def sample_auxiliary_v_tensor(
        self,
        batch_size: int,
        v_star: float,
        sigma_v: float,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        noise = torch.randn((batch_size, 1), dtype=torch.float32, device=device, generator=generator)
        return float(v_star) + float(sigma_v) * noise

    def update_v_star(self, total_costs: np.ndarray, previous_v_star: float) -> float:
        return float(np.mean(total_costs))

    def critic_target(
        self,
        critic: CriticNetwork,
        batch: dict[str, torch.Tensor],
        *,
        env_config: EnvConfig | None = None,
        mc_samples: int = 1,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_value = critic(batch["next_inputs"])
            terminal_total_cost = batch["costs"] - batch["y"]
            terminal_score = variance_score_torch(terminal_total_cost, batch["v"])
            return torch.where(batch["done"], terminal_score, next_value)

    def select_metric(self, metrics: dict[str, float]) -> float:
        return metrics["variance"]


class AsymmetricVarianceObjective(VarianceObjective):
    selection_metric = "AVar_0.8"

    def __init__(self, alpha: float = 0.8, name: str = "avar08"):
        super().__init__(alpha)
        self.name = name

    def update_v_star(self, total_costs: np.ndarray, previous_v_star: float) -> float:
        return sample_expectile(total_costs, self.alpha)

    def critic_target(
        self,
        critic: CriticNetwork,
        batch: dict[str, torch.Tensor],
        *,
        env_config: EnvConfig | None = None,
        mc_samples: int = 1,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_value = critic(batch["next_inputs"])
            terminal_total_cost = batch["costs"] - batch["y"]
            terminal_score = asymmetric_variance_score_torch(terminal_total_cost, batch["v"], self.alpha)
            return torch.where(batch["done"], terminal_score, next_value)

    def select_metric(self, metrics: dict[str, float]) -> float:
        return metrics["AVar_0.8"]


class MeanVarianceObjective(VarianceObjective):
    name = "mean_var"
    selection_metric = "mean_var_utility"

    def __init__(self, alpha: float, lambda_var: float = 1.0):
        super().__init__(alpha)
        self.lambda_var = float(lambda_var)

    def critic_target(
        self,
        critic: CriticNetwork,
        batch: dict[str, torch.Tensor],
        *,
        env_config: EnvConfig | None = None,
        mc_samples: int = 1,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_value = critic(batch["next_inputs"])
            terminal_total_cost = batch["costs"] - batch["y"]
            terminal_score = mean_variance_score_torch(terminal_total_cost, batch["v"], self.lambda_var)
            return torch.where(batch["done"], terminal_score, next_value)

    def select_metric(self, metrics: dict[str, float]) -> float:
        return metrics["mean_var_utility"]


class OneStepExpectedShortfallObjective(Objective):
    """Recursive one-step ES baseline.

    This follows the paper's RL-OneStepES baseline: ES is applied at each
    Bellman step to cost_t + V_{t+1}, rather than once to the terminal
    total-cost distribution.
    """

    uses_auxiliary_v = False

    def __init__(self, alpha: float, name: str):
        super().__init__(alpha)
        self.name = name
        self.selection_metric = f"ES_{alpha:.1f}"

    def tensor_inputs(
        self,
        v: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros_like(v), torch.zeros_like(y)

    def scalar_inputs(self, v: float, y: float) -> tuple[float, float]:
        return 0.0, 0.0

    def sample_auxiliary_v(self, rng: np.random.Generator, v_star: float, sigma_v: float) -> float:
        return 0.0

    def sample_auxiliary_v_tensor(
        self,
        batch_size: int,
        v_star: float,
        sigma_v: float,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        return torch.zeros((batch_size, 1), dtype=torch.float32, device=device)

    def update_v_star(self, total_costs: np.ndarray, previous_v_star: float) -> float:
        return 0.0

    def critic_target(
        self,
        critic: CriticNetwork,
        batch: dict[str, torch.Tensor],
        *,
        env_config: EnvConfig | None = None,
        mc_samples: int = 1,
    ) -> torch.Tensor:
        if env_config is None:
            raise ValueError("OneStepES critic target requires env_config.")
        if mc_samples < 2:
            raise ValueError("OneStepES critic target requires at least 2 Monte Carlo samples.")

        with torch.no_grad():
            cfg = env_config
            batch_size = batch["p"].shape[0]
            sample_count = int(mc_samples)
            p = batch["p"].repeat_interleave(sample_count, dim=0)
            q = batch["q"].repeat_interleave(sample_count, dim=0)
            action = batch["actions"].repeat_interleave(sample_count, dim=0)
            t = batch["t"].repeat_interleave(sample_count, dim=0)

            eps = torch.randn((batch_size * sample_count, 1), dtype=p.dtype, device=p.device)
            p_next = p + cfg.kappa * (cfg.mu - p) * cfg.dt
            p_next = p_next + cfg.sigma * float(np.sqrt(cfg.dt)) * eps
            p_next = torch.clamp(p_next, cfg.price_min, cfg.price_max)

            q_next = torch.clamp(q + action, -cfg.q_max, cfg.q_max)
            cost = -q * (p_next - p) + cfg.phi * action.pow(2)
            terminal = t >= float(cfg.T - 1)
            cost = cost + terminal.float() * cfg.psi * q_next.pow(2)

            next_t = t + 1.0
            zero = torch.zeros_like(cost)
            next_inputs = build_inputs(next_t, zero, p_next, q_next, zero, cfg.T)
            next_value = critic(next_inputs)
            one_step_return = cost + (~terminal).float() * next_value
            one_step_return = one_step_return.view(batch_size, sample_count)

            local_var = torch.quantile(one_step_return, float(self.alpha), dim=1, keepdim=True)
            local_es = local_var + torch.relu(one_step_return - local_var).mean(dim=1, keepdim=True) / (
                1.0 - self.alpha
            )
            return local_es

    def select_metric(self, metrics: dict[str, float]) -> float:
        return metrics["ES"]


def make_objective(name: str, alpha: float, risk_lambda: float = 0.5) -> Objective:
    if name == "es":
        return ExpectedShortfallObjective(alpha, name="es")
    if name == "es08":
        return ExpectedShortfallObjective(0.8, name="es08")
    if name == "es06":
        return ExpectedShortfallObjective(0.6, name="es06")
    if name == "composite_es":
        return CompositeExpectedShortfallObjective(alpha, risk_lambda=risk_lambda, name="composite_es")
    if name == "conditioned_composite_es":
        return ConditionedCompositeExpectedShortfallObjective(alpha)
    if name == "mean":
        return MeanObjective(alpha)
    if name == "var":
        return VarianceObjective(alpha)
    if name == "avar08":
        return AsymmetricVarianceObjective(0.8, name="avar08")
    if name == "mean_var":
        return MeanVarianceObjective(alpha, lambda_var=1.0)
    if name == "onestep_es08":
        return OneStepExpectedShortfallObjective(0.8, name="onestep_es08")
    if name == "onestep_es06":
        return OneStepExpectedShortfallObjective(0.6, name="onestep_es06")
    raise ValueError(f"Unsupported objective: {name}")
