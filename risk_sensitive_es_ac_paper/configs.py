"""Configuration objects for the paper-aligned reproduction branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvConfig:
    T: int = 5
    a_max: float = 2.0
    q_max: float = 5.0
    kappa: float = 2.0
    mu: float = 1.0
    sigma: float = 0.2
    dt: Optional[float] = None
    phi: float = 0.005
    psi: float = 0.5
    price_min: float = 0.0
    price_max: float = 2.0

    def __post_init__(self) -> None:
        if self.dt is None:
            self.dt = 1.0 / float(self.T)


@dataclass
class TrainConfig:
    objective: str = "es"
    iterations: int = 300
    num_episodes: int = 256
    eval_episodes: int = 10_000
    critic_updates: int = 5
    actor_updates: int = 1
    batch_size: int = 512
    learning_rate: Optional[float] = None
    actor_learning_rate: float = 3e-5
    critic_learning_rate: float = 3e-4
    alpha: float = 0.8
    risk_alpha: Optional[float] = None
    risk_lambda: float = 0.5
    initial_v_star: float = 0.0
    sigma_v: float = 1.0
    sigma_v_decay: float = 0.98
    sigma_v_decay_every: int = 10
    min_sigma_v: float = 0.05
    seed: int = 7
    log_interval: int = 10
    output_dir: str = "outputs"
    device: str = "auto"
    make_heatmap: bool = True
    # 训练采样默认使用批量 rollout；需要和旧实现对照时可切回 scalar。
    rollout_mode: str = "batched"
    # 训练中的 validation/final eval 默认使用批量评估，避免标量评估拖慢实验。
    eval_vectorized: bool = True
    eval_batch_size: int = 65_536
    # 论文对齐版新增：best checkpoint 可以由训练 rollout 指标或独立验证集指标选择。
    best_selection: str = "rollout"
    validation_interval: int = 10
    validation_episodes: int = 1_000
    # OneStepES baseline uses state/action-conditional Monte Carlo samples to
    # estimate ES(cost_t + V_{t+1} | state_t, action_t).
    onestep_mc_samples: int = 64
    # Discrete preference grid for conditioned_composite_es. The first
    # prototype keeps alpha fixed and conditions primarily on lambda.
    conditioned_lambdas: str = "0,0.25,0.5,0.75,1"
    conditioned_alphas: str = "0.8"
    conditioned_eval_lambdas: str = "0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1"
    conditioned_calibration_rounds: int = 2
