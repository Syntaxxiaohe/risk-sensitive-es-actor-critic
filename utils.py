"""Shared utilities."""

from __future__ import annotations

import csv
from pathlib import Path
import random
from typing import Mapping, Sequence, Union

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def ensure_dir(path: Union[str, Path]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_inputs(
    t: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    y: torch.Tensor,
    T: int,
) -> torch.Tensor:
    t_norm = t / float(T)
    return torch.cat([t_norm, v, p, q, y], dim=1)


def build_conditioned_inputs(
    t: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    y: torch.Tensor,
    risk_lambda: torch.Tensor | float,
    alpha: torch.Tensor | float,
    T: int,
) -> torch.Tensor:
    t_norm = t / float(T)
    if not torch.is_tensor(risk_lambda):
        risk_lambda = torch.full_like(t_norm, float(risk_lambda))
    if not torch.is_tensor(alpha):
        alpha = torch.full_like(t_norm, float(alpha))
    return torch.cat([t_norm, v, p, q, y, risk_lambda, alpha], dim=1)


def build_single_input(
    t: int,
    v: float,
    p: float,
    q: float,
    y: float,
    T: int,
    device: torch.device,
) -> torch.Tensor:
    arr = np.array([[float(t) / float(T), v, p, q, y]], dtype=np.float32)
    return torch.as_tensor(arr, device=device)


def build_conditioned_single_input(
    t: int,
    v: float,
    p: float,
    q: float,
    y: float,
    risk_lambda: float,
    alpha: float,
    T: int,
    device: torch.device,
) -> torch.Tensor:
    arr = np.array([[float(t) / float(T), v, p, q, y, risk_lambda, alpha]], dtype=np.float32)
    return torch.as_tensor(arr, device=device)


def sample_var(costs: Sequence[float], alpha: float) -> float:
    arr = np.asarray(costs, dtype=np.float64)
    return float(np.quantile(arr, alpha))


def sample_es(costs: Sequence[float], alpha: float) -> float:
    arr = np.asarray(costs, dtype=np.float64)
    var = sample_var(arr, alpha)
    return float(var + np.maximum(arr - var, 0.0).mean() / (1.0 - alpha))


def sample_expectile(costs: Sequence[float], alpha: float, iterations: int = 80) -> float:
    arr = np.asarray(costs, dtype=np.float64)
    lo = float(arr.min())
    hi = float(arr.max())
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        left = (1.0 - alpha) * np.maximum(mid - arr, 0.0).mean()
        right = alpha * np.maximum(arr - mid, 0.0).mean()
        if left < right:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sample_asymmetric_variance(costs: Sequence[float], alpha: float) -> float:
    arr = np.asarray(costs, dtype=np.float64)
    expectile = sample_expectile(arr, alpha)
    diff = arr - expectile
    score = alpha * np.maximum(diff, 0.0) ** 2
    score = score + (1.0 - alpha) * np.maximum(-diff, 0.0) ** 2
    return float(score.mean())


def metric_summary(costs: Sequence[float], alpha: float) -> dict[str, float]:
    arr = np.asarray(costs, dtype=np.float64)
    variance = float(arr.var())
    es_alpha = sample_es(arr, alpha)
    return {
        "mean_cost": float(arr.mean()),
        "variance": variance,
        "std_cost": float(arr.std()),
        "VaR": sample_var(arr, alpha),
        "ES": es_alpha,
        "VaR_0.8": sample_var(arr, 0.8),
        "ES_0.8": sample_es(arr, 0.8),
        "VaR_0.6": sample_var(arr, 0.6),
        "ES_0.6": sample_es(arr, 0.6),
        "AVar_0.8": sample_asymmetric_variance(arr, 0.8),
        "mean_var_utility": float(arr.mean() + variance),
    }


def save_metrics_csv(path: Union[str, Path], rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path = Path(path)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
