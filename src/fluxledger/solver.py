"""Public API; all solution arrays represent cell averages, not point samples."""

from dataclasses import dataclass
from operator import index

import numpy as np

from . import _core


@dataclass(frozen=True)
class Result:
    """Final state and integration ledger. Masses use the rectangle rule on averages."""

    centers: np.ndarray
    values: np.ndarray
    duration: float
    steps: int
    dt: float
    initial_mass: float
    final_mass: float

    @property
    def mass_drift(self) -> float:
        return self.final_mass - self.initial_mass


def _grid(cells: int, length: float) -> np.ndarray:
    if isinstance(cells, bool) or index(cells) < 4:
        raise ValueError("cells must be an integer >=4")
    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be finite and positive")
    return (np.arange(cells, dtype=float) + 0.5) * (length / cells)


def fourier_average(
    cells: int,
    *,
    length: float = 1.0,
    time: float = 0.0,
    velocity: float = 1.0,
    diffusivity: float = 0.0,
    mode: int = 1,
) -> np.ndarray:
    """Exact averages of 1 + 0.5 sin(k(x-a t)) exp(-nu k^2 t).

    Modes at or above Nyquist are rejected to keep validation unaliased.
    """
    centers = _grid(cells, length)
    if isinstance(mode, bool) or not 1 <= index(mode) < cells / 2:
        raise ValueError("mode must be a positive integer below Nyquist")
    if not all(np.isfinite(v) for v in (time, velocity, diffusivity)):
        raise ValueError("parameters must be finite")
    if time < 0 or diffusivity < 0:
        raise ValueError("time and diffusivity must be nonnegative")
    k = 2 * np.pi * mode / length
    return 1 + 0.5 * np.sinc(mode / cells) * np.exp(-diffusivity * k * k * time) * np.sin(
        k * (centers - velocity * time)
    )


def top_hat_average(
    cells: int, *, length: float = 1.0, left: float = 0.2, right: float = 0.4
) -> np.ndarray:
    """Exact cell averages of a unit pulse on [left, right] inside [0, length]."""
    _grid(cells, length)
    if not 0 <= left < right <= length:
        raise ValueError("require 0 <= left < right <= length")
    edges = np.linspace(0, length, cells + 1)
    return np.maximum(0, np.minimum(edges[1:], right) - np.maximum(edges[:-1], left)) / (
        length / cells
    )


def solve(
    initial,
    *,
    velocity: float = 1.0,
    diffusivity: float = 0.0,
    length: float = 1.0,
    duration: float = 1.0,
    cfl: float = 0.8,
    scheme: str = "mc",
    max_steps: int = 1_000_000,
) -> Result:
    """Evolve periodic cell averages to duration; input is copied and never mutated.

    cfl is a fraction of 1/(2|a|/dx + 2 nu/dx^2), not the advective CFL.
    A uniform step is shortened to land exactly on the requested final time.
    """
    if isinstance(max_steps, bool) or index(max_steps) <= 0:
        raise ValueError("max_steps must be a positive integer")
    raw = np.asarray(initial)
    if np.iscomplexobj(raw):
        raise ValueError("initial values must be real")
    values = np.asarray(raw, dtype=float)
    output, steps, dt = _core.solve(
        values, velocity, diffusivity, length, duration, cfl, scheme, max_steps
    )
    centers = _grid(len(values), length)
    dx = length / len(values)
    return Result(
        centers,
        output,
        duration,
        steps,
        dt,
        float(np.sum(values) * dx),
        float(np.sum(output) * dx),
    )
