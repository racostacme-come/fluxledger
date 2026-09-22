"""Periodic finite-volume transport and numerical diffusion diagnostics."""

from .solver import Result, fourier_average, solve, top_hat_average

__all__ = ["Result", "fourier_average", "solve", "top_hat_average"]
__version__ = "0.1.0"
