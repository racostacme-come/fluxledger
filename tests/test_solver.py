import numpy as np
import pytest

from fluxledger import fourier_average, solve, top_hat_average


@pytest.mark.parametrize("scheme", ["upwind", "mc"])
@pytest.mark.parametrize("velocity", [-1.3, 0.0, 1.3])
@pytest.mark.parametrize("diffusivity", [0.0, 0.02])
def test_conservation_bounds_and_total_variation(scheme, velocity, diffusivity):
    initial = np.random.default_rng(927).uniform(-0.2, 1.5, 80)
    original = initial.copy()
    result = solve(
        initial, scheme=scheme, velocity=velocity, diffusivity=diffusivity, duration=0.2, cfl=1
    )
    assert abs(result.mass_drift) < 2e-13
    assert result.values.min() >= initial.min() - 1e-13
    assert result.values.max() <= initial.max() + 1e-13
    tv = lambda u: np.sum(np.abs(u - np.roll(u, 1)))  # noqa: E731
    assert tv(result.values) <= tv(initial) + 1e-12
    np.testing.assert_array_equal(initial, original)


@pytest.mark.parametrize("scheme", ["upwind", "mc"])
def test_constant_and_zero_duration(scheme):
    np.testing.assert_allclose(solve(np.full(40, 3.0), scheme=scheme).values, 3, atol=1e-14)
    initial = np.arange(10.0)
    result = solve(initial, duration=0, scheme=scheme)
    np.testing.assert_array_equal(result.values, initial)
    assert result.steps == 0
    assert not np.shares_memory(initial, result.values)


def test_reflection_negative_velocity():
    initial = top_hat_average(91)
    positive = solve(initial, velocity=1.2, duration=0.37).values
    negative = solve(initial[::-1], velocity=-1.2, duration=0.37).values
    np.testing.assert_allclose(positive, negative[::-1], atol=2e-14)


@pytest.mark.parametrize("velocity,diffusivity", [(1.0, 0), (-0.7, 0.015), (0, 0.03)])
def test_mc_second_order_convergence(velocity, diffusivity):
    errors = []
    for n in (64, 128, 256):
        initial = fourier_average(n)
        result = solve(initial, velocity=velocity, diffusivity=diffusivity, duration=0.2)
        exact = fourier_average(n, velocity=velocity, diffusivity=diffusivity, time=0.2)
        errors.append(np.mean(np.abs(result.values - exact)))
    assert np.log2(errors[-2] / errors[-1]) > 1.7


def test_upwind_discrete_fourier_amplification():
    # Independent closed-form eigenvalue of the spatial stencil + RK stability polynomial.
    n, a, nu, time = 50, -0.9, 0.002, 0.4
    initial = fourier_average(n, mode=3)
    result = solve(initial, velocity=a, diffusivity=nu, duration=time, scheme="upwind")
    theta = 2 * np.pi * 3 / n
    eigenvalue = -abs(a) * n * (1 - np.exp(1j * theta)) - 4 * nu * n**2 * np.sin(theta / 2) ** 2
    z = result.dt * eigenvalue
    expected = (1 + z + z**2 / 2 + z**3 / 6) ** result.steps
    measured = np.fft.rfft(result.values)[3] / np.fft.rfft(initial)[3]
    np.testing.assert_allclose(measured, expected, atol=2e-14)
    assert result.steps * result.dt == pytest.approx(time)
    assert result.dt * (2 * abs(a) * n + 2 * nu * n**2) <= 0.8 + 1e-15


@pytest.mark.parametrize(
    "kwargs",
    [
        {"velocity": np.nan},
        {"diffusivity": -1},
        {"length": 0},
        {"duration": -1},
        {"cfl": 0},
        {"cfl": 1.01},
        {"scheme": "unknown"},
        {"max_steps": 1},
        {"max_steps": 0},
        {"max_steps": True},
        {"velocity": np.inf},
        {"length": 1e-320},
    ],
)
def test_reject_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        solve(np.ones(16), **kwargs)


@pytest.mark.parametrize("initial", [[1, 2], [1, 2, 3, np.nan], [[1, 2], [3, 4]], [1j] * 4])
def test_reject_bad_data(initial):
    with pytest.raises(ValueError):
        solve(initial)


def test_exact_average_helpers():
    assert np.sum(
        top_hat_average(17, length=2, left=0.21, right=0.53)
    ) * 2 / 17 == pytest.approx(0.32)
    # Average of sine over first cell, integrated analytically without sinc helper.
    expected = 1 + 0.5 * (1 - np.cos(2 * np.pi / 16)) * 16 / (2 * np.pi)
    assert fourier_average(16)[0] == pytest.approx(expected)
    for kwargs in ({"mode": 8}, {"time": -1}, {"velocity": np.nan}, {"length": 0}):
        with pytest.raises(ValueError):
            fourier_average(16, **kwargs)
    with pytest.raises(ValueError):
        top_hat_average(20, left=0.8, right=0.2)
