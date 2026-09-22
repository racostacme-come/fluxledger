#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fluxledger {
enum class Scheme { upwind, mc };
struct Result {
    std::vector<double> values;
    std::size_t steps;
    double dt;
};

inline Scheme parse_scheme(const std::string &name) {
    if (name == "upwind")
        return Scheme::upwind;
    if (name == "mc")
        return Scheme::mc;
    throw std::invalid_argument("scheme must be 'upwind' or 'mc'");
}

inline double minmod(double a, double b, double c) {
    if (a > 0 && b > 0 && c > 0)
        return std::min({a, b, c});
    if (a < 0 && b < 0 && c < 0)
        return std::max({a, b, c});
    return 0.0;
}

// Each interface flux is computed once and shared by its two adjacent cells.
inline void residual(const std::vector<double> &u, double a, double nu, double dx, Scheme scheme,
                     std::vector<double> &slope, std::vector<double> &flux,
                     std::vector<double> &rhs) {
    const auto n = u.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double left = u[i] - u[(i + n - 1) % n];
        const double right = u[(i + 1) % n] - u[i];
        slope[i] = scheme == Scheme::mc ? minmod(2 * left, (left + right) / 2, 2 * right) : 0.0;
    }
    for (std::size_t i = 0; i < n; ++i) {
        const auto j = (i + 1) % n;
        const double face = a >= 0 ? u[i] + slope[i] / 2 : u[j] - slope[j] / 2;
        flux[i] = a * face - (nu / dx) * (u[j] - u[i]);
        if (!std::isfinite(flux[i]))
            throw std::overflow_error("non-finite interface flux");
    }
    for (std::size_t i = 0; i < n; ++i) {
        rhs[i] = -(flux[i] - flux[(i + n - 1) % n]) / dx;
        if (!std::isfinite(rhs[i]))
            throw std::overflow_error("non-finite residual");
    }
}

inline Result solve(const std::vector<double> &initial, double a, double nu, double length,
                    double duration, double cfl, Scheme scheme, std::size_t max_steps = 1000000) {
    if (initial.size() < 4)
        throw std::invalid_argument("at least four cells are required");
    if (!std::isfinite(a) || !std::isfinite(nu) || nu < 0 || !std::isfinite(length) ||
        length <= 0 || !std::isfinite(duration) || duration < 0 || !std::isfinite(cfl) ||
        cfl <= 0 || cfl > 1 || max_steps == 0)
        throw std::invalid_argument(
            "require finite a, nu>=0, length>0, duration>=0, 0<cfl<=1, max_steps>0");
    if (scheme != Scheme::upwind && scheme != Scheme::mc)
        throw std::invalid_argument("invalid reconstruction enum");
    for (double value : initial)
        if (!std::isfinite(value))
            throw std::invalid_argument("initial values must be finite");
    const auto n = initial.size();
    const double dx = length / static_cast<double>(n);
    if (!std::isfinite(dx) || dx <= 0)
        throw std::invalid_argument("unrepresentable cell width");
    const double rate = 2 * (std::abs(a) / dx + (nu / dx) / dx);
    if (!std::isfinite(rate))
        throw std::invalid_argument("unrepresentable stability rate");
    if (duration == 0 || rate == 0)
        return {initial, 0, 0.0};
    const double requested = std::max(1.0, std::ceil(duration * (rate / cfl)));
    if (!std::isfinite(requested) || requested >= static_cast<double>(max_steps) + 1.0)
        throw std::invalid_argument(
            "step budget exceeded; reduce duration/resolution or raise max_steps");
    const auto steps = static_cast<std::size_t>(requested);
    const double dt = duration / static_cast<double>(steps);
    if (dt == 0)
        throw std::invalid_argument("time step underflow");
    std::vector<double> u(initial), stage(n), slope(n), flux(n), rhs(n);
    for (std::size_t step = 0; step < steps; ++step) {
        residual(u, a, nu, dx, scheme, slope, flux, rhs);
        for (std::size_t i = 0; i < n; ++i)
            stage[i] = u[i] + dt * rhs[i];
        residual(stage, a, nu, dx, scheme, slope, flux, rhs);
        for (std::size_t i = 0; i < n; ++i)
            stage[i] = 0.75 * u[i] + 0.25 * (stage[i] + dt * rhs[i]);
        residual(stage, a, nu, dx, scheme, slope, flux, rhs);
        for (std::size_t i = 0; i < n; ++i) {
            u[i] = u[i] / 3.0 + (2.0 / 3.0) * (stage[i] + dt * rhs[i]);
            if (!std::isfinite(u[i]))
                throw std::overflow_error("non-finite time step");
        }
    }
    return {std::move(u), steps, dt};
}
} // namespace fluxledger
