"""Deterministic analytical validation and portfolio figures."""

import csv
import json
from pathlib import Path

import numpy as np

from .solver import fourier_average, solve, top_hat_average


def write_csv(path, rows):
    """Write homogeneous records as UTF-8 CSV."""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def campaign(output) -> dict:
    """Run 24 smooth refinements and two pulse experiments; raise on failed gates."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, traces, pulse_rows = [], [], []
    for case, velocity, nu in [
        ("advection", 1.0, 0.0),
        ("mixed", -0.7, 0.01),
        ("heat", 0, 0.02),
    ]:
        for scheme in ("upwind", "mc"):
            previous_error = None
            for cells in (40, 80, 160, 320):
                initial = fourier_average(cells)
                result = solve(
                    initial, velocity=velocity, diffusivity=nu, duration=0.5, scheme=scheme
                )
                exact = fourier_average(cells, time=0.5, velocity=velocity, diffusivity=nu)
                error = float(np.mean(np.abs(result.values - exact)))
                amplitude_ratio = abs(np.fft.rfft(result.values)[1] / np.fft.rfft(initial)[1])
                effective_nu = float(-np.log(amplitude_ratio) / ((2 * np.pi) ** 2 * 0.5))
                rows.append(
                    {
                        "case": case,
                        "scheme": scheme,
                        "cells": cells,
                        "velocity": velocity,
                        "physical_diffusivity": nu,
                        "duration": 0.5,
                        "steps": result.steps,
                        "dt": result.dt,
                        "l1_error": error,
                        "order": ""
                        if previous_error is None
                        else float(np.log2(previous_error / error)),
                        "mass_drift": result.mass_drift,
                        "effective_diffusivity": effective_nu,
                        "excess_diffusivity": effective_nu - nu,
                    }
                )
                previous_error = error
                if case == "mixed" and cells == 80:
                    for x, numerical, analytical in zip(
                        result.centers, result.values, exact, strict=True
                    ):
                        traces.append(
                            {
                                "x": x,
                                "scheme": scheme,
                                "numerical": numerical,
                                "exact": analytical,
                            }
                        )
    initial = top_hat_average(200)
    for scheme in ("upwind", "mc"):
        result = solve(initial, duration=1, scheme=scheme)
        for x, value, exact in zip(result.centers, result.values, initial, strict=True):
            pulse_rows.append({"x": x, "scheme": scheme, "numerical": value, "exact": exact})
    orders = {
        f"{row['case']}/{row['scheme']}": row["order"] for row in rows if row["cells"] == 320
    }
    max_mass = max(abs(row["mass_drift"]) for row in rows)
    minimum = float(min(row["numerical"] for row in pulse_rows))
    maximum = float(max(row["numerical"] for row in pulse_rows))
    gates = {
        "smooth_mass_drift_below_1e-12": max_mass < 1e-12,
        "mc_finest_l1_orders_above_1.7": all(
            orders[f"{case}/mc"] > 1.7 for case in ("advection", "mixed", "heat")
        ),
        "pulse_bounds_within_1e-12": minimum >= -1e-12 and maximum <= 1 + 1e-12,
    }
    report = {
        "schema_version": 1,
        "smooth_cases": len(rows),
        "pulse_cases": 2,
        "finest_l1_orders": orders,
        "max_smooth_mass_drift": max_mass,
        "pulse_minimum": minimum,
        "pulse_maximum": maximum,
        "gates": gates,
        "passed": all(gates.values()),
    }
    write_csv(output / "convergence.csv", rows)
    write_csv(output / "smooth_profiles.csv", traces)
    write_csv(output / "pulse_profiles.csv", pulse_rows)
    write_json(output / "validation.json", report)
    _plot(output, rows, traces, pulse_rows)
    if not report["passed"]:
        raise RuntimeError(f"validation failed; see {output / 'validation.json'}")
    return report


def _plot(output, rows, traces, pulses):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"upwind": "#d16b34", "mc": "#197f8e"}
    with plt.rc_context(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    ):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
        fig.suptitle("FluxLedger | Where did the tracer go?", fontsize=20, fontweight="bold")
        for scheme, color in colors.items():
            subset = [r for r in rows if r["case"] == "advection" and r["scheme"] == scheme]
            axes[0, 0].loglog(
                [r["cells"] for r in subset],
                [r["l1_error"] for r in subset],
                "o-",
                color=color,
                label=scheme.upper(),
            )
            axes[1, 1].loglog(
                [r["cells"] for r in subset],
                [r["excess_diffusivity"] for r in subset],
                "o-",
                color=color,
                label=scheme.upper(),
            )
            for ax, records in [(axes[0, 1], traces), (axes[1, 0], pulses)]:
                selection = [r for r in records if r["scheme"] == scheme]
                ax.plot(
                    [r["x"] for r in selection],
                    [r["numerical"] for r in selection],
                    color=color,
                    label=scheme.upper(),
                )
        for ax, records in [(axes[0, 1], traces), (axes[1, 0], pulses)]:
            selection = [r for r in records if r["scheme"] == "mc"]
            ax.plot(
                [r["x"] for r in selection],
                [r["exact"] for r in selection],
                "--",
                color="#263445",
                label="Exact cell averages",
                linewidth=1.4,
            )
            ax.set(xlabel="Position x / L", ylabel="Tracer cell average")
        axes[0, 0].set(
            title="Smooth transport converges",
            xlabel="Number of cells",
            ylabel="Mean absolute error",
        )
        axes[0, 1].set(title="Physical + numerical diffusion | 80 cells")
        axes[1, 0].set(title="One circuit | no physical diffusion | 200 cells")
        cells = np.array([40, 80, 160, 320])
        axes[1, 1].loglog(
            cells, 0.5 / cells, "--", color="#263445", label="Upwind leading term |a| dx / 2"
        )
        axes[1, 1].set(
            title="Excess diffusion from fundamental amplitude",
            xlabel="Number of cells",
            ylabel="Apparent excess diffusivity",
        )
        for ax in axes.flat:
            ax.grid(alpha=0.18, which="both")
            ax.legend(fontsize=8)
        fig.savefig(
            output / "transport_audit.png", dpi=180, metadata={"Software": "FluxLedger"}
        )
        plt.close(fig)
