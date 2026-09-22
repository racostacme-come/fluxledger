"""Command-line experiments; errors return a nonzero exit status."""

import argparse
import json
from pathlib import Path

from .campaign import campaign, write_csv, write_json
from .solver import fourier_average, solve, top_hat_average


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("campaign", help="Run the fixed analytical validation suite")
    audit.add_argument("--output", type=Path, default=Path("out/campaign"))
    run = sub.add_parser("run", help="Evolve a periodic sine or unit pulse")
    run.add_argument("--profile", choices=["sine", "pulse"], default="sine")
    run.add_argument("--cells", type=int, default=128)
    run.add_argument("--velocity", type=float, default=1)
    run.add_argument("--diffusivity", type=float, default=0.001)
    run.add_argument("--duration", type=float, default=0.5)
    run.add_argument("--length", type=float, default=1)
    run.add_argument("--cfl", type=float, default=0.8)
    run.add_argument("--scheme", choices=["upwind", "mc"], default="mc")
    run.add_argument("--output", type=Path, default=Path("out/run"))
    args = parser.parse_args(argv)
    try:
        if args.command == "campaign":
            report = campaign(args.output)
        else:
            if args.profile == "sine":
                initial = fourier_average(args.cells, length=args.length)
            else:
                initial = top_hat_average(
                    args.cells,
                    length=args.length,
                    left=0.2 * args.length,
                    right=0.4 * args.length,
                )
            result = solve(
                initial,
                velocity=args.velocity,
                diffusivity=args.diffusivity,
                length=args.length,
                duration=args.duration,
                cfl=args.cfl,
                scheme=args.scheme,
            )
            args.output.mkdir(parents=True, exist_ok=True)
            rows = [
                {"x": x, "initial": start, "final": end}
                for x, start, end in zip(result.centers, initial, result.values, strict=True)
            ]
            write_csv(args.output / "solution.csv", rows)
            report = {
                "parameters": {
                    key: value
                    for key, value in vars(args).items()
                    if key not in ("output", "command")
                },
                "steps": result.steps,
                "dt": result.dt,
                "initial_mass": result.initial_mass,
                "final_mass": result.final_mass,
                "mass_drift": result.mass_drift,
            }
            write_json(args.output / "ledger.json", report)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, OverflowError, RuntimeError, OSError) as error:
        parser.exit(2, f"fluxledger: {error}\n")


if __name__ == "__main__":
    main()
