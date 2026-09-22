"""Compare tracer spreading after one circuit of a periodic ring."""

import numpy as np

from fluxledger import solve, top_hat_average

initial = top_hat_average(200)
for method in ("upwind", "mc"):
    result = solve(initial, duration=1, scheme=method)
    error = np.mean(np.abs(result.values - initial))
    print(f"{method:7s}: L1 error={error:.6f}, mass drift={result.mass_drift:.2e}")
