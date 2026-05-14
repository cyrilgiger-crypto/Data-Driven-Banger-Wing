from __future__ import annotations

from typing import Dict, Tuple

import aerosandbox as asb
import aerosandbox.numpy as np

from get_Cm import get_Cm


def solve_trim_aoa(
    airplane: asb.Airplane,
    velocity: float = 20,
    target_cm: float = 0.0,
    tol: float = 1e-2,
    aoa_bounds: Tuple[float, float] = (np.deg2rad(-10), np.deg2rad(15)),
    max_iter: int = 40,
) -> Dict[str, float | bool | int]:
    lo, hi = float(aoa_bounds[0]), float(aoa_bounds[1])
    if hi <= lo:
        raise ValueError("`aoa_bounds` must satisfy upper > lower.")

    eval_count = 0

    def residual(alpha: float) -> float:
        nonlocal eval_count
        eval_count += 1
        return get_Cm(airplane=airplane, aoa=alpha, velocity=velocity) - target_cm

    f_lo = residual(lo)
    f_hi = residual(hi)
    best_alpha, best_res = (lo, f_lo) if abs(f_lo) <= abs(f_hi) else (hi, f_hi)
    if abs(best_res) <= tol:
        return {
            "aoa": float(best_alpha),
            "Cm": float(best_res + target_cm),
            "converged": True,
            "iterations": 0,
            "evaluations": eval_count,
        }

    for it in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)

        if abs(f_mid) < abs(best_res):
            best_alpha, best_res = mid, f_mid

        if abs(f_mid) <= tol:
            return {
                "aoa": float(mid),
                "Cm": float(f_mid + target_cm),
                "converged": True,
                "iterations": it,
                "evaluations": eval_count,
            }

        if np.sign(f_lo) != np.sign(f_hi):
            if np.sign(f_mid) == np.sign(f_lo):
                lo, f_lo = mid, f_mid
            else:
                hi, f_hi = mid, f_mid
        else:
            # No sign change in [lo, hi]: keep shrinking from the worse endpoint.
            if abs(f_lo) >= abs(f_hi):
                lo, f_lo = mid, f_mid
            else:
                hi, f_hi = mid, f_mid

    return {
        "aoa": float(best_alpha),
        "Cm": float(best_res + target_cm),
        "converged": False,
        "iterations": max_iter,
        "evaluations": eval_count,
    }
