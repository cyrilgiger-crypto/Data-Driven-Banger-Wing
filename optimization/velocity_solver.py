from __future__ import annotations
from typing import Dict, Tuple
import aerosandbox as asb
import aerosandbox.numpy as np
from build_wing import build_wing

def get_lift(airplane: asb.Airplane, aoa: float, velocity: float) -> float:
    op = asb.OperatingPoint(
        atmosphere=asb.Atmosphere(altitude=0),
        velocity=velocity,
        alpha=np.rad2deg(aoa),
        beta=0,
    )
    analysis = asb.AeroBuildup(airplane=airplane, op_point=op)
    results = analysis.run()
    return float(np.asarray(results["L"]).item())

def solve_velocity_for_lift(
    airplane: asb.Airplane,
    aoa: float = np.deg2rad(2.0),
    target_lift: float = 7.0,
    tol: float = 1e-2,
    velocity_bounds: Tuple[float, float] = (10.0, 40.0),
    max_iter: int = 40,
) -> Dict[str, float | bool | int]:
    lo, hi = float(velocity_bounds[0]), float(velocity_bounds[1])
    eval_count = 0

    def residual(v: float) -> float:
        nonlocal eval_count
        eval_count += 1
        return get_lift(airplane, aoa, v) - target_lift

    f_lo, f_hi = residual(lo), residual(hi)

    # Simple Bisection
    if np.sign(f_lo) == np.sign(f_hi):
        return {"velocity": lo if abs(f_lo) < abs(f_hi) else hi, "converged": False}

    for it in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if abs(f_mid) <= tol:
            return {"velocity": mid, "converged": True, "iterations": it, "evaluations": eval_count}
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return {"velocity": mid, "converged": False}

def solve_velocity_from_geometry(target_lift: float, aoa: float, **kwargs) -> Dict:
    airplane = build_wing(**kwargs)
    return solve_velocity_for_lift(airplane=airplane, aoa=aoa, target_lift=target_lift)