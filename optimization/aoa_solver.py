from __future__ import annotations

from typing import Dict, Tuple

import aerosandbox as asb
import aerosandbox.numpy as np

from build_wing import build_wing
from get_Cm import get_Cm


def solve_trim_aoa(
    airplane: asb.Airplane,
    velocity: float = 20,
    target_cm: float = 0.0,
    tol: float = 1e-2,
    aoa_bounds: Tuple[float, float] = (np.deg2rad(-10), np.deg2rad(15)),
    max_iter: int = 40,
    bracket_samples: int = 31,
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
    if abs(f_lo) <= tol:
        return {"aoa": lo, "Cm": f_lo + target_cm, "converged": True, "iterations": 0, "evaluations": eval_count}

    f_hi = residual(hi)
    if abs(f_hi) <= tol:
        return {"aoa": hi, "Cm": f_hi + target_cm, "converged": True, "iterations": 0, "evaluations": eval_count}

    if np.sign(f_lo) == np.sign(f_hi):
        scan = np.linspace(lo, hi, int(bracket_samples))
        scan_vals = [residual(a) for a in scan]
        for i in range(len(scan) - 1):
            if np.sign(scan_vals[i]) == 0:
                return {
                    "aoa": float(scan[i]),
                    "Cm": float(scan_vals[i] + target_cm),
                    "converged": True,
                    "iterations": 0,
                    "evaluations": eval_count,
                }
            if np.sign(scan_vals[i]) != np.sign(scan_vals[i + 1]):
                lo, hi = float(scan[i]), float(scan[i + 1])
                f_lo = float(scan_vals[i])
                break
        else:
            best_i = int(np.argmin(np.abs(np.array(scan_vals))))
            return {
                "aoa": float(scan[best_i]),
                "Cm": float(scan_vals[best_i] + target_cm),
                "converged": False,
                "iterations": max_iter,
                "evaluations": eval_count,
            }

    for it in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if abs(f_mid) <= tol:
            return {"aoa": float(mid), "Cm": float(f_mid + target_cm), "converged": True, "iterations": it, "evaluations": eval_count}
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid

    cm_final = get_Cm(airplane=airplane, aoa=mid, velocity=velocity)
    eval_count += 1
    return {
        "aoa": float(mid),
        "Cm": float(cm_final),
        "converged": abs(cm_final - target_cm) <= tol,
        "iterations": max_iter,
        "evaluations": eval_count,
    }


def solve_trim_aoa_from_geometry(
    span: float = 0.8,
    root_chord: float = 0.2,
    tip_chord: float = 0.1,
    sweep: float = np.deg2rad(-45),
    tip_twist: float = -1 * np.pi / 180,
    A: float = 0.5,
    c: float = 0.5,
    delta: float = np.deg2rad(5),
    velocity: float = 20,
    target_cm: float = 0.0,
    tol: float = 1e-2,
    aoa_bounds: Tuple[float, float] = (np.deg2rad(-10), np.deg2rad(15)),
    max_iter: int = 40,
    bracket_samples: int = 31,
) -> Dict[str, float | bool | int]:
    airplane = build_wing(
        span=span,
        root_chord=root_chord,
        tip_chord=tip_chord,
        sweep=sweep,
        tip_twist=tip_twist,
        A=A,
        c=c,
        delta=delta,
    )
    return solve_trim_aoa(
        airplane=airplane,
        velocity=velocity,
        target_cm=target_cm,
        tol=tol,
        aoa_bounds=aoa_bounds,
        max_iter=max_iter,
        bracket_samples=bracket_samples,
    )
