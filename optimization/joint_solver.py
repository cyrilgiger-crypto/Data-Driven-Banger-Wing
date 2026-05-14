from __future__ import annotations
import aerosandbox.numpy as np
from aoa_solver import solve_trim_aoa
from get_lift import get_lift
from build_wing import build_wing

def solve_aoa_and_velocity(
    airplane,
    target_lift: float = 8.0,
    target_cm: float = 0.0,
    aoa_bounds: tuple = (0, np.deg2rad(5)), # 0 to 5 degrees
    velocity_bounds: tuple = (15.0, 25.0),
    tol_cm: float = 0.01,
    tol_lift: float = 0.01,
    max_outer_iter: int = 12,
):
    v_lo, v_hi = map(float, velocity_bounds)
    aoa_lo, aoa_hi = map(float, aoa_bounds)
    eval_cache = {}
    last_alpha = 0.5 * (aoa_lo + aoa_hi)
    
    def get_trimmed_lift(v):
        nonlocal last_alpha
        key = round(float(v), 6)
        if key in eval_cache:
            return eval_cache[key]

        local_width = np.deg2rad(2.0)
        lo_local = max(aoa_lo, float(last_alpha - local_width))
        hi_local = min(aoa_hi, float(last_alpha + local_width))
        local_bounds = (lo_local, hi_local) if hi_local > lo_local else aoa_bounds

        # Inner loop: Find AoA that trims the plane at this specific velocity
        res = solve_trim_aoa(
            airplane=airplane, velocity=v, target_cm=target_cm, 
            tol=tol_cm, aoa_bounds=local_bounds, max_iter=18, bracket_samples=11
        )
        if not res["converged"]:
            res = solve_trim_aoa(
                airplane=airplane, velocity=v, target_cm=target_cm,
                tol=tol_cm, aoa_bounds=aoa_bounds, max_iter=24, bracket_samples=21
            )

        # Calculate lift at that trimmed AoA
        alpha = float(res["aoa"])
        lift = get_lift(airplane, alpha, v)
        last_alpha = alpha
        out = (alpha, lift, bool(res["converged"]))
        eval_cache[key] = out
        return out

    # Outer loop: Bisection for Velocity to hit target lift
    for _ in range(max_outer_iter):
        v_mid = (v_lo + v_hi) / 2
        alpha_mid, lift_mid, converged = get_trimmed_lift(v_mid)
        
        if abs(lift_mid - target_lift) < tol_lift and converged:
            return {"aoa": alpha_mid, "velocity": v_mid, "lift": lift_mid, "converged": True}
        
        if lift_mid < target_lift:
            v_lo = v_mid
        else:
            v_hi = v_mid

    return {"aoa": alpha_mid, "velocity": v_mid, "lift": lift_mid, "converged": converged}