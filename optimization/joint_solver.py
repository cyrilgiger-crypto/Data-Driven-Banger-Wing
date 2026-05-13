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
    velocity_bounds: tuple = (19.0, 25.0),
    tol_cm: float = 0.01,
    tol_lift: float = 0.01
):
    v_lo, v_hi = velocity_bounds
    
    def get_trimmed_lift(v):
        # Inner loop: Find AoA that trims the plane at this specific velocity
        res = solve_trim_aoa(
            airplane=airplane, velocity=v, target_cm=target_cm, 
            tol=tol_cm, aoa_bounds=aoa_bounds
        )
        # Calculate lift at that trimmed AoA
        lift = get_lift(airplane, res["aoa"], v)
        return res["aoa"], lift, res["converged"]

    # Outer loop: Bisection for Velocity to hit target lift
    for _ in range(15):
        v_mid = (v_lo + v_hi) / 2
        alpha_mid, lift_mid, converged = get_trimmed_lift(v_mid)
        
        if abs(lift_mid - target_lift) < tol_lift and converged:
            return {"aoa": alpha_mid, "velocity": v_mid, "lift": lift_mid, "converged": True}
        
        if lift_mid < target_lift:
            v_lo = v_mid
        else:
            v_hi = v_mid

    return {"aoa": alpha_mid, "velocity": v_mid, "lift": lift_mid, "converged": converged}