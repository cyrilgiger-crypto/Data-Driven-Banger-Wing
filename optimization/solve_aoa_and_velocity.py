from __future__ import annotations
from typing import Dict, Tuple
import aerosandbox as asb
import aerosandbox.numpy as np

# Importing existing functions from provided files
from aoa_solver import solve_trim_aoa
from get_lift import get_lift
from build_wing import build_wing

def solve_aoa_and_velocity(
    airplane: asb.Airplane,
    target_lift: float = 8.0,
    target_cm: float = 0.0,
    aoa_bounds: Tuple[float, float] = (np.deg2rad(0), np.deg2rad(5)),
    velocity_bounds: Tuple[float, float] = (19.0, 25.0),
    tol_cm: float = 0.01,
    tol_lift: float = 0.1,
    max_iter: int = 20,
) -> Dict[str, float | bool | int]:
    """
    Solves for the AoA and Velocity such that the airplane is trimmed (Cm ~ 0)
    and produces the target lift.
    """
    v_lo, v_hi = float(velocity_bounds[0]), float(velocity_bounds[1])
    
    def get_trim_lift_at_v(v: float) -> Tuple[float, float, bool]:
        """Helper to find trim AoA at a specific velocity and return the resulting lift."""
        trim_res = solve_trim_aoa(
            airplane=airplane,
            velocity=v,
            target_cm=target_cm,
            tol=tol_cm,
            aoa_bounds=aoa_bounds,
            max_iter=30
        )
        if not trim_res["converged"]:
            # If not converged, use the best AoA found
            alpha_trim = trim_res["aoa"]
        else:
            alpha_trim = trim_res["aoa"]
            
        lift = get_lift(airplane, alpha_trim, v)
        return alpha_trim, lift, trim_res["converged"]

    # Evaluate bounds
    alpha_lo, lift_lo, conv_lo = get_trim_lift_at_v(v_lo)
    if abs(lift_lo - target_lift) <= tol_lift and conv_lo:
        return {"aoa_deg": np.rad2deg(alpha_lo), "velocity": v_lo, "lift": lift_lo, "converged": True}

    alpha_hi, lift_hi, conv_hi = get_trim_lift_at_v(v_hi)
    if abs(lift_hi - target_lift) <= tol_lift and conv_hi:
        return {"aoa_deg": np.rad2deg(alpha_hi), "velocity": v_hi, "lift": lift_hi, "converged": True}

    # Bisection on velocity
    # Assumption: Lift increases with velocity at the trimmed state
    if (lift_lo - target_lift) * (lift_hi - target_lift) > 0:
        # Target lift might be outside the velocity bounds at trim
        best_v = v_lo if abs(lift_lo - target_lift) < abs(lift_hi - target_lift) else v_hi
        alpha_best, lift_best, conv_best = get_trim_lift_at_v(best_v)
        return {
            "aoa_deg": np.rad2deg(alpha_best),
            "velocity": best_v,
            "lift": lift_best,
            "converged": False,
            "message": "Target lift not bracketed within velocity bounds."
        }

    for i in range(max_iter):
        v_mid = 0.5 * (v_lo + v_hi)
        alpha_mid, lift_mid, conv_mid = get_trim_lift_at_v(v_mid)
        
        if abs(lift_mid - target_lift) <= tol_lift and conv_mid:
            return {
                "aoa_deg": np.rad2deg(alpha_mid),
                "velocity": v_mid,
                "lift": lift_mid,
                "converged": True,
                "iterations": i
            }
        
        if lift_mid < target_lift:
            v_lo = v_mid
        else:
            v_hi = v_mid

    return {
        "aoa_deg": np.rad2deg(alpha_mid),
        "velocity": v_mid,
        "lift": lift_mid,
        "converged": False,
        "message": "Maximum iterations reached."
    }

def solve_from_geometry(target_lift: float = 8.0, **geometry_kwargs) -> Dict:
    """Wrapper to build wing and solve."""
    airplane = build_wing(**geometry_kwargs)
    return solve_aoa_and_velocity(airplane, target_lift=target_lift)