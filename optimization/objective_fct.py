import numpy as np
from get_aero import get_aero
from aoa_solver import solve_trim_aoa_from_geometry
from velocity_solver import solve_velocity_from_geometry
from joint_solver import solve_aoa_and_velocity
from build_wing import build_wing


def objective(
    x: np.ndarray,
    stability_targets: dict,
    verbose: bool = False,
    weight_kg: float = 1,
    velocity: float = 20,
    enable_plot: bool = False
) -> float:

    # unpack design variables
    taper_ratio, aspect_ratio, sweep, tip_twist, A = x

    Cma_target = stability_targets["Cma"]
    Clb_target = stability_targets["Clb"]
    Cnb_target = stability_targets["Cnb"]

    # compute derived parameters
    root_chord = 2 * 0.8 / (aspect_ratio * (1 + taper_ratio))
    tip_chord = root_chord * taper_ratio

    # ------------------------------------------------------------------
    # Solve Trim AoA and Velocity simultaneously
    # ------------------------------------------------------------------

    # 1. Build the wing once
    airplane = build_wing(
        span=0.8, root_chord=root_chord, tip_chord=tip_chord,
        sweep=sweep, tip_twist=tip_twist, A=A
    )

    # 2. Use the joint solver
    sol = solve_aoa_and_velocity(
        airplane=airplane,
        target_lift=9.81 * weight_kg, # Or 8.0 as per your requirement
        target_cm=0.0
    )

    HARD_FAIL_PENALTY = 5e3
    MAX_RETURNED_OBJECTIVE = 2e4

    aoa = sol.get("aoa", np.nan)
    velocity = sol.get("velocity", np.nan)
    converged = bool(sol.get("converged", False))

    if not np.isfinite(aoa) or not np.isfinite(velocity):
        return HARD_FAIL_PENALTY

    aoa_deg = np.rad2deg(float(aoa))
    aoa_violation_deg = max(0.0, -aoa_deg) + max(0.0, aoa_deg - 5.0)
    penalty_aoa = 200.0 * aoa_violation_deg ** 2
    penalty_convergence = 400.0 if not converged else 0.0

    # ------------------------------------------------------------------
    # Aero evaluation
    # ------------------------------------------------------------------
    try:
        results = get_aero(
            tip_chord=tip_chord,
            root_chord=root_chord,
            sweep=sweep,
            aoa=aoa,
            tip_twist=tip_twist,
            A=A,
            velocity=velocity,
            enable_plot=enable_plot,
            verbose=False
        )
    except Exception:
        return HARD_FAIL_PENALTY

    # unpack results
    aero_eff = results["aero_efficiency"]
    Cm  = results["Cm"]
    Cma = results["Cma"]
    Clb = results["Clb"]
    Cnb = results["Cnb"]
    L   = results["L"]

    # Objective function weights
    w_Cm   = 0
    w_Cma = 300 
    w_Clb = 200
    w_Cnb = 200
    w_lift = 200

    # contributions
    contrib_Cm = w_Cm * abs(Cm)**2

    contrib_Cma = w_Cma * abs((Cma - Cma_target) / Cma_target)**2
    contrib_Clb = w_Clb * abs((Clb - Clb_target) / Clb_target)**2
    contrib_Cnb = w_Cnb * abs((Cnb - Cnb_target) / Cnb_target)**2
    contrib_lift = w_lift * abs((L - 9.81 * weight_kg)/ (9.81 * weight_kg))**2

    contrib_stability = (
        contrib_Cma +
        contrib_Clb +
        contrib_Cnb
    )

    # final objective
    obj_raw = (
        -aero_eff
        + contrib_lift
        + contrib_stability
        + penalty_aoa
        + penalty_convergence
    )
    obj = float(np.clip(obj_raw, -1e4, MAX_RETURNED_OBJECTIVE))

    if verbose:
        print(
            f"Solved AoA: {aoa:.4g} rad "
            f"({aoa_deg:.4g} deg), "
            f"converged={converged}, "
            f"velocity={velocity:.4g} m/s"
        )

        print(f"Aero Efficiency: {aero_eff:.4g}")
        print(f"Cm: {Cm:.4g}")
        print(f"Cma: {Cma:.4g} (target: {Cma_target:.4g}, contribution: {contrib_Cma:.4g})")
        print(f"Clb: {Clb:.4g} (target: {Clb_target:.4g}, contribution: {contrib_Clb:.4g})")
        print(f"Cnb: {Cnb:.4g} (target: {Cnb_target:.4g}, contribution: {contrib_Cnb:.4g})")
        print(f"Lift: {L:.4g} N (target: {9.81*weight_kg:.4g} N, contribution: {contrib_lift:.4g})")
        print(f"AoA penalty: {penalty_aoa:.4g}")
        print(f"Convergence penalty: {penalty_convergence:.4g}")
        print(f"Objective Value: {obj:.4g}")
        print("------------------------------------------------")

    return obj

if __name__ == "__main__":
    # input vector from optimization
    x = np.array([0.516116, 13.111064, 0.5, -0.056322, 1.0]) # WORKING!!!!!!!!

    # default stability targets
    stability_targets = {
    "Cma": -0.2,
    "Clb": -0.1,
    "Cnb":  0.02,
    }
    obj_value = objective(x, stability_targets, verbose=True, enable_plot=True)
    print(f"Objective Value for example design: {obj_value:.4f}")