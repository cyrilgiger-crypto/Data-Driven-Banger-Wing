import numpy as np
from get_aero import get_aero
from aoa_solver import solve_trim_aoa_from_geometry

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
    # Solve trim AoA
    # ------------------------------------------------------------------
    trim = solve_trim_aoa_from_geometry(
        root_chord=root_chord,
        tip_chord=tip_chord,
        sweep=sweep,
        tip_twist=tip_twist,
        A=A,
        velocity=velocity,
        target_cm=0.0,
    )

    # Huge penalty if trim solution fails
    HUGE_PENALTY = 1e9

    # 1. Check for solver convergence and validity
    if (
        trim is None
        or not trim.get("converged", False)
        or "aoa" not in trim
        or not np.isfinite(trim["aoa"])
    ):
        if verbose:
            print("Trim AoA solver failed to converge.")
            print(f"Returning penalty: {HUGE_PENALTY}")
            print("------------------------------------------------")
        return HUGE_PENALTY

    aoa = float(trim["aoa"])
    aoa_deg = np.rad2deg(aoa)

    # 2. NEW: Check if AoA is within the [-7, 7] degree range
    if aoa_deg < 0.0 or aoa_deg > 5.0:
        if verbose:
            print(f"Trim AoA out of range: {aoa_deg:.4g}° (Allowed: 0° to 5°)")
            print(f"Returning penalty: {HUGE_PENALTY}")
            print("------------------------------------------------")
        return HUGE_PENALTY

    # ------------------------------------------------------------------
    # Aero evaluation
    # ------------------------------------------------------------------
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

    # unpack results
    aero_eff = results["aero_efficiency"]
    Cm  = results["Cm"]
    Cma = results["Cma"]
    Clb = results["Clb"]
    Cnb = results["Cnb"]
    L   = results["L"]

    # Objective function weights
    w_Cm   = 0
    w_stab = 20
    w_lift = 20

    # contributions
    contrib_Cm = w_Cm * abs(Cm)**2

    contrib_Cma = w_stab * abs((Cma - Cma_target) / Cma_target)**2
    contrib_Clb = w_stab * abs((Clb - Clb_target) / Clb_target)**2
    contrib_Cnb = w_stab * abs((Cnb - Cnb_target) / Cnb_target)**2

    contrib_stability = (
        contrib_Cma +
        contrib_Clb +
        contrib_Cnb
    )

    contrib_lift = w_lift * abs(L - 9.81 * weight_kg)**2

    # final objective
    obj = (
        -aero_eff
        + contrib_Cm
        + contrib_lift
        + contrib_stability
    )

    if verbose:
        print(
            f"Solved AoA: {aoa:.4g} rad "
            f"({aoa_deg:.4g} deg), "
            f"converged={trim['converged']}, "
            f"iterations={trim['iterations']}, "
            f"evals={trim['evaluations']}"
        )

        print(f"Aero Efficiency: {aero_eff:.4g}")
        print(f"Cm: {Cm:.4g} (contribution: {contrib_Cm:.4g})")
        print(f"Cma: {Cma:.4g} (target: {Cma_target:.4g}, contribution: {contrib_Cma:.4g})")
        print(f"Clb: {Clb:.4g} (target: {Clb_target:.4g}, contribution: {contrib_Clb:.4g})")
        print(f"Cnb: {Cnb:.4g} (target: {Cnb_target:.4g}, contribution: {contrib_Cnb:.4g})")
        print(f"Lift: {L:.4g} N (target: {9.81*weight_kg:.4g} N, contribution: {contrib_lift:.4g})")
        print(f"Objective Value: {obj:.4g}")
        print("------------------------------------------------")

    return obj