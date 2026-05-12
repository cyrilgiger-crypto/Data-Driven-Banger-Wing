# optimise_parallel.py

import numpy as np
from scipy.optimize import differential_evolution

from objective_fct import objective
from aoa_solver import solve_trim_aoa_from_geometry
from get_aero import get_aero


# ============================================================
# GLOBAL SETTINGS
# ============================================================

SPAN = 0.8
VELOCITY = 20
WEIGHT_KG = 1.0

# Stability targets
stability_targets = {
    "Cma": -0.2,
    "Clb": -0.1,
    "Cnb":  0.02,
}

# Design variable bounds
# [ taper_ratio, aspect_ratio, sweep, tip_twist, A ]
bounds = [
    (0.1, 1.0),     # taper_ratio
    (4.0, 15.0),    # aspect_ratio
    (-0.5, 0.5),    # sweep [rad]
    (-0.4, 0.2),    # tip_twist [rad]
    (0.0, 1.0),     # A
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def unpack_design(x):
    """
    Convert optimisation vector into physical geometry.
    """

    taper_ratio, aspect_ratio, sweep, tip_twist, A = x

    root_chord = (
        2 * SPAN /
        (aspect_ratio * (1 + taper_ratio))
    )

    tip_chord = root_chord * taper_ratio

<<<<<<< HEAD
    return {
        "taper_ratio": taper_ratio,
        "aspect_ratio": aspect_ratio,
        "sweep": sweep,
        "tip_twist": tip_twist,
        "A": A,
        "root_chord": root_chord,
        "tip_chord": tip_chord,
    }


def evaluate_design(x, verbose=False, enable_plot=False):
    """
    Full evaluation pipeline:
        1. Build geometry
        2. Solve trim AoA
        3. Evaluate aerodynamics
    """

    geom = unpack_design(x)

    # --------------------------------------------------------
    # Solve trimmed AoA
    # --------------------------------------------------------

    trim = solve_trim_aoa_from_geometry(
        root_chord=geom["root_chord"],
        tip_chord=geom["tip_chord"],
        sweep=geom["sweep"],
        tip_twist=geom["tip_twist"],
        A=geom["A"],
        velocity=VELOCITY,
        target_cm=0.0,
    )

    if (
        trim is None
        or not trim.get("converged", False)
        or "aoa" not in trim
        or not np.isfinite(trim["aoa"])
    ):
        raise RuntimeError("Trim solver failed.")

    aoa = float(trim["aoa"])
    aoa_deg = np.rad2deg(aoa)

    # Optional AoA constraints
    if aoa_deg < 0.0 or aoa_deg > 5.0:
        raise RuntimeError(
            f"Trim AoA outside allowed range: {aoa_deg:.3f} deg"
        )

    # --------------------------------------------------------
    # Aero evaluation
    # --------------------------------------------------------

    aero = get_aero(
        span=SPAN,
        root_chord=geom["root_chord"],
        tip_chord=geom["tip_chord"],
        sweep=geom["sweep"],
        aoa=aoa,
        tip_twist=geom["tip_twist"],
        A=geom["A"],
        velocity=VELOCITY,
        enable_plot=enable_plot,
        verbose=False,
    )

    if verbose:

        print("\n" + "=" * 60)
        print("DESIGN EVALUATION")
        print("=" * 60)

        print(f"Taper Ratio : {geom['taper_ratio']:.4f}")
        print(f"Aspect Ratio: {geom['aspect_ratio']:.4f}")
        print(f"Sweep       : {geom['sweep']:.4f} rad")
        print(f"Tip Twist   : {geom['tip_twist']:.4f} rad")
        print(f"A            : {geom['A']:.4f}")

        print("-" * 60)

        print(f"Root Chord  : {geom['root_chord']:.4f} m")
        print(f"Tip Chord   : {geom['tip_chord']:.4f} m")

        print("-" * 60)

        print(f"Trim AoA    : {aoa:.4f} rad")
        print(f"Trim AoA    : {aoa_deg:.4f} deg")

        print("-" * 60)

        print(f"Aero Eff.   : {aero['aero_efficiency']:.4f}")
        print(f"Cm          : {aero['Cm']:.4f}")
        print(f"Cma         : {aero['Cma']:.4f}")
        print(f"Clb         : {aero['Clb']:.4f}")
        print(f"Cnb         : {aero['Cnb']:.4f}")
        print(f"Lift        : {aero['L']:.4f} N")

        print("=" * 60)

    return geom, trim, aero


# ============================================================
# FITNESS FUNCTION
# ============================================================

def fitness_wrapper(x):
    """
    Differential evolution fitness function.
    """

    HUGE_PENALTY = 1e9

    try:

        obj = objective(
            x=x,
            stability_targets=stability_targets,
            verbose=False,
            weight_kg=WEIGHT_KG,
            velocity=VELOCITY,
            enable_plot=False,
        )

        # Prevent NaN/Inf from poisoning optimiser
        if not np.isfinite(obj):
            return HUGE_PENALTY

        return obj

    except Exception:
        return HUGE_PENALTY


# ============================================================
# MAIN OPTIMISATION
# ============================================================

if __name__ == "__main__":

    print("\nStarting Differential Evolution Optimisation...")
    print("=" * 60)

    result = differential_evolution(
        func=fitness_wrapper,
        bounds=bounds,

        # Evolution settings
        strategy="rand1bin",
        maxiter=100,
        popsize=20,

        mutation=(0.7, 1.9),
        recombination=0.4,

        # Parallelisation
        workers=-1,
        updating="deferred",

        # Output
        disp=True,

        # IMPORTANT:
        # Keep disabled to avoid local-minimum polishing
        polish=False,
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("OPTIMISATION COMPLETE")
    print("=" * 60)

    print(f"Success         : {result.success}")
    print(f"Message         : {result.message}")
    print(f"Objective Value : {result.fun:.6f}")

    opt_x = result.x

    # --------------------------------------------------------
    # Re-evaluate optimum EXACTLY as optimiser saw it
    # --------------------------------------------------------

    geom, trim, aero = evaluate_design(
        opt_x,
        verbose=True,
        enable_plot=False,
    )

    print("\n")
    print("=" * 60)
    print("OPTIMUM DESIGN VECTOR")
    print("=" * 60)

    print(f"x = np.array([")
    print(f"    {geom['taper_ratio']:.6f},")
    print(f"    {geom['aspect_ratio']:.6f},")
    print(f"    {geom['sweep']:.6f},")
    print(f"    {geom['tip_twist']:.6f},")
    print(f"    {geom['A']:.6f},")
    print(f"])")

    print("=" * 60)
=======
    # Final evaluation without triggering interactive plot windows
    print("\nFinal Performance Metrics:")
    objective(opt_x, stability_targets, verbose=True, enable_plot=False)
>>>>>>> 12aa553477b0425191cf4137fe1a8b1d13a2452b
