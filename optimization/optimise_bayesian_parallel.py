# optimise_bayesian_parallel.py

import numpy as np
from skopt import Optimizer
from joblib import Parallel, delayed

# Re-using your existing logic
from objective_fct import objective
from aoa_solver import solve_trim_aoa_from_geometry
from get_aero import get_aero

# ============================================================
# GLOBAL SETTINGS (Same as original)
# ============================================================

SPAN = 0.8
WEIGHT_KG = 1.0

stability_targets = {
    "Cma": -0.2,
    "Clb": -0.1,
    "Cnb":  0.02,
}

bounds = [
    (0.1, 1.0),     # taper_ratio
    (4.0, 15.0),    # aspect_ratio
    (-0.5, 0.5),    # sweep [rad]
    (-0.4, 0.2),    # tip_twist [rad]
    (0.0, 1.0),     # A
]

# ============================================================
# HELPER FUNCTIONS (Identical to original)
# ============================================================

def unpack_design(x):
    taper_ratio, aspect_ratio, sweep, tip_twist, A = x
    root_chord = (2 * SPAN / (aspect_ratio * (1 + taper_ratio)))
    tip_chord = root_chord * taper_ratio
    return {
        "taper_ratio": taper_ratio, "aspect_ratio": aspect_ratio,
        "sweep": sweep, "tip_twist": tip_twist, "A": A,
        "root_chord": root_chord, "tip_chord": tip_chord,
    }

def evaluate_design(x, verbose=False, enable_plot=False):
    geom = unpack_design(x)
    trim = solve_trim_aoa_from_geometry(
        root_chord=geom["root_chord"], tip_chord=geom["tip_chord"],
        sweep=geom["sweep"], tip_twist=geom["tip_twist"],
        A=geom["A"], target_cm=0.0,
    )

    if trim is None or not trim.get("converged", False) or "aoa" not in trim:
        raise RuntimeError("Trim solver failed.")

    aoa = float(trim["aoa"])
    aero = get_aero(
        span=SPAN, root_chord=geom["root_chord"], tip_chord=geom["tip_chord"],
        sweep=geom["sweep"], aoa=aoa, tip_twist=geom["tip_twist"],
        A=geom["A"], enable_plot=enable_plot, verbose=False,
    )

    if verbose:
        print(f"\n--- DESIGN EVALUATION ---")
        print(f"Taper: {geom['taper_ratio']:.4f}, AR: {geom['aspect_ratio']:.4f}")
        print(f"Trim AoA: {np.rad2deg(aoa):.4f} deg, Eff: {aero['aero_efficiency']:.4f}")
        print(f"Cma: {aero['Cma']:.4f}, Clb: {aero['Clb']:.4f}, Cnb: {aero['Cnb']:.4f}")

    return geom, trim, aero

def fitness_wrapper(x):
    HUGE_PENALTY = 1e9
    try:
        obj = objective(
            x=x, stability_targets=stability_targets,
            verbose=False, weight_kg=WEIGHT_KG, enable_plot=False,
        )
        return obj if np.isfinite(obj) else HUGE_PENALTY
    except Exception:
        return HUGE_PENALTY

# ============================================================
# MAIN BAYESIAN OPTIMISATION
# ============================================================

if __name__ == "__main__":
    
    # Settings for BO
    N_ITERATIONS = 1      # Number of batches
    BATCH_SIZE = 4         # Number of parallel workers/evaluations per batch
    N_INITIAL_POINTS = 10  # Random points to seed the GP model

    print(f"\nStarting Bayesian Optimisation ({N_ITERATIONS * BATCH_SIZE} total evals)...")
    print("=" * 60)

    # Initialize the Optimizer
    # base_estimator="GP" uses Gaussian Processes
    # acq_func="EI" stands for Expected Improvement
    opt = Optimizer(
        dimensions=bounds,
        base_estimator="GP",
        acq_func="EI", 
        n_initial_points=N_INITIAL_POINTS,
        random_state=42
    )

    for i in range(N_ITERATIONS):
        # 1. ASK: Get a batch of points to evaluate
        x_next = opt.ask(n_points=BATCH_SIZE)

        # 2. EVALUATE: Run evaluations in parallel
        # This replaces the 'workers' parameter in differential_evolution
        y_next = Parallel(n_jobs=BATCH_SIZE)(
            delayed(fitness_wrapper)(x) for x in x_next
        )

        # 3. TELL: Feed the results back into the optimizer to update the surrogate model
        result = opt.tell(x_next, y_next)

        print(f"Iteration {i+1}/{N_ITERATIONS} | Best so far: {result.fun:.6f}")

# ========================================================
# FINAL RESULTS
# ========================================================

import pandas as pd

print("\n" + "=" * 60)
print("OPTIMISATION COMPLETE")
print("=" * 60)

# result.x is the best point found
best_x = result.x
best_fun = result.fun

print(f"\nObjective Value : {best_fun:.6f}")

# Re-evaluate the best design
geom, trim, aero = evaluate_design(best_x, verbose=False)

# --------------------------------------------------------
# Extract velocity
# --------------------------------------------------------

velocity = None

# Try trim dictionary first
if "velocity" in trim:
    velocity = trim["velocity"]

# Fallbacks if stored under another name
elif "V" in trim:
    velocity = trim["V"]

elif "velocity" in aero:
    velocity = aero["velocity"]

# --------------------------------------------------------
# Create summary table
# --------------------------------------------------------

summary_data = {
    "Parameter": [
        "Taper Ratio",
        "Aspect Ratio",
        "Sweep [deg]",
        "Tip Twist [deg]",
        "A",
        "Root Chord [m]",
        "Tip Chord [m]",
        "Trim AoA [deg]",
        "Velocity [m/s]",
        "Aerodynamic Efficiency",
        "Cma",
        "Clb",
        "Cnb",
    ],
    "Value": [
        geom["taper_ratio"],
        geom["aspect_ratio"],
        np.rad2deg(geom["sweep"]),
        np.rad2deg(geom["tip_twist"]),
        geom["A"],
        geom["root_chord"],
        geom["tip_chord"],
        np.rad2deg(trim["aoa"]),
        velocity,
        aero["aero_efficiency"],
        aero["Cma"],
        aero["Clb"],
        aero["Cnb"],
    ]
}

df = pd.DataFrame(summary_data)

# Format display
pd.set_option("display.float_format", "{:.6f}".format)

print("\nOPTIMUM DESIGN SUMMARY")
print("=" * 60)
print(df.to_string(index=False))

print("\nOPTIMUM DESIGN VECTOR")
print(f"x = np.array({[round(val, 6) for val in best_x]})")
print("=" * 60)