import math
import random
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import differential_evolution, basinhopping, dual_annealing

from aero_eval_fct import main as aero_eval


BOUNDS: List[Tuple[float, float]] = [
    (0.2, 0.9),                        # taper ratio
    (1.0, 20.0),                       # aspect ratio
    (np.deg2rad(-70), np.deg2rad(70)),  # sweep_deg [rad]
    (np.deg2rad(-45), np.deg2rad(45)),  # root_twist [rad]
    (np.deg2rad(-45), np.deg2rad(45)),   # tip_twist [rad]
    (0.0, 1.0),                         # A (seagull dihedral shape)
    (0.3, 1.0),                         # c (seagull dihedral shape)
    (np.deg2rad(-45), np.deg2rad(45)), # delta [rad]
]


def objective(x: np.ndarray) -> float:
    taper_ratio, aspect_ratio, sweep, root_twist, tip_twist, A, c, delta = x
    root_chord = 2*0.8 / (aspect_ratio * (1 + taper_ratio))
    tip_chord = root_chord * taper_ratio

    results = aero_eval(
        tip_chord=tip_chord,
        root_chord=root_chord,
        sweep=sweep,
        root_twist=root_twist,
        tip_twist=tip_twist,
        A=A,
        c=c,
        delta=delta,
        enable_plot=False,
    )
    aero_eff = results["aero_efficiency"]
    dCmda = results["Cma"]
    dCldb = results["Clb"]
    dCndb = results["Cnb"]

    # Minimize negative efficiency (i.e., maximize efficiency)
    obj = -float(aero_eff) + 1*max(dCmda,0) + 2*max(0,dCldb) - 2*min(0,dCndb)
    print(f"Objective Value: {obj}")
    print("------------------------------------------------")
    return obj


def run_with_differential_evolution() -> Dict[str, object]:
    """Differential Evolution - robust global optimizer"""
    print("\nRunning Differential Evolution...")
    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        strategy='best1bin',
        maxiter=200,
        popsize=15,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        disp=True,
        workers=1,
        updating='deferred'
    )
    return {"x": result.x, "fun": result.fun, "nit": result.nit, "message": result.message}


def run_with_basin_hopping() -> Dict[str, object]:
    """Basin Hopping - combines global and local optimization"""
    print("\nRunning Basin Hopping...")
    
    # Define local minimizer
    from scipy.optimize import minimize
    
    # Starting point (random within bounds)
    x0 = np.array([random.uniform(b[0], b[1]) for b in BOUNDS])
    
    result = basinhopping(
        objective,
        x0,
        niter=100,
        T=1.0,
        stepsize=0.5,
        minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': BOUNDS},
        seed=42,
        disp=True
    )
    return {"x": result.x, "fun": result.fun, "nit": result.niter}


def run_with_dual_annealing() -> Dict[str, object]:
    """Dual Annealing - original method from your code"""
    print("\nRunning Dual Annealing...")
    result = dual_annealing(
        objective,
        bounds=BOUNDS,
        maxiter=200,
        seed=42,
        no_local_search=False
    )
    return {"x": result.x, "fun": result.fun, "nit": result.nit, "message": result.message}

def run():
    # Choose optimization method here
    # Options: "de", "basin", "annealing", "ga"
    optimization_method = "de"  # Change this to try different methods
    
    try:
        if optimization_method == "de":
            result = run_with_differential_evolution()
            used = "Differential Evolution (scipy)"
        elif optimization_method == "basin":
            result = run_with_basin_hopping()
            used = "Basin Hopping (scipy)"
        elif optimization_method == "annealing":
            result = run_with_dual_annealing()
        else:
            raise ValueError(f"Unknown method: {optimization_method}")
            
    except Exception as exc:
        print(f"Optimization method '{optimization_method}' failed")
        print(f"Reason: {exc}")
        return

    x = result["x"]
    taper_ratio, aspect_ratio ,sweep, root_twist, tip_twist, A, c, delta = x
    best_eff = -float(result["fun"])

    print("\n" + "="*50)
    print(f"Optimizer: {used}")
    print("Best parameters:")
    print(f"  taper_ratio = {taper_ratio:.6f} rad")
    print(f"  aspect_ratio = {aspect_ratio:.6f}")
    print(f"  sweep       = {sweep:.6f} rad ({np.rad2deg(sweep):.2f} deg)")
    print(f"  root_twist  = {root_twist:.6f} rad ({np.rad2deg(root_twist):.2f} deg)")
    print(f"  tip_twist   = {tip_twist:.6f} rad ({np.rad2deg(tip_twist):.2f} deg)")
    print(f"  A           = {A:.6f}")
    print(f"  c           = {c:.6f}")
    print(f"  delta       = {delta:.6f} rad ({np.rad2deg(delta):.2f} deg)")
    print(f"Best objective value: {result['fun']:.6f}")
    print(f"Best aero efficiency: {best_eff:.6f}")
    print("="*50)


if __name__ == "__main__":
    run()