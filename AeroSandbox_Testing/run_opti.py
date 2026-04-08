import math
import random
from typing import Callable, Dict, List, Tuple

import numpy as np

from aero_eval_fct import main as aero_eval


BOUNDS: List[Tuple[float, float]] = [
    (0.2, 0.9),                        # taper ratio
    (np.deg2rad(-45), np.deg2rad(60)),  # sweep_deg [rad]
    (np.deg2rad(-10), np.deg2rad(10)),  # root_twist [rad]
    (np.deg2rad(-10), np.deg2rad(8)),   # tip_twist [rad]
    (0.0, 1.0),                         # A (seagull dihedral shape)
    (0.3, 1.0),                         # c (seagull dihedral shape)
    (np.deg2rad(0.0), np.deg2rad(5.0)), # delta [rad]
]


def clamp(x: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    x_clamped = x.copy()
    for i, (lo, hi) in enumerate(bounds):
        x_clamped[i] = min(max(x_clamped[i], lo), hi)
    return x_clamped


def objective(x: np.ndarray) -> float:
    taper_ratio, sweep, root_twist, tip_twist, A, c, delta = x
    tip_chord = 0.2*taper_ratio
    results = aero_eval(
        tip_chord=tip_chord,
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
    return obj

def run_with_scipy() -> Dict[str, object]:
    from scipy.optimize import dual_annealing

    result = dual_annealing(
        objective,
        bounds=BOUNDS,
        maxiter=200,
        maxfun=200,
        seed=42,
        no_local_search=True,
    )
    return {"x": result.x, "fun": result.fun, "nit": result.nit, "message": result.message}

def run():
    try:
        result = run_with_scipy()
        used = "scipy.optimize.dual_annealing"
    except Exception as exc:
        print("SciPy dual_annealing not available")
        print(f"Reason: {exc}")

    x = result["x"]
    taper_ratio, sweep, root_twist, tip_twist, A, c, delta = x
    best_eff = -float(result["fun"])

    print("\nOptimizer:", used)
    print("Best parameters:")
    print(f"  taper_ratio = {taper_ratio:.4f}")
    print(f"  sweep       = {sweep:.4f}")
    print(f"  root_twist  = {root_twist:.4f}")
    print(f"  tip_twist   = {tip_twist:.4f}")
    print(f"  A           = {A:.4f}")
    print(f"  c           = {c:.4f}")
    print(f"  delta       = {delta:.4f}")
    print(f"Best aero efficiency: {best_eff:.6f}")


if __name__ == "__main__":
    run()
