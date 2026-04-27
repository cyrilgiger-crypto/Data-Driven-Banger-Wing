from typing import Dict, List, Sequence, Tuple

import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from objective_fct import objective


BOUNDS: List[Tuple[float, float]] = [
    (0.2, 0.9),                        # taper ratio
    (1.0, 20.0),                       # aspect ratio
    (np.deg2rad(-70), np.deg2rad(70)), # sweep [rad]
    (np.deg2rad(-45), np.deg2rad(45)), # root_twist [rad]
    (np.deg2rad(-45), np.deg2rad(45)), # tip_twist [rad]
    (0.0, 1.0),                        # A (seagull dihedral shape)
    (0.3, 1.0),                        # c (seagull dihedral shape)
    (np.deg2rad(-45), np.deg2rad(45)), # delta [rad]
]

PARAM_NAMES: Sequence[str] = (
    "taper_ratio",
    "aspect_ratio",
    "sweep",
    "root_twist",
    "tip_twist",
    "A",
    "c",
    "delta",
)

ANGLE_IDS = {2, 3, 4, 7}

def make_dimensions() -> List[Real]:
    return [Real(lo, hi, name=name) for (lo, hi), name in zip(BOUNDS, PARAM_NAMES)]


def make_rbf_gp(seed: int) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(len(BOUNDS)), length_scale_bounds=(1e-3, 1e3))
        + WhiteKernel(noise_level=1e-5, noise_level_bounds="fixed")
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=seed,
    )


def run_bayesian_optimization(
    max_evals: int | None,
    n_initial_points: int,
    exploration_ratio: float,
    seed: int,
    targets: Dict[str, float],
) -> Dict[str, object]:
    if max_evals is None:
        # gp_minimize needs a finite call budget; keep this large and interrupt when needed.
        max_evals = 10000
        print("No max_evals provided. Using n_calls=10000; stop anytime with Ctrl+C.")

    if max_evals < n_initial_points:
        raise ValueError("max_evals must be >= n_initial_points.")

    def objective_wrapper(x_list: List[float]) -> float:
        x = np.array(x_list, dtype=float)
        return float(objective(x, stability_targets=targets, verbose=False))

    print("Running Bayesian optimization (skopt gp_minimize + RBF kernel).")
    latest_result = None

    def snapshot_callback(res) -> bool:
        nonlocal latest_result
        latest_result = res
        return False

    try:
        result = gp_minimize(
            func=objective_wrapper,
            dimensions=make_dimensions(),
            base_estimator=make_rbf_gp(seed),
            acq_func="EI",
            acq_optimizer="sampling",
            xi=exploration_ratio,
            n_calls=max_evals,
            n_initial_points=n_initial_points,
            initial_point_generator="random",
            random_state=seed,
            verbose=True,
            callback=[snapshot_callback],
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted. Returning best point found so far.")
        result = latest_result

    if result is None or len(result.x_iters) == 0:
        raise RuntimeError("No completed evaluations available to return.")

    return {
        "x_best": np.array(result.x, dtype=float),
        "y_best": float(result.fun),
        "eval_count": len(result.func_vals),
        "raw_result": result,
    }


def print_best_solution(x_best: np.ndarray, y_best: float) -> None:
    print("\n" + "=" * 60)
    print("Bayesian Optimization Result")
    print(f"Best objective value: {y_best:.6f}")
    print("Best design parameters:")
    for i, (name, value) in enumerate(zip(PARAM_NAMES, x_best)):
        if i in ANGLE_IDS:
            print(f"  {name:12s} = {value:.6f} rad ({np.rad2deg(value):.2f} deg)")
        else:
            print(f"  {name:12s} = {value:.6f}")
    print("=" * 60)


def run() -> None:

    # Maximum objective evaluations. If omitted, runs until interrupted (Ctrl+C).
    max_evals = None
    # Number of initial random evaluations before BO guidance.
    n_initial = 50
    # EI exploration parameter (mapped to xi).
    exploration_ratio = 0.3
    # Random seed.
    seed = 42
    # Stability targets for Cma, Clb, Cnb.
    targets = {
        "Cma": -0.2,   # desired static margin (negative for stability)
        "Clb": -0.1,   # desired dutch-roll mitigation (negative for stability)
        "Cnb": 0.02    # desired directional stability (positive for stability)
    }

    # Run optimization
    result = run_bayesian_optimization(
        max_evals=max_evals,
        n_initial_points=n_initial,
        exploration_ratio=exploration_ratio,
        seed=seed,
        targets=targets
    )

    x_best = result["x_best"]
    y_best = float(result["y_best"])
    print_best_solution(x_best, y_best)

    print("\nDetailed objective breakdown at best design:")
    objective(np.array(x_best, dtype=float), stability_targets=targets, verbose=True)


if __name__ == "__main__":
    run()
