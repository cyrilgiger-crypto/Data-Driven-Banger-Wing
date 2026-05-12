import numpy as np
from scipy.optimize import differential_evolution
from objective_fct import objective

# 1. Define stability targets
stability_targets = {
    "Cma": -0.2,   # Static margin
    "Clb": -0.1,   # Dutch-roll stability
    "Cnb": 0.02    # Directional stability
}

# 2. Define bounds for the 8 design variables
bounds = [
    (0.1, 1.0),     # taper_ratio
    (4.0, 15.0),    # aspect_ratio
    (-0.5, 0.5),    # sweep [rad]
    # (0.0, 0.08),  # aoa [rad]
    (-0.4, 0.2),    # tip_twist [rad]
    (0.0, 1.0),     # A
    # (0.0, 1),       # c
    # (0.139, 0.140), # delta [rad]
]

def fitness_wrapper(x):
    try:
        # Heavily penalize failed configurations to keep the GA moving
        return objective(
            x, 
            stability_targets=stability_targets, 
            verbose=False, 
            enable_plot=False
        )
    except Exception:
        return 1e6 

if __name__ == "__main__":
    print("Starting Parallel Evolutionary Optimization (No Polishing)...")
    
    result = differential_evolution(
        fitness_wrapper, 
        bounds, 
        strategy='rand1bin', # Better for escaping stagnation than 'best1bin' 'rand1bin'
        maxiter=200, 
        popsize=20, 
        mutation=(0.7, 1.9), 
        recombination=0.4, 
        disp=True,
        workers=-1,          # Parallel computing enabled
        updating='deferred', # Required for parallel workers
        polish=False         # EXPLICITLY REMOVED polishing step
    )

    # 3. Final Results Display
    print("\n" + "="*40)
    print("         OPTIMIZATION COMPLETE")
    print("="*40)
    print(f"Final Objective Value: {result.fun:.4g}")
    
    opt_x = result.x
    labels = ["Taper", "AR", "Sweep (rad)", "AoA (rad)", "Twist (rad)", "A", "c", "Delta"]
    for label, val in zip(labels, opt_x):
        print(f"{label:12}: {val:.4f}")

    # Final evaluation without triggering interactive plot windows
    print("\nFinal Performance Metrics:")
    objective(opt_x, stability_targets, verbose=True, enable_plot=False)
