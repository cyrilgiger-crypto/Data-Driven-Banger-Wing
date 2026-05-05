import numpy as np
from scipy.optimize import differential_evolution
from objective_fct import objective

# 1. Define the stability targets expected by the objective function
stability_targets = {
    "Cma": -0.2,   # desired static margin (negative for stability)
    "Clb": -0.1,   # desired dutch-roll mitigation
    "Cnb": 0.02    # desired directional stability
}

# 2. Define upper and lower bounds for the 8 design variables
# Formatted as (min, max) for each variable. You can adjust these based on physical limits.
# Variables: taper_ratio, aspect_ratio, sweep, aoa, tip_twist, A, c, delta
bounds = [
    (0.1, 1.0),                 # taper_ratio
    (4.0, 15.0),                # aspect_ratio
    (-0.5, 0.5),                # sweep [rad] (~ -28 to 28 deg)
    (0.0, 0.25),                # aoa [rad] (~ 0 to 14 deg)
    (-0.4, 0.2),                # tip_twist [rad] (~ -22 to 11 deg)
    (0.0, 1.0),                 # A (seagull dihedral shape)
    (0.0, 1.0),                 # c (seagull dihedral shape)
    (-0.35, 0.35),              # delta [rad] (~ -20 to 20 deg)
]

def fitness_wrapper(x):
    """
    Wrapper for the objective function. 
    GAs often test extreme geometries that might crash the VLM solver.
    We use a try-except block to heavily penalize failed configurations.
    """
    try:
        # Evaluate the individual design
        return objective(x, stability_targets=stability_targets, verbose=False, enable_plot=False)
    except Exception as e:
        # If AeroSandbox fails to mesh or solve, return a huge penalty
        return 1e6 

if __name__ == "__main__":
    print("Starting Evolutionary Optimization (Differential Evolution)...")
    print("This may take a while depending on your CPU and the population size.")

    # 3. Run the Differential Evolution Algorithm
    # Adjust maxiter and popsize to trade-off between speed and optimality
    result = differential_evolution(
        fitness_wrapper, 
        bounds, 
        strategy='best1bin', 
        maxiter=30,       # Maximum number of generations
        popsize=10,       # Multiplier for population size (pop = popsize * len(bounds) = 80)
        mutation=(0.5, 1.0), 
        recombination=0.7, 
        disp=True,        # Set to True to print progress of each generation
        tol=0.01          # Tolerance for convergence
    )

    # 4. Print the final results
    print("\n" + "="*40)
    print("         OPTIMIZATION COMPLETE")
    print("="*40)
    print(f"Optimization Success : {result.success}")
    print(f"Final Objective Value: {result.fun:.4g}")
    print(f"Total Iterations     : {result.nit}")

    opt_x = result.x
    print("\nOptimal Design Variables Found:")
    print(f"  Taper Ratio  : {opt_x[0]:.4f}")
    print(f"  Aspect Ratio : {opt_x[1]:.4f}")
    print(f"  Sweep        : {opt_x[2]:.4f} rad ({np.degrees(opt_x[2]):.2f}°)")
    print(f"  AoA          : {opt_x[3]:.4f} rad ({np.degrees(opt_x[3]):.2f}°)")
    print(f"  Tip Twist    : {opt_x[4]:.4f} rad ({np.degrees(opt_x[4]):.2f}°)")
    print(f"  A (Seagull)  : {opt_x[5]:.4f}")
    print(f"  c (Seagull)  : {opt_x[6]:.4f}")
    print(f"  Delta        : {opt_x[7]:.4f} rad ({np.degrees(opt_x[7]):.2f}°)")

    # 5. Evaluate and plot the final optimized design
    print("\nEvaluating final optimized design to display breakdown and plot...")
    try:
        final_obj = objective(opt_x, stability_targets, verbose=True, enable_plot=True)
    except Exception as e:
        print(f"Could not plot final design: {e}")