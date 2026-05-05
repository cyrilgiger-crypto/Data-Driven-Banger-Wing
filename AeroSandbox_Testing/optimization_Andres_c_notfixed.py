import numpy as np
from scipy.optimize import minimize
from objective_fct import objective

STABILITY_TARGETS = {"Cma": -0.2, "Clb": -0.1, "Cnb": 0.02}

BOUNDS = [
    (0.2, 0.9),                        # taper ratio
    (1.0, 20.0),                       # aspect ratio
    (np.deg2rad(-20), np.deg2rad(60)), # sweep [rad]
    (np.deg2rad(1.0), np.deg2rad(5)),  # angle of attack [rad]
    (np.deg2rad(-10), np.deg2rad(10)), # tip_twist [rad]
    (0.0, 1.0),                        # A (seagull dihedral shape)
    (0.3, 1.0),                        # c (seagull dihedral shape)
    (np.deg2rad(-45), np.deg2rad(45)), # delta [rad]
]

iteration_count = 0

def iteration_callback(xk):
    global iteration_count
    iteration_count += 1
    print(f"\n>>> End of iteration number {iteration_count} - Best actual candidate performance :")
    
    objective(xk, STABILITY_TARGETS, True, 1.0, 20, False)

if __name__ == "__main__":
    
    x0 = np.array([
        0.55,       # taper_ratio
        10.0,       # aspect_ratio
        0.0,        # sweep (0 rad)
        0.05,       # aoa (~3 deg)
        -0.05,      # tip_twist (~-3 deg)
        0.5,        # A
        0.5,        # c
        0.0         # delta
    ])

    MAX_ITERATIONS = 50

    options={
        'maxiter': MAX_ITERATIONS, 
        'disp': True,
        'ftol': 1e-9  # Par défaut à 1e-06, baissez-le pour forcer la recherche
    }


    print("Beginning of optimization...")
    
    # Utilisation de 'minimize' au lieu de 'differential_evolution'
    result = minimize(
        fun=objective,
        x0=x0,
        args=(STABILITY_TARGETS, False, 1.0, 20, False),
        method='SLSQP',
        bounds=BOUNDS,
        callback=iteration_callback,
        options=options
    )

    labels = ["taper ratio:", "aspect ratio:", "sweep:", "aoa:", "tip twist:", "A:", "c:", "delta:"]

    # Final results
    print("\n" + "="*50)
    print("END OF OPTIMIZATION")
    print("="*50)
    print(f"Algorithm success : {result.success}")
    print(f"Message : {result.message}")
    # print(f"Final optimized parameters : \n{np.round(x_final_full, 4)}")
    for label, value in zip(labels, result.x):
        print(f"{label:<15} {np.round(value, 4)}")
    print("\nFinal performance :")
    objective(result.x, STABILITY_TARGETS, True, 1.0, 20, True)