import numpy as np
from scipy.optimize import minimize
from objective_fct import objective
from get_aero import get_aero

STABILITY_TARGETS = {"Cma": -0.2, "Clb": -0.1, "Cnb": 0.02}

BOUNDS_7 = [ 
    (0.2, 0.9),                        # taper_ratio
    (1.0, 20.0),                       # aspect_ratio
    (np.deg2rad(15), np.deg2rad(35)), # sweep
    (np.deg2rad(-10), np.deg2rad(10)),  # tip_twist
    (0.0, 1.0)                        # A
]

def objective_wrapper(x_reduced, *args):
    # Reconstruit le vecteur complet (avec c=0.5 fixe)
    x_full = np.array([
        x_reduced[0], # taper_ratio
        x_reduced[1], # aspect_ratio
        x_reduced[2], # sweep
        #x_reduced[3], # aoa
        x_reduced[3], # tip_twist
        x_reduced[4] # A
        #0.5,          # c (fixed)
        #5.0           # delta
    ])
    
    return objective(x_full, *args)


iteration_count = 0

def iteration_callback(xk):
    global iteration_count
    iteration_count += 1
    print(f"\n>>> End of iteration number {iteration_count} - Best actual candidate performance :")
    
    objective_wrapper(xk, STABILITY_TARGETS, True, 1.0, 20, False)

if __name__ == "__main__":
    
    x0 = np.array([
        0.8,       # taper_ratio
        10.0,       # aspect_ratio
        0.5,        # sweep (0 rad)
        #0.05,       # aoa (~3 deg)
        0.05,      # tip_twist (~-3 deg)
        0.7,        # A
    ])

    MAX_ITERATIONS = 100

    options={
        'maxiter': MAX_ITERATIONS,
        'disp': True,
        'ftol': 1e-4,
        'eps': 1.4e-04 # Default value
    }


    print("Beginning of optimization...")

    # Utilisation de 'minimize' au lieu de 'differential_evolution'
    result = minimize(
        fun=objective_wrapper,
        x0=x0,
        args=(STABILITY_TARGETS, False, 1.0, 20, False),
        method='SLSQP',
        bounds=BOUNDS_7,
        callback=iteration_callback,
        options=options
    )

    x_final_full = np.array([
        result.x[0],    # taper ratio
        result.x[1],    # aspect ratio
        result.x[2] ,   # sweep
        #result.x[3],    # aoa
        result.x[3],    # tip twist
        result.x[4],    # A
        0.5,            # c
        5.0             # delta
    ])

    root_chord = 2*0.8 / (result.x[1] * (1 + result.x[0]))
    tip_chord = root_chord * result.x[0]

    labels = ["taper ratio:", "aspect ratio:", "sweep:", "tip twist:", "A:", "c:", "delta:"]

    # Final results
    print("\n" + "="*50)
    print("END OF OPTIMIZATION")
    print("="*50)
    print(f"Algorithm success : {result.success}")
    print(f"Message : {result.message}")
    # print(f"Final optimized parameters : \n{np.round(x_final_full, 4)}")
    for label, value in zip(labels, x_final_full):
        print(f"{label:<15} {np.round(value, 4)}")
    print("\nFinal performance :")
    objective_wrapper(result.x, STABILITY_TARGETS, True, 1.0, 20, True)
    #get_aero(
    #    tip_chord=tip_chord,
    #    root_chord=root_chord,
    #    sweep=result.x[2],
    #    aoa=result.x[3],
    #    tip_twist=result.x[3],
    #    A=result.x[4],
    #    c=0.5,
    #    delta=5.0,
    #    velocity=20,
    #    enable_plot=False,
    #    verbose=True
    #)