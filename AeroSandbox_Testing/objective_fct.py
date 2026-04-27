import numpy as np
from aero_eval_fct import main as aero_eval

def objective(x: np.ndarray, stability_targets: dict, verbose: bool = False, weight_kg: float = 1, velocity: float = 20) -> float:
   
    # unpack design variables
    taper_ratio, aspect_ratio, sweep, aoa, tip_twist, A, c, delta = x
    Cma_target = stability_targets["Cma"]
    Clb_target = stability_targets["Clb"]
    Cnb_target = stability_targets["Cnb"]

    # compute derived parameters
    root_chord = 2*0.8 / (aspect_ratio * (1 + taper_ratio))
    tip_chord = root_chord * taper_ratio

    # call aero evaluation function
    results = aero_eval(
        tip_chord=tip_chord,
        root_chord=root_chord,
        sweep=sweep,
        aoa=aoa,
        tip_twist=tip_twist,
        A=A,
        c=c,
        delta=delta,
        velocity=velocity,
        enable_plot=False,
        verbose=False
    )

    # unpack results
    aero_eff = results["aero_efficiency"]
    Cm =  results["Cm"]
    Cma = results["Cma"]
    Clb = results["Clb"]
    Cnb = results["Cnb"]
    L = results["L"]

    # Objective function weights (TBD: tune these)
    w_Cm = 100.0
    w_stability = 10.0
    w_lift = 1.0

    # Calculate individual contributions
    # Minimize negative efficiency (i.e., maximize efficiency)
    # Require Cm = 0, dCm/da < 0, dCl/db < 0, dCn/db > 0
    contrib_efficiency = aero_eff
    contrib_Cm = w_Cm * abs(Cm)**2
    contrib_stability = w_stability * (abs(Cma - Cma_target)**2 + abs(Clb - Clb_target)**2 + abs(Cnb - Cnb_target)**2)
    contrib_lift = w_lift * abs(L - 9.81*weight_kg)**2

    # Combine contributions into a single objective function
    obj = -contrib_efficiency + contrib_Cm + contrib_lift + contrib_stability
    
    if verbose:
        print(f"Aero Efficiency: {aero_eff:.4g} (contribution: {contrib_efficiency:.4g})")
        print(f"Cm: {Cm:.4g} (contribution: {contrib_Cm:.4g})")
        print(f"Cma: {Cma:.4g} (target: {Cma_target:.4g}, contribution: {contrib_stability:.4g})")
        print(f"Clb: {Clb:.4g} (target: {Clb_target:.4g}, contribution: {contrib_stability:.4g})")
        print(f"Cnb: {Cnb:.4g} (target: {Cnb_target:.4g}, contribution: {contrib_stability:.4g})")
        print(f"Objective Value: {obj:.4g}")
        print("------------------------------------------------")
    
    return obj

if __name__ == "__main__":

    # test the objective function with a sample input
    input = np.array([
        0.485563,                   # taper ratio
        4.230328,                  # aspect ratio
        0.453217,                   # sweep [rad]
        0.274881,                   # root_twist [rad]
        0.366791,                  # tip_twist [rad]
        0.816124,                   # A (seagull dihedral shape)
        0.526879,                   # c (seagull dihedral shape)
        -0.032036,                   # delta [rad]
    ])

    targets = {
        "Cma": -0.2,   # desired static margin (negative for stability)
        "Clb": -0.1,   # desired dutch-roll mitigation (negative for stability)
        "Cnb": 0.02    # desired directional stability (positive for stability)
    }

    res = objective(input, targets, verbose=True)