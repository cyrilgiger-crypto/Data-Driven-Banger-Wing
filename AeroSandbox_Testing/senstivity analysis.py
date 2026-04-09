# %%
from aero_eval_fct import main as aero_eval
import numpy as np
from scipy.stats import qmc
import random
from typing import Dict, List, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler


# %%
def run_sensitivity_analysis(bounds):
    bounds_array = np.array(bounds)
    samples_norm = qmc.LatinHypercube(len(bounds)).random(100)
    samples_scaled = samples_norm * (bounds_array[:, 1] - bounds_array[:, 0]) + bounds_array[:, 0]

    aero_eff = np.zeros(samples_scaled.shape[0])
    for i, sample in enumerate(samples_scaled):
        # unpack current sample
        taper_ratio, aspect_ratio, sweep, root_twist, tip_twist, A, c, delta = sample
        # compute root and tip chord based on aspect ratio and taper ratio
        root_chord = 2*0.8 / (aspect_ratio * (1 + taper_ratio))
        tip_chord = root_chord * taper_ratio

        # evaluate aero performance
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

        aero_eff[i] = results["aero_efficiency"]
    
    # %% Analyze sensitivity

    # ARD RBF: one length scale per feature
    n_features = samples_norm.shape[1]
    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=np.ones(n_features),
        length_scale_bounds=(1e-3, 1e3)
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,    # standardize y internally
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=5
    )

    gpr.fit(samples_norm, aero_eff)
    length_scales = gpr.kernel_.k2.length_scale
    
    param_names = ["taper_ratio", "aspect_ratio", "sweep", "root_twist", "tip_twist", "A", "c", "delta"]
    print("Length scales:")
    for name, scale in zip(param_names, length_scales):
        print(f"  {name}: {scale:.6f}")



# %%
if __name__ == "__main__":
    BOUNDS = np.array([
    (0.2, 1.0),                         # taper ratio   
    (1.0, 15.0),                        # aspect ratio  
    (np.deg2rad(-45), np.deg2rad(45)),  # sweep [rad]
    (np.deg2rad(-20), np.deg2rad(20)),  # root_twist [rad]
    (np.deg2rad(-20), np.deg2rad(20)),  # tip_twist [rad]
    (-2.0, 2.0),                        # A (seagull dihedral shape)
    (0.1, 1.0),                         # c (seagull dihedral shape)
    (np.deg2rad(-20), np.deg2rad(20)),  # delta [rad]
    ], dtype=float)

    run_sensitivity_analysis(BOUNDS)

# %%
