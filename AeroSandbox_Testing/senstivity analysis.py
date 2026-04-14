## run %matplotlib widget in Jupyter for interactive plots

# %%
from aero_eval_fct import main as aero_eval
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import os

##############################################################################################################
# %%
# Define parameter bounds
# Define bounds
BOUNDS = np.array([
(0.1, 1.0),                         # taper ratio
(1.0, 10.0),                        # aspect ratio
(np.deg2rad(-45), np.deg2rad(45)),  # sweep [rad]
(np.deg2rad(-15), np.deg2rad(15)),  # root_twist [rad]
(np.deg2rad(-15), np.deg2rad(15)),  # tip_twist [rad]
(0.0, 3.0),                         # A (seagull dihedral shape)
(0.1, 1.0),                         # c (seagull dihedral shape)
(np.deg2rad(-10), np.deg2rad(20)),  # delta [rad]
], dtype=float)

##############################################################################################################
# Create LHS samples
# %%
# Sample design space (LHS)
BOUNDS_ARRAY = np.array(BOUNDS)
samples_norm = qmc.LatinHypercube(len(BOUNDS_ARRAY)).random(500)
samples_scaled = samples_norm * (BOUNDS_ARRAY[:, 1] - BOUNDS_ARRAY[:, 0]) + BOUNDS_ARRAY[:, 0]

##############################################################################################################
# %%
# Run LHS sample collection
# Evaluate aerodynamic efficiency
n_samples = samples_scaled.shape[0]
aero_eff = np.zeros(n_samples)
Cma = np.zeros(n_samples)
Clb = np.zeros(n_samples)
Cnb = np.zeros(n_samples)

for i, sample in enumerate(samples_scaled):
    taper_ratio, aspect_ratio, sweep, root_twist, tip_twist, A, c, delta = sample

    # compute root and tip chord based on aspect ratio and taper ratio
    root_chord = 2 * 0.8 / (aspect_ratio * (1 + taper_ratio))
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
    Cma[i] = results["Cma"]
    Clb[i] = results["Clb"]
    Cnb[i] = results["Cnb"]

#######################################################################################################
# %%
# Train GPR on Aero Efficiency

n_features = samples_norm.shape[1]
kernel = C(1.0, (1e-3, 1e3)) * RBF(
    length_scale=np.ones(n_features),
    length_scale_bounds=(1e-3, 1e4)
)

gpr_aero = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,    # standardize y internally
    optimizer="fmin_l_bfgs_b",
    n_restarts_optimizer=20
)

gpr_aero.fit(samples_norm, aero_eff)
length_scales = gpr_aero.kernel_.k2.length_scale

param_names = ["taper_ratio", "aspect_ratio", "sweep", "root_twist", "tip_twist", "A", "c", "delta"]
print('-' * 40)
print("Length scales:")
for name, scale in zip(param_names, length_scales):
    print(f"  {name}: {scale:.6f}")

###########################################################################################################
# %%
# Train GPR on Cma 

gpr_Cma = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    optimizer="fmin_l_bfgs_b",
    n_restarts_optimizer=20
)
gpr_Cma.fit(samples_norm, Cma)
length_scales_Cma = gpr_Cma.kernel_.k2.length_scale

print('-' * 40)
print("Length scales (Cma):")
for name, scale in zip(param_names, length_scales_Cma):
    print(f"  {name}: {scale:.6f}")

###########################################################################################################
# %%
# Plotting Aero Efficiency Results
export_toggle = False
# GPR surface projection
# 0: taper_ratio
# 1: aspect_ratio
# 2: sweep
# 3: root_twist
# 4: tip_twist
# 5: A
# 6: c
# 7: delta
param_idx = (3, 4)  # zero-based indices
grid_res = 50

# actual ranges for selected parameters
x_min, x_max = BOUNDS_ARRAY[param_idx[0]]
y_min, y_max = BOUNDS_ARRAY[param_idx[1]]

x_range = np.linspace(x_min, x_max, grid_res)
y_range = np.linspace(y_min, y_max, grid_res)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# build normalized inputs for prediction (GPR was trained on samples_norm)
X_plot = np.full((grid_res**2, n_features), 0.8)
X_plot[:, param_idx[0]] = (X_grid.ravel() - x_min) / (x_max - x_min)
X_plot[:, param_idx[1]] = (Y_grid.ravel() - y_min) / (y_max - y_min)

Z_pred = gpr_aero.predict(X_plot)
Z_grid = Z_pred.reshape(grid_res, grid_res)

# 3D view of GPR surface
fig1 = plt.figure(figsize=(6, 4.5), facecolor="white")
ax = fig1.add_subplot(111, projection="3d")
surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap="jet", edgecolor="none", alpha=0.8)
fig1.colorbar(surf, ax=ax, shrink=0.7, pad=0.13)

# Training data points (actual parameter values)
ax.scatter(samples_scaled[:, param_idx[0]], samples_scaled[:, param_idx[1]], aero_eff,
           c="k", s=10, depthshade=True)

param_labels = {i: name for i, name in enumerate(param_names)}
ax.set_xlabel(param_labels[param_idx[0]])
ax.set_ylabel(param_labels[param_idx[1]])
ax.set_zlabel("Aero Efficiency (C_L / C_D)")

ax.view_init(elev=30, azim=-45)

plt.tight_layout()

# Export figure
if export_toggle:
    plot_dir = "plot"
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"Aero_3D_surface_{param_labels[param_idx[0]]}_{param_labels[param_idx[1]]}.png"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
plt.show()

#### Top-down contour view
fig2, ax2 = plt.subplots(figsize=(6, 4.5), facecolor="white")

# Filled contour + contour lines for a MATLAB-like top-down view
levels = 30
cf = ax2.contourf(X_grid, Y_grid, Z_grid, levels=levels, cmap="jet")
cs = ax2.contour(X_grid, Y_grid, Z_grid, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
fig2.colorbar(cf, ax=ax2, shrink=0.8, pad=0.02, label='Aero Efficiency (C_L / C_D)')

# Training data points (actual parameter values)
ax2.scatter(samples_scaled[:, param_idx[0]], samples_scaled[:, param_idx[1]],
            s=10, c="k", alpha=0.8)

param_labels = {i: name for i, name in enumerate(param_names)}
ax2.set_xlabel(param_labels[param_idx[0]])
ax2.set_ylabel(param_labels[param_idx[1]])
# ax2.set_title("Top-Down GPR Surface")

ax2.set_aspect("auto")
# ax2.grid(True, which="major", alpha=0.3)

plt.tight_layout()

# Export figure
if export_toggle:
    plot_dir = "plot"
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"Aero_2D_surface_{param_labels[param_idx[0]]}_{param_labels[param_idx[1]]}.png"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
plt.show()

###########################################################################################################
# %%
# Plotting Cma Results

export_toggle = False
# GPR surface projection
# 0: taper_ratio
# 1: aspect_ratio
# 2: sweep
# 3: root_twist
# 4: tip_twist
# 5: A
# 6: c
# 7: delta
param_idx = (3, 5)  # zero-based indices
grid_res = 50

# actual ranges for selected parameters
x_min, x_max = BOUNDS_ARRAY[param_idx[0]]
y_min, y_max = BOUNDS_ARRAY[param_idx[1]]

x_range = np.linspace(x_min, x_max, grid_res)
y_range = np.linspace(y_min, y_max, grid_res)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# build normalized inputs for prediction (GPR was trained on samples_norm)
X_plot = np.full((grid_res**2, n_features), 0.5)
X_plot[:, param_idx[0]] = (X_grid.ravel() - x_min) / (x_max - x_min)
X_plot[:, param_idx[1]] = (Y_grid.ravel() - y_min) / (y_max - y_min)

Z_pred = gpr_Cma.predict(X_plot)
Z_grid = Z_pred.reshape(grid_res, grid_res)

# 3D view of GPR surface
fig1 = plt.figure(figsize=(6, 4.5), facecolor="white")
ax = fig1.add_subplot(111, projection="3d")
surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap="jet", edgecolor="none", alpha=0.8)
fig1.colorbar(surf, ax=ax, shrink=0.5, pad=0.13)

# Training data points (actual parameter values)
ax.scatter(samples_scaled[:, param_idx[0]], samples_scaled[:, param_idx[1]], aero_eff,
           c="k", s=10, depthshade=True)

param_labels = {i: name for i, name in enumerate(param_names)}
ax.set_xlabel(param_labels[param_idx[0]])
ax.set_ylabel(param_labels[param_idx[1]])
ax.set_zlabel("Logitudinal Static Stabilty (Cma < 0)")

ax.view_init(elev=30, azim=-45)

plt.tight_layout()

# Export figure
if export_toggle:
    plot_dir = "plot"
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"Cma_3D_surface_{param_labels[param_idx[0]]}_{param_labels[param_idx[1]]}.png"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
plt.show()

#### Top-down contour view
fig2, ax2 = plt.subplots(figsize=(6, 4.5), facecolor="white")

# Filled contour + contour lines for a MATLAB-like top-down view
levels = 30
cf = ax2.contourf(X_grid, Y_grid, Z_grid, levels=levels, cmap="jet")
cs = ax2.contour(X_grid, Y_grid, Z_grid, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
fig2.colorbar(cf, ax=ax2, shrink=0.8, pad=0.02, label='Logitudinal Static Stabilty (Cma < 0)')

# Training data points (actual parameter values)
ax2.scatter(samples_scaled[:, param_idx[0]], samples_scaled[:, param_idx[1]],
            s=10, c="k", alpha=0.8)

param_labels = {i: name for i, name in enumerate(param_names)}
ax2.set_xlabel(param_labels[param_idx[0]])
ax2.set_ylabel(param_labels[param_idx[1]])
# ax2.set_title("Top-Down GPR Surface")

ax2.set_aspect("auto")
# ax2.grid(True, which="major", alpha=0.3)

plt.tight_layout()

# Export figure
if export_toggle:
    plot_dir = "plot"
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"Cma_2D_surface_{param_labels[param_idx[0]]}_{param_labels[param_idx[1]]}.png"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
plt.show()

# %%