import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt

plt.show(block=False)
plt.pause(0.1)

# -----------------------
# Wing parameters
# -----------------------
span = 0.8
half_span = span / 2
n_sections = 10

root_chord = 0.2
tip_chord = 0.1

# Sweep angle
sweep_deg = 20
sweep = np.deg2rad(sweep_deg)

# Twist (washout)
root_twist = 2
tip_twist = -1

# Seagull dihedral polynomial parameters
k1 = 0.5
k2 = 0.9
k3 = 2 * np.pi / 180

# Span stations
y_stations = np.linspace(0, half_span, n_sections)

# -----------------------
# Build wing sections
# -----------------------
sections = []

for y in y_stations:

    s = y / half_span

    # Sweep
    x = y * np.tan(sweep)

    # Seagull dihedral
    z = half_span * (k1 * s * (s - k2) * (s - 1) + np.tan(k3) * s)

    # Chord distribution
    chord = root_chord + (tip_chord - root_chord) * s

    # Twist distribution
    twist = root_twist + (tip_twist - root_twist) * s

    sections.append(
        asb.WingXSec(
            xyz_le=[x, y, z],
            chord=chord,
            twist=twist,
            airfoil=asb.Airfoil("MH45"),
        )
    )

# -----------------------
# Create wing
# -----------------------
wing = asb.Wing(
    name="Seagull Wing",
    symmetric=True,
    xsecs=sections,
)

# Reference values
s_ref = wing.area()
b_ref = wing.span()
c_ref = s_ref / b_ref if b_ref != 0 else 1.0

# -----------------------
# Create airplane
# -----------------------
airplane = asb.Airplane(
    wings=[wing],
    s_ref=s_ref,
    b_ref=b_ref,
    c_ref=c_ref,
)

# -----------------------
# Flight condition
# -----------------------
op = asb.OperatingPoint(
    atmosphere=asb.Atmosphere(altitude=0),
    velocity=10,
    alpha=2,
    beta=0
)

# -----------------------
# Run VLM
# -----------------------
analysis = asb.VortexLatticeMethod(
    airplane=airplane,
    op_point=op
)

results = analysis.run()

# -----------------------
# Print results
# -----------------------
print("CL:", float(results["CL"]))
print("CD:", float(results["CD"]))
print("Cm:", float(results["Cm"]))
print("Lift:", float(results["L"]))
print("Drag:", float(results["D"]))

# -----------------------
# Spanwise lift distribution (sum over chordwise panels)
# -----------------------
forces_g = analysis.forces_geometry          # (Npanels, 3) in geometry axes
centers = analysis.vortex_centers            # (Npanels, 3) panel centers

# Convert per-panel forces to wind axes, then take lift = -Z_wind
Fx, Fy, Fz = forces_g[:, 0], forces_g[:, 1], forces_g[:, 2]
Fw = np.stack(
    analysis.op_point.convert_axes(
        Fx, Fy, Fz, from_axes="geometry", to_axes="wind"
    ),
    axis=1,
)
lift_panel = -Fw[:, 2]                        # Lift on each panel [N]

# Bin by spanwise station and sum across chordwise panels
# Use |y| so left/right wings combine into a smooth half-span distribution.
y = np.abs(centers[:, 1])
bin_idx = np.digitize(y, y_stations) - 1

lift_strip = np.array([
    lift_panel[bin_idx == i].sum() for i in range(n_sections-1)
])
y_mid = 0.5 * (y_stations[:-1] + y_stations[1:])
y_mid_tot = np.concatenate([-y_mid[::-1], y_mid])
lift_strip_tot = np.concatenate([lift_strip[::-1], lift_strip])

# analysis.calculate_streamlines(seed_points=None, n_steps=50, length=None)
# analysis.draw(c=None, cmap=None, colorbar_label=None, show=True, show_kwargs=None, draw_streamlines=True, recalculate_streamlines=False, backend='pyvista')

airplane.draw()

plt.figure()
plt.plot(y_mid_tot, lift_strip_tot, "-o", markersize=3)
plt.xlabel("Spanwise Position y (m)")
plt.ylabel("Lift (N)")
plt.title("Spanwise Lift Distribution")
plt.grid(True, alpha=0.3)
plt.show()