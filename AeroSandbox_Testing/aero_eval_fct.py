import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt


def main(span=0.8, 
         root_chord=0.2, tip_chord=0.1, 
         sweep=np.deg2rad(20), 
         aoa=2, tip_twist=-1*np.pi/180, 
         A=0.5, c=0.9, delta=np.deg2rad(5), 
         velocity=20,
         enable_plot = True, verbose=True):
    
    plt.show(block=False)
    plt.pause(0.1)

    # -----------------------
    # Wing parameters
    # -----------------------
    half_span = span / 2
    n_sections = 10
    root_twist = 0.0 # default to no twist at root

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
        z = half_span * (A * s * (s - c) * (s - 1) + np.tan(delta) * s)

        # Chord distribution
        chord = root_chord + (tip_chord - root_chord) * s

        # Twist distribution
        twist = root_twist + (tip_twist - root_twist) * s

        sections.append(
            asb.WingXSec(
                xyz_le=[x, y, z],
                chord=chord,
                twist=np.rad2deg(twist),
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

    # winglet
    # After creating `wing` (and before airplane = asb.Airplane(...))

    tip_xyz = sections[-1].xyz_le  # [x, y, z] at the wing tip
    tip_y = tip_xyz[1]

    winglet_height = 0.05
    winglet_root_chord = tip_chord
    winglet_tip_chord = tip_chord * 0.5

    winglet_cant = 75
    dy = winglet_height / np.tand(winglet_cant)

    winglet = asb.Wing(
        name="Tip Winglet",
        symmetric=True,  # we already put it at +half-span; no mirror
        xsecs=[
            asb.WingXSec(
                xyz_le=[tip_xyz[0], tip_y, tip_xyz[2]],
                chord=winglet_root_chord,
                twist=0,
                airfoil=asb.Airfoil("MH45"),
            ),
            asb.WingXSec(
                xyz_le=[tip_xyz[0] + winglet_tip_chord, tip_y + dy, tip_xyz[2] + winglet_height],
                chord=winglet_tip_chord,
                twist=0,
                airfoil=asb.Airfoil("MH45"),
            ),
        ],
    )


    # -----------------------
    # Create airplane
    # -----------------------
    x_CG_geom = span/6 * (root_chord + 2*tip_chord) / (root_chord + tip_chord) *np.tan(sweep)
    x_CG = x_CG_geom * 0.8 # account for battery / electronics being heavier
    airplane = asb.Airplane(
        wings=[wing, winglet],
        s_ref=s_ref,
        b_ref=b_ref,
        c_ref=c_ref,
        xyz_ref=[x_CG, 0, 0]
    )

    # -----------------------
    # Flight condition
    # -----------------------
    op = asb.OperatingPoint(
        atmosphere=asb.Atmosphere(altitude=0),
        velocity=velocity,
        alpha=np.rad2deg(aoa),
        beta=0
    )

    # -----------------------

    # Run VLM
    # -----------------------

    analysis = asb.AeroBuildup(airplane=airplane, op_point=op)
    # analysis = asb.VortexLatticeMethod(airplane=airplane, op_point=op)

    results = analysis.run_with_stability_derivatives(
        alpha=True, beta=True
    )

    def scalar(x):
        return float(np.asarray(x).item())

    aero_efficiency = scalar(results["CL"]) / scalar(results["CD"]) if scalar(results["CD"]) != 0 else 0.0

    # -----------------------
    # Print results
    # -----------------------
    # print("CL:", scalar(results["CL"]))
    # print("CD:", scalar(results["CD"]))
    # print("Lift:", scalar(results["L"]))
    # print("Drag:", scalar(results["D"]))
    if verbose:
        print("-" * 30)
        print("Aero Efficiency:", f"{scalar(aero_efficiency):.4g}")
        print("dCm/da:", f"{scalar(results['Cma']):.4g}", " < 0")
        print("dCl/db:", f"{scalar(results['Clb']):.4g}", " < 0")
        print("dCn/db:", f"{scalar(results['Cnb']):.4g}", " > 0")
        print("x_np:", f"{scalar(results['x_np'])*100:.4g}", "cm > x_CG: ", f"{x_CG*100:.4g}"," cm")
        print("-" * 30)
        print("Cm: ", f"{scalar(results['Cm']):.4g}", " = 0")

    # -----------------------
    # Spanwise lift distribution (sum over chordwise panels)
    # -----------------------
    if enable_plot:
        vlm = asb.VortexLatticeMethod(airplane=airplane, op_point=op)
        _ = vlm.run()

        forces_g = vlm.forces_geometry          # (Npanels, 3) in geometry axes
        centers = vlm.vortex_centers            # (Npanels, 3) panel centers

        # Convert per-panel forces to wind axes, then take lift = -Z_wind
        Fx, Fy, Fz = forces_g[:, 0], forces_g[:, 1], forces_g[:, 2]
        Fw = np.stack(
            vlm.op_point.convert_axes(
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

        vlm.calculate_streamlines(seed_points=None, n_steps=50, length=None)
        vlm.draw(c=None, cmap=None, colorbar_label=None, show=True, show_kwargs=None, draw_streamlines=True, recalculate_streamlines=False, backend='pyvista')

        airplane.draw()

        lift_ellit = (max(lift_strip[1:-1]) * np.sqrt(1 - (y_mid_tot / half_span) ** 2))  # Elliptical distribution for comparison

        plt.figure()
        plt.plot(y_mid_tot, lift_strip_tot, "-o", markersize=3)
        plt.plot(y_mid_tot, lift_ellit, "--", label="Elliptical")
        plt.xlabel("Spanwise Position y (m)")
        plt.ylabel("Lift (N)")
        plt.title("Spanwise Lift Distribution")
        plt.grid(True, alpha=0.3)
        plt.show()

    return {
        "aero_efficiency": aero_efficiency,
        "Cm":  scalar(results["Cm"]),
        "Cma": scalar(results["Cma"]),
        "Clb": scalar(results["Clb"]),
        "Cnb": scalar(results["Cnb"]),
        "L": scalar(results["L"]),
    }


if __name__ == "__main__":

    taper_ratio  = 0.415260
    aspect_ratio = 11.495605
    sweep        = -0.321983
    aoa          = 0.166822 
    tip_twist    = -0.124538 
    A            = 0.496059
    c            = 0.726379
    delta        = -0.198346 

    root_chord = 2 * 0.8 / (aspect_ratio * (1 + taper_ratio))
    tip_chord = root_chord * taper_ratio

    results = main(
        enable_plot=True,
        tip_chord=tip_chord,
        root_chord=root_chord,
        sweep=sweep,
        aoa=aoa,
        tip_twist=tip_twist,
        A=A,
        c=c,
        delta=delta,
    )
