from __future__ import annotations

from typing import Dict

import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt

from build_wing import build_wing


def _scalar(x) -> float:
    return float(np.asarray(x).item())


def get_aero(
    span: float = 0.8,
    root_chord: float = 0.2,
    tip_chord: float = 0.1,
    sweep: float = np.deg2rad(-45),
    aoa: float = 2,
    tip_twist: float = -1 * np.pi / 180,
    A: float = 0.5,
    c: float = 0.5,
    delta: float = np.deg2rad(5),
    velocity: float = 20,
    enable_plot: bool = False,
    verbose: bool = True,
) -> Dict[str, float]:
    plt.show(block=False)
    plt.pause(0.1)

    airplane, meta = build_wing(
        span=span,
        root_chord=root_chord,
        tip_chord=tip_chord,
        sweep=sweep,
        tip_twist=tip_twist,
        A=A,
        c=c,
        delta=delta,
        return_metadata=True,
    )

    op = asb.OperatingPoint(
        atmosphere=asb.Atmosphere(altitude=0),
        velocity=velocity,
        alpha=np.rad2deg(aoa),
        beta=0,
    )

    analysis = asb.AeroBuildup(airplane=airplane, op_point=op)
    results = analysis.run_with_stability_derivatives(alpha=True, beta=True)

    aero_efficiency = _scalar(results["CL"]) / _scalar(results["CD"]) if _scalar(results["CD"]) != 0 else 0.0

    if verbose:
        print("-" * 30)
        print("Aero Efficiency:", f"{_scalar(aero_efficiency):.4g}")
        print("dCm/da:", f"{_scalar(results['Cma']):.4g}", " < 0")
        print("dCl/db:", f"{_scalar(results['Clb']):.4g}", " < 0")
        print("dCn/db:", f"{_scalar(results['Cnb']):.4g}", " > 0")
        print("x_np:", f"{_scalar(results['x_np']) * 100:.4g}", "cm > x_CG: ", f"{float(meta['x_cg']) * 100:.4g}", " cm")
        print("-" * 30)
        print("Cm: ", f"{_scalar(results['Cm']):.4g}", " = 0")

    if enable_plot:
        y_stations = np.asarray(meta["y_stations"])
        half_span = float(meta["half_span"])
        n_sections = int(meta["n_sections"])

        vlm = asb.VortexLatticeMethod(airplane=airplane, op_point=op)
        _ = vlm.run()

        forces_g = vlm.forces_geometry
        centers = vlm.vortex_centers
        fx, fy, fz = forces_g[:, 0], forces_g[:, 1], forces_g[:, 2]
        fw = np.stack(vlm.op_point.convert_axes(fx, fy, fz, from_axes="geometry", to_axes="wind"), axis=1)
        lift_panel = -fw[:, 2]

        y = np.abs(centers[:, 1])
        bin_idx = np.digitize(y, y_stations) - 1

        lift_strip = np.array([lift_panel[bin_idx == i].sum() for i in range(n_sections - 1)])
        y_mid = 0.5 * (y_stations[:-1] + y_stations[1:])
        y_mid_tot = np.concatenate([-y_mid[::-1], y_mid])
        lift_strip_tot = np.concatenate([lift_strip[::-1], lift_strip])

        vlm.calculate_streamlines(seed_points=None, n_steps=50, length=None)
        vlm.draw(
            c=None,
            cmap=None,
            colorbar_label=None,
            show=True,
            show_kwargs=None,
            draw_streamlines=True,
            recalculate_streamlines=False,
            backend="pyvista",
        )

        airplane.draw()

        lift_ellipt = max(lift_strip[1:-1]) * np.sqrt(1 - (y_mid_tot / half_span) ** 2)

        plt.figure()
        plt.plot(y_mid_tot, lift_strip_tot, "-o", markersize=3)
        plt.plot(y_mid_tot, lift_ellipt, "--", label="Elliptical")
        plt.xlabel("Spanwise Position y (m)")
        plt.ylabel("Lift (N)")
        plt.title("Spanwise Lift Distribution")
        plt.grid(True, alpha=0.3)
        plt.show()

    return {
        "aero_efficiency": aero_efficiency,
        "Cm": _scalar(results["Cm"]),
        "Cma": _scalar(results["Cma"]),
        "Clb": _scalar(results["Clb"]),
        "Cnb": _scalar(results["Cnb"]),
        "L": _scalar(results["L"]),
    }
<<<<<<< HEAD
=======

if __name__ == "__main__":
    span = 0.8
    root_chord = 0.2
    tip_chord = 0.1
    sweep = np.deg2rad(-45)
    aoa = np.deg2rad(2)
    tip_twist = np.deg2rad(-1)
    A = 0.5
    velocity = 20
    enable_plot = True
    verbose = True

    get_aero(
        span=span,
        root_chord=root_chord,
        tip_chord=tip_chord,
        sweep=sweep,
        aoa=aoa,
        tip_twist=tip_twist,
        A=A,
        velocity=velocity,
        enable_plot=enable_plot,
        verbose=verbose,
    )
>>>>>>> c94fde1184ee1c77df08cf86c78275a21a316e19
