from __future__ import annotations

from typing import Dict, Tuple

import aerosandbox as asb
import aerosandbox.numpy as np


def build_wing(
    span: float = 0.8,
    root_chord: float = 0.2,
    tip_chord: float = 0.1,
    sweep: float = np.deg2rad(-45),
    tip_twist: float = -1 * np.pi / 180,
    A: float = 0.5,
    c: float = 0.5,
    delta: float = np.deg2rad(5),
    return_metadata: bool = False,
) -> asb.Airplane | Tuple[asb.Airplane, Dict[str, float | np.ndarray | int]]:
    half_span = span / 2
    n_sections = 10
    root_twist = 0.0
    y_stations = np.linspace(0, half_span, n_sections)

    sections = []
    for y in y_stations:
        s = y / half_span
        x = y * np.tan(sweep)
        z = half_span * (A * s * (s - c) * (s - 1) + np.tan(delta) * s)
        chord = root_chord + (tip_chord - root_chord) * s
        twist = root_twist + (tip_twist - root_twist) * s

        sections.append(
            asb.WingXSec(
                xyz_le=[x, y, z],
                chord=chord,
                twist=np.rad2deg(twist),
                airfoil=asb.Airfoil("MH45"),
            )
        )

    wing = asb.Wing(
        name="Seagull Wing",
        symmetric=True,
        xsecs=sections,
    )

    s_ref = wing.area()
    b_ref = wing.span()
    c_ref = s_ref / b_ref if b_ref != 0 else 1.0

    tip_xyz = sections[-1].xyz_le
    tip_y = tip_xyz[1]
    winglet_height = 0.05
    winglet_root_chord = tip_chord
    winglet_tip_chord = tip_chord * 0.5
    winglet_cant = 75
    dy = winglet_height / np.tand(winglet_cant)

    winglet = asb.Wing(
        name="Tip Winglet",
        symmetric=True,
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

    sweep_05 = np.arctan(np.tan(sweep) - (root_chord - tip_chord) / span)
    x_cg_geom = span / 6 * (root_chord + 2 * tip_chord) / (root_chord + tip_chord) * np.tan(sweep_05)
    x_cg = x_cg_geom * 0.8 + 0.5 * root_chord

    airplane = asb.Airplane(
        wings=[wing, winglet],
        s_ref=s_ref,
        b_ref=b_ref,
        c_ref=c_ref,
        xyz_ref=[x_cg, 0, 0],
    )

    if not return_metadata:
        return airplane

    metadata: Dict[str, float | np.ndarray | int] = {
        "half_span": half_span,
        "n_sections": n_sections,
        "y_stations": y_stations,
        "x_cg": x_cg,
    }
    return airplane, metadata
