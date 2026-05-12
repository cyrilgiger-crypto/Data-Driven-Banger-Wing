from __future__ import annotations

import aerosandbox as asb
import aerosandbox.numpy as np


def get_Cm(
    airplane: asb.Airplane,
    aoa: float,
    velocity: float = 20,
) -> float:
    op = asb.OperatingPoint(
        atmosphere=asb.Atmosphere(altitude=0),
        velocity=velocity,
        alpha=np.rad2deg(aoa),
        beta=0,
    )
    analysis = asb.AeroBuildup(airplane=airplane, op_point=op)
    results = analysis.run()
    return float(np.asarray(results["Cm"]).item())
