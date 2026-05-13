import aerosandbox as asb
import aerosandbox.numpy as np

def get_lift(
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
    # "L" returns the lift force in Newtons. 
    # Use "CL" if you specifically want the lift coefficient.
    return float(np.asarray(results["L"]).item())