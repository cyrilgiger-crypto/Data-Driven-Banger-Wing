import aerosandbox as asb
import aerosandbox.numpy as np

span = 10
n_sections = 15  # Higher = smoother curve

# Seagull-like dihedral via a 3rd-order polynomial in normalized spanwise position
# z(y) = half_span * (k1*s + k2*s^2 + k3*s^3), where s = y / half_span
k1 = 0.8
k2 = 0.9
k3 = 2

half_span = span / 2

y_stations = np.linspace(0, half_span, n_sections)

sections = []
for y in y_stations:
    s = y / half_span
    z = half_span * (k1*s*(s-k2)*(s-1) + np.tan(k3*np.pi/180)*s)  # Add a small linear dihedral to ensure the wing isn't perfectly flat
    sections.append(
        asb.WingXSec(
            xyz_le=[0, y, z],
            chord=1.0,
            twist=0.0,
            airfoil=asb.Airfoil("naca0012"),
        )
    )

wing = asb.Wing(
    name="Polynomial Wing",
    symmetric=True,
    xsecs=sections,
)

# Provide explicit reference values to avoid MAC division-by-zero
s_ref = wing.area()
b_ref = wing.span()
c_ref = s_ref / b_ref if b_ref != 0 else 1.0

airplane = asb.Airplane(
    wings=[wing],
    s_ref=s_ref,
    b_ref=b_ref,
    c_ref=c_ref,
)

# VLM at 10 m/s, alpha=2 deg, sea-level density
op = asb.OperatingPoint(
    atmosphere=asb.Atmosphere(altitude=0),
    velocity=10.0,
    alpha=2.0,
    beta=0.0,
)

analysis = asb.VortexLatticeMethod(airplane=airplane, op_point=op)
res = analysis.run()

print("rho", float(op.atmosphere.density()))
print("S_ref", float(s_ref))
print("b_ref", float(b_ref))
print("c_ref", float(c_ref))
print("CL", float(res["CL"]))
print("CD", float(res["CD"]))
print("Cm", float(res["Cm"]))
print("CY", float(res["CY"]))
print("Cl", float(res["Cl"]))
print("Cn", float(res["Cn"]))
print("L", float(res["L"]))
print("D", float(res["D"]))
print("Y", float(res["Y"]))

# Draw with the current AeroSandbox API
airplane.draw()
