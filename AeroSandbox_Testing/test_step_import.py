import os

import aerosandbox as asb

try:
    import cadquery as cq
    from cadquery import importers
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "The `cadquery` library is required for STEP import. Install it with: pip install cadquery"
    ) from e

# Path to a STEP file you want to import
step_path = r"C:\path\to\your\model.step"

if not os.path.isfile(step_path):
    raise FileNotFoundError(f"STEP file not found: {step_path}")

# Import STEP geometry with CadQuery (AeroSandbox uses CadQuery under the hood for CAD I/O)
shape = importers.importStep(step_path)

# Optional: wrap in an Assembly so you can export or inspect
assembly = cq.Assembly()
assembly.add(shape, name="ImportedSTEP")

# Example: export back out to confirm import succeeded
out_path = os.path.splitext(step_path)[0] + "_reexport.step"
assembly.export(out_path)

print("Imported STEP and re-exported to:", out_path)
