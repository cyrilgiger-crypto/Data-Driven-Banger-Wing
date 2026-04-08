import os
root = r"C:\Users\cyril\AppData\Local\Programs\Python\Python311\Lib\site-packages\aerosandbox"
needle = "STEP"
results = []
for dirpath, dirnames, filenames in os.walk(root):
    for f in filenames:
        if not f.endswith(".py"):
            continue
        path = os.path.join(dirpath, f)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                txt = fh.read()
            if "step" in txt.lower():
                if "step" in txt.lower():
                    results.append(path)
        except Exception:
            pass
for p in results:
    print(p)
