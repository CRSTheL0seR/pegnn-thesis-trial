import pandas as pd
import subprocess
from pathlib import Path

ANSYS_EXE = r"C:\Program Files\ANSYS Inc\v241\ansys\bin\winx64\ANSYS241.exe"
WORKDIR = Path(r"E:\My Thesis\ML Model\Neural Network\Script")

df = pd.read_csv(WORKDIR / "parameters.csv")

for _, row in df.iterrows():
    sid = int(row["sample_id"])
    header_path = WORKDIR / f"params_{sid:04d}.mac"

    with open(header_path, "w") as f:
        f.write(f"sample_id = {sid}\n")
        for comp in ["comp1", "comp2", "comp3", "comp4"]:
            for field in ["x1", "y1", "x2", "y2", "power"]:
                f.write(f"{comp}_{field} = {row[f'{comp}_{field}']}\n")
        f.write("/INPUT,run_case,mac\n")

    out_file = WORKDIR / f"out_{sid:04d}.out"

    subprocess.run(
        [ANSYS_EXE, "-b", "-i", str(header_path), "-o", str(out_file)],
        cwd=str(WORKDIR),
        check=True,
    )
    print(f"Completed sample {sid}")

print("Test batch complete.")