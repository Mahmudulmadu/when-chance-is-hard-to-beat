"""
Download a starter subset of the CWRU Bearing Data Center dataset.

Source: https://engineering.case.edu/bearingdatacenter/download-data-file
File-number -> label mapping below was scraped from the Center's own
"12k Drive End Bearing Fault Data" and "Normal Baseline Data" tables
(2026-07-27). Motor load = 0 HP (~1797 RPM) and fault diameter = 0.007"
for all seeded faults, to keep the starter set small and comparable.

Labels:
    Normal   - healthy baseline
    IR007    - inner race fault, 0.007"
    B007     - ball fault, 0.007"
    OR007_6  - outer race fault, 0.007", centered at 6:00 (load zone)
"""

import pathlib
import requests

BASE_URL = "https://engineering.case.edu/sites/default/files/{}.mat"

# (file_number, label) -- all at 0 HP load, 1797 RPM
MANIFEST = [
    (97, "Normal_0"),
    (105, "IR007_0"),
    (118, "B007_0"),
    (130, "OR007_6_0"),
]

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "cwru"


def download():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for file_num, label in MANIFEST:
        url = BASE_URL.format(file_num)
        dest = OUT_DIR / f"{label}_{file_num}.mat"
        if dest.exists():
            print(f"skip (already exists): {dest.name}")
            continue
        print(f"downloading {url} -> {dest.name}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    print(f"\nDone. Files saved in {OUT_DIR}")


if __name__ == "__main__":
    download()
