"""
Download the Paderborn University (PU) Bearing DataCenter bearings used by this
project's artificial->real benchmark (notebooks 02-05).

Source: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter
Archives hosted at: https://groups.uni-paderborn.de/kat/BearingDataCenter/<CODE>.rar

Why Paderborn (vs. CWRU): it records vibration *and* two motor phase currents
(64 kHz), plus speed/torque/load-force/temperature — the closest public match
to our 8-channel sensor list (RESEARCH_NOTES.md section 2).

Benchmark set (~170 MB per bearing, ~4.2 GB total, 25 bearings):
    Healthy               K001 K002 K003 K004 K005 K006
    Artificial, outer race  KA01(EDM) KA03(engraver) KA05(engraver) KA07(drill)
    Artificial, inner race  KI01(EDM) KI03(engraver) KI05(engraver) KI07(engraver)
    Real, outer race       KA04 KA15 KA16 KA22 KA30  (accelerated lifetime test)
    Real, inner race       KI04 KI14 KI16 KI17 KI18 KI21  (accelerated lifetime test)

notebooks/02-04 used the 5-bearing starter subset (K001/KA01/KI01/KA04/KI04) -
one bearing per class x domain cell, so "domain" and "bearing identity" were
confounded. This 25-bearing set gives multiple bearings per cell (several
damage-creation methods for artificial; several real units for real), so a
train/test split can hold out *bearings*, not just recordings, and the
artificial->real gap gets error bars instead of a single number.

The artificial-vs-real distinction matters (the sim-to-real gap):
artificial damage is the physical analogue of our simulated faults, real
damage is the deployment target.

Each .rar unpacks to a folder <CODE>/ with 80 .mat files: 20 repeated
measurements x 4 operating conditions, encoded in the file name, e.g.
N15_M07_F10_K001_1.mat = 1500 rpm, 0.7 Nm torque, 1000 N radial force, run 1.

Extraction uses bsdtar (`C:\\Windows\\system32\\tar.exe` on Windows 11, or any
`tar`/`bsdtar` with libarchive rar support on PATH).
"""

import pathlib
import shutil
import subprocess
import sys

import requests

BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/{}.rar"

# (bearing_code, short description)
MANIFEST = [
    ("K001", "healthy"),
    ("K002", "healthy"),
    ("K003", "healthy"),
    ("K004", "healthy"),
    ("K005", "healthy"),
    ("K006", "healthy"),
    ("KA01", "outer race, artificial (EDM), level 1"),
    ("KA03", "outer race, artificial (engraver), level 2"),
    ("KA05", "outer race, artificial (engraver), level 1"),
    ("KA07", "outer race, artificial (drilling), level 1"),
    ("KI01", "inner race, artificial (EDM), level 1"),
    ("KI03", "inner race, artificial (engraver), level 1"),
    ("KI05", "inner race, artificial (engraver), level 1"),
    ("KI07", "inner race, artificial (engraver), level 2"),
    ("KA04", "outer race, real (lifetime test), level 1"),
    ("KA15", "outer race, real (lifetime test), level 1"),
    ("KA16", "outer race, real (lifetime test), level 2"),
    ("KA22", "outer race, real (lifetime test), level 1"),
    ("KA30", "outer race, real (lifetime test), level 1"),
    ("KI04", "inner race, real (lifetime test), level 1"),
    ("KI14", "inner race, real (lifetime test), level 1"),
    ("KI16", "inner race, real (lifetime test), level 3"),
    ("KI17", "inner race, real (lifetime test), level 1"),
    ("KI18", "inner race, real (lifetime test), level 2"),
    ("KI21", "inner race, real (lifetime test), level 1"),
]

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "paderborn"


def find_tar():
    # Prefer Windows 11's built-in System32 tar.exe (bsdtar/libarchive, reads
    # RAR) over whatever "tar"/"bsdtar" a shell's PATH resolves first - Git
    # Bash ships its own tar.exe (usr/bin/tar.exe) with no RAR support, and
    # if that shadows System32 on PATH, extraction fails with exit code 128
    # even though the archive downloaded fine.
    win_tar = pathlib.Path("C:/Windows/system32/tar.exe")
    if win_tar.exists():
        return str(win_tar)
    for candidate in ("bsdtar", "tar"):
        path = shutil.which(candidate)
        if path:
            return path
    sys.exit("No tar/bsdtar with rar support found - install libarchive bsdtar.")


def download():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tar = find_tar()
    for code, desc in MANIFEST:
        extracted = OUT_DIR / code
        if extracted.is_dir() and any(extracted.glob("*.mat")):
            print(f"skip (already extracted): {code}  [{desc}]")
            continue

        rar_path = OUT_DIR / f"{code}.rar"
        if not rar_path.exists():
            url = BASE_URL.format(code)
            print(f"downloading {url} ...")
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(rar_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)

        print(f"extracting {rar_path.name} ...")
        subprocess.run([tar, "-xf", str(rar_path), "-C", str(OUT_DIR)], check=True)
        rar_path.unlink()  # keep only the extracted .mat files
        n = len(list(extracted.glob("*.mat")))
        print(f"  -> {code}: {n} .mat files  [{desc}]")

    print(f"\nDone. Files saved in {OUT_DIR}")


if __name__ == "__main__":
    download()
