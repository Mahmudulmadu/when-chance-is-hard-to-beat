"""
Download the XJTU-SY rolling element bearing accelerated life test datasets:
15 bearings run from healthy to failure, vibration sampled every minute.

Source page: https://biaowang.tech/xjtu-sy-bearing-datasets/
Mirror used here: Google Drive, via gdown (handles the large-file virus-scan
interstitial). Measured ~3.7 MB/s against MediaFire's ~0.2 MB/s on the same
connection (20 minutes vs. 6 hours for the same 4.4 GB), so this is a speed
choice, not an availability one. MediaFire is kept below as a fallback in case
the Drive quota is exhausted ("too many users have viewed this file"); Dropbox
and Baidu both need an account, and MEGA needs a browser session.

Why XJTU-SY (vs. staying on Paderborn):
Paderborn records each bearing at one health state, so it structurally cannot
measure detection under per-unit calibration, which notebook 08 identified
as the open question and notebook 10 showed is condition-dependent. XJTU-SY
records every unit healthy and then damaged, so each bearing is its own
baseline. It also documents the failed element per bearing (below), which
notebook 11 showed is a hard requirement: leave-one-bearing-out CV on
artificial damage anti-selects attribution features (rank correlation -0.66),
so attribution features can only be validated on real units of known fault type.

Rig: LDK UER204 bearings, 8 balls, ball d = 7.92 mm, pitch D = 34.55 mm,
contact angle 0 deg -> BPFO = 3.083 x fr, BPFI = 4.917 x fr, BSF = 2.067 x fr,
FTF = 0.3854 x fr. (Close to Paderborn's 6203: BPFO 3.071, BPFI 4.929; the
notebook 06-11 envelope/order pipeline transfers with only fr and the band
table changed.)

Two accelerometers, horizontal + vertical, 25.6 kHz, 32768 samples (1.28 s)
per snapshot, one snapshot per minute until failure. CSV per snapshot:
column 1 = horizontal, column 2 = vertical.

Ground truth, from Table 2 of Introduction_to_XJTU-SY_Bearing_Dataset.pdf,
which the archive itself ships (it lands next to the condition folders). Both
columns below were checked against it, and verify() re-checks the file counts
against the extracted tree. The fault element matters as much as the lifetime
here: notebook 11 showed attribution features can only be selected and
validated on real units of known fault type.

Note the class balance: 8 outer race, 3 inner race, 2 cage, 2 compound. Cage
and compound failures are modes Paderborn does not contain at all, so the
race-only OR/IR framing of notebooks 05-11 does not cover this dataset.

  Condition 1: 2100 rpm (35 Hz), 12 kN
    Bearing1_1  123  outer race
    Bearing1_2  161  outer race
    Bearing1_3  158  outer race
    Bearing1_4  122  cage
    Bearing1_5   52  inner race + outer race
  Condition 2: 2250 rpm (37.5 Hz), 11 kN
    Bearing2_1  491  inner race
    Bearing2_2  161  outer race
    Bearing2_3  533  cage
    Bearing2_4   42  outer race
    Bearing2_5  339  outer race
  Condition 3: 2400 rpm (40 Hz), 10 kN
    Bearing3_1 2538  outer race
    Bearing3_2 2496  inner + ball + cage + outer race
    Bearing3_3  371  inner race
    Bearing3_4 1515  inner race
    Bearing3_5  114  outer race

That is 9216 snapshots / ~12 GB unpacked; the download is a 6-part
multi-volume RAR5 archive, ~4.4 GB.

Citation required by the authors:
  Biao Wang, Yaguo Lei, Naipeng Li, Ningbo Li, "A Hybrid Prognostics Approach
  for Estimating Remaining Useful Life of Rolling Element Bearings", IEEE
  Transactions on Reliability, vol. 69, no. 1, pp. 401-412, 2020.

Extraction needs a real RAR tool. The archive is a 6-volume RAR5 set, and
libarchive/bsdtar (which is what download_paderborn.py uses) reads only the
entries stored in volume 1 and then stops with "Too small block encountered"
at 35Hz12kN + part of 37.5Hz11kN (1556 of 9216 files). It does not follow the
volume chain. RARLAB's UnRAR does: `UnRAR t part01.rar` walks all three
conditions and reports "All OK".

So `find_unrar()` below bootstraps one: it downloads RARLAB's UnRAR, which is
shipped as a single-volume self-extracting RAR. Single-volume is the case bsdtar
handles fine, so the tool we have unpacks the tool we need.
Nothing is installed system-wide; UnRAR.exe lands in data/raw/xjtu/_tools/.
"""

import pathlib
import re
import shutil
import subprocess
import sys

import requests

# Google Drive file ids for the 6 archive volumes, from the "Data" folder of
# the author's share (folder id 1_ycmG46PARiykt82ShfnFfyQsaXv3_VK). Hard-coded
# rather than listed at run time: gdown's folder listing crashes on the two
# Chinese-named note files that sit alongside them, and the ids are stable.
DRIVE_IDS = [
    ("XJTU-SY_Bearing_Datasets.part01.rar", "1ATvZuD6j3bPxhyR07Zm-PURmOC4b4uRn"),
    ("XJTU-SY_Bearing_Datasets.part02.rar", "162KvWNIpBGtd7EDWo4yP1j5XsaoNHOYU"),
    ("XJTU-SY_Bearing_Datasets.part03.rar", "1NvzrGW-KOSy48OZmiFxlE3TPV4CKAcw0"),
    ("XJTU-SY_Bearing_Datasets.part04.rar", "1VuQ5-mK11p1S2pTxUZaH_IxOwUlsmN0S"),
    ("XJTU-SY_Bearing_Datasets.part05.rar", "1WH4OU4MLaMGQkbh6DghxPA5Dwvsq8tEf"),
    ("XJTU-SY_Bearing_Datasets.part06.rar", "1wzQzQUx6-J8DuGczT81OkrkTgOUwL-I_"),
]

# Volume sizes in exact bytes, as reported by the MediaFire API for the same
# files (read the raw `size` field, not the rounded MB shown in a listing).
# Used to tell a complete volume from a truncated one, since a half-written
# .rar still looks like a file on disk.
SIZES = {
    "XJTU-SY_Bearing_Datasets.part01.rar": 744488960,
    "XJTU-SY_Bearing_Datasets.part02.rar": 744488960,
    "XJTU-SY_Bearing_Datasets.part03.rar": 744488960,
    "XJTU-SY_Bearing_Datasets.part04.rar": 744488960,
    "XJTU-SY_Bearing_Datasets.part05.rar": 744488960,
    "XJTU-SY_Bearing_Datasets.part06.rar": 722155640,
}

# Fallback host. Same files, ~20x slower; folder key is stable and its public
# API needs no account, so --mediafire re-resolves links at run time.
FOLDER_KEY = "0m0g59yk1pmh5"
API = "https://www.mediafire.com/api/1.5/folder/get_content.php"
UA = {"User-Agent": "Mozilla/5.0"}

# RARLAB's UnRAR, distributed as a single-volume self-extracting RAR.
SFX_URL = "https://www.rarlab.com/rar/unrarw64.exe"

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "xjtu"

# One .csv per minute of life; used as a post-extraction sanity check.
EXPECTED = {
    "35Hz12kN": {"Bearing1_1": 123, "Bearing1_2": 161, "Bearing1_3": 158,
                 "Bearing1_4": 122, "Bearing1_5": 52},
    "37.5Hz11kN": {"Bearing2_1": 491, "Bearing2_2": 161, "Bearing2_3": 533,
                   "Bearing2_4": 42, "Bearing2_5": 339},
    "40Hz10kN": {"Bearing3_1": 2538, "Bearing3_2": 2496, "Bearing3_3": 371,
                 "Bearing3_4": 1515, "Bearing3_5": 114},
}


def find_tar():
    # Same reasoning as download_paderborn.py: prefer System32's bsdtar
    # (libarchive, reads RAR) over Git Bash's tar.exe, which has no RAR support
    # and fails with exit 128 on a perfectly good archive.
    win_tar = pathlib.Path("C:/Windows/system32/tar.exe")
    if win_tar.exists():
        return str(win_tar)
    for candidate in ("bsdtar", "tar"):
        path = shutil.which(candidate)
        if path:
            return path
    sys.exit("No tar/bsdtar with rar support found - install libarchive bsdtar.")


def list_parts(session):
    """Ask MediaFire for the .rar volumes, sorted by part number."""
    resp = session.get(API, params={"r": "utga", "content_type": "files",
                                    "folder_key": FOLDER_KEY,
                                    "response_format": "json"}, timeout=60)
    resp.raise_for_status()
    files = resp.json()["response"]["folder_content"]["files"]
    parts = [(f["filename"], f["quickkey"], int(f["size"]))
             for f in files if f["filename"].lower().endswith(".rar")]
    if not parts:
        sys.exit("No .rar volumes found - the MediaFire share may have moved; "
                 "check https://biaowang.tech/xjtu-sy-bearing-datasets/")
    return sorted(parts)


def resolve(session, quickkey, filename):
    """Scrape the per-file page for its (short-lived) direct download URL."""
    page = session.get(f"https://www.mediafire.com/file/{quickkey}/{filename}/file",
                       timeout=60)
    page.raise_for_status()
    match = re.search(r'href="(https://download[^"]+)"', page.text)
    if not match:
        sys.exit(f"Could not find a direct link for {filename} - MediaFire may "
                 "have changed its page layout.")
    return match.group(1)


def fetch(session, path, url, size):
    """Download to `path`, resuming a partial file if one is there."""
    have = path.stat().st_size if path.exists() else 0
    if have == size:
        print(f"skip (complete): {path.name}")
        return
    headers = dict(UA)
    if have:
        print(f"resuming {path.name} at {have / 1e6:.0f} / {size / 1e6:.0f} MB")
        headers["Range"] = f"bytes={have}-"
    else:
        print(f"downloading {path.name}  ({size / 1e6:.0f} MB)")

    with session.get(url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        # A server that ignores Range answers 200 with the whole file; only
        # append when it honoured the request (206), else start over.
        mode = "ab" if r.status_code == 206 else "wb"
        done = have if mode == "ab" else 0
        with open(path, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {done / 1e6:7.0f} / {size / 1e6:.0f} MB"
                      f"  ({100 * done / size:5.1f} %)", end="", flush=True)
    print()
    if path.stat().st_size != size:
        sys.exit(f"{path.name}: got {path.stat().st_size} bytes, expected {size}")


def verify():
    """Check the unpacked tree against the published per-bearing lifetimes."""
    ok = True
    for condition, bearings in EXPECTED.items():
        for bearing, n_expected in bearings.items():
            found = len(list((OUT_DIR / condition / bearing).glob("*.csv")))
            flag = "" if found == n_expected else f"  <-- expected {n_expected}"
            if found != n_expected:
                ok = False
            print(f"  {condition:<11} {bearing:<12} {found:>5} csv{flag}")
    return ok


def find_unrar():
    """Return a path to UnRAR.exe, fetching RARLAB's copy if none is installed."""
    for candidate in ("UnRAR", "unrar"):
        path = shutil.which(candidate)
        if path:
            return path
    for installed in (r"C:\Program Files\WinRAR\UnRAR.exe",
                      r"C:\Program Files (x86)\WinRAR\UnRAR.exe"):
        if pathlib.Path(installed).exists():
            return installed

    tools = OUT_DIR / "_tools"
    unrar = tools / "UnRAR.exe"
    if unrar.exists():
        return str(unrar)

    tools.mkdir(parents=True, exist_ok=True)
    sfx = tools / "unrarw64.exe"
    print(f"no UnRAR found; fetching {SFX_URL}")
    # requests, not curl: Windows' schannel backend stalls in a TLS
    # renegotiation loop against rarlab.com, and OpenSSL does not.
    resp = requests.get(SFX_URL, headers=UA, timeout=120)
    resp.raise_for_status()
    sfx.write_bytes(resp.content)
    # The SFX is a single-volume RAR, which bsdtar reads without trouble.
    subprocess.run([find_tar(), "-xf", str(sfx), "-C", str(tools)], check=True)
    sfx.unlink()
    if not unrar.exists():
        sys.exit("UnRAR.exe was not in the RARLAB self-extractor.")
    print(f"  -> {unrar}")
    return str(unrar)


def fetch_drive(name, file_id, path):
    """Pull one volume from Google Drive. gdown handles the confirm token."""
    import gdown

    size = SIZES[name]
    if path.exists() and path.stat().st_size == size:
        print(f"skip (complete): {name}")
        return
    print(f"downloading {name}  ({size / 1e6:.0f} MB)")
    gdown.download(id=file_id, output=str(path), quiet=False, resume=True)
    got = path.stat().st_size if path.exists() else 0
    if got != size:
        sys.exit(f"{name}: got {got} bytes, expected {size}. If Drive is "
                 "rate-limiting, retry later or run with --mediafire.")


def download(host="drive"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if (OUT_DIR / "35Hz12kN").is_dir():
        print(f"already extracted in {OUT_DIR}; verifying:")
        verify()
        return

    unrar = find_unrar()
    names = [name for name, _ in DRIVE_IDS]
    print(f"{len(names)} volumes, {sum(SIZES.values()) / 1e9:.2f} GB total "
          f"(host: {host})\n")

    if host == "drive":
        for name, file_id in DRIVE_IDS:
            fetch_drive(name, file_id, OUT_DIR / name)
    else:
        with requests.Session() as session:
            session.headers.update(UA)
            for filename, quickkey, size in list_parts(session):
                fetch(session, OUT_DIR / filename,
                      resolve(session, quickkey, filename), size)

    first = OUT_DIR / names[0]
    print(f"\nextracting {first.name} (6 volumes, ~12 GB unpacked) ...")
    # x = keep paths, -o+ = overwrite, -idq = quiet. The trailing separator is
    # required: UnRAR treats a bare word as a file filter, not a destination.
    subprocess.run([unrar, "x", "-o+", "-idq", str(first), str(OUT_DIR) + "\\"],
                   check=True)

    # UnRAR has no --strip-components, so flatten the archive's own top level
    # into OUT_DIR, matching the data/raw/paderborn/<CODE>/ layout.
    top = OUT_DIR / "XJTU-SY_Bearing_Datasets"
    if top.is_dir():
        for child in top.iterdir():
            target = OUT_DIR / child.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(child), str(target))
        top.rmdir()

    print("\nverifying against published lifetimes:")
    if not verify():
        sys.exit("Extraction incomplete - keeping the .rar volumes for a retry.")

    for name in names:
        (OUT_DIR / name).unlink()
    print(f"\nDone. Files saved in {OUT_DIR}")


if __name__ == "__main__":
    download("mediafire" if "--mediafire" in sys.argv else "drive")
