# -*- coding: utf-8 -*-
"""Check every reference in paper/paper-a.md against its authoritative record.

Same rule as results/registry.csv: nothing is trusted because it is written
down. Each entry is fetched from Crossref (by DOI, or by title where the
publisher deposits none) or from the arXiv API, and title, first author, year,
volume and pages are compared field by field against what the paper prints.

Six entries are not in Crossref and cannot be checked this way. Each is listed
in KNOWN below with the record it *was* checked against, so that a re-run
distinguishes "verified elsewhere" from "not verified".

    python src/verify_references.py            # exits non-zero on any new problem

Last full pass: 2026-09-22. Result: 40 checked automatically, 6 by the routes
in KNOWN, 0 outstanding.
"""
import json
import pathlib
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "paper-a.md"
UA = {"User-Agent": "paper-a-ref-check/1.0 (mailto:mahmudulmadu@gmail.com)"}

# Entries Crossref does not index, or indexes under a different record.
# The value is how the entry was verified instead.
KNOWN = {
    7: "JMLR deposits no DOIs. Verified against jmlr.org/papers/v11/ojala10a.html: "
       "Markus Ojala, Gemma C. Garriga, JMLR 11(62):1833-1863, 2010. A Crossref title "
       "search returns the earlier ICDM 2009 paper of the same name, which is a different work.",
    10: "Preprint only. The arXiv API refuses this client (HTTP 406); verified against the "
        "citation metadata on arxiv.org/abs/2607.08273: Yang Shaoliang, Wang Jun, Wang Yunsheng, "
        "posted 2026-07-09. No journal version found in Crossref.",
    13: "Preprint only. Verified against arxiv.org/abs/2407.14625: R. K. Rosa, Danilo Braga, "
        "Danilo Silva, posted 2024-07-19. No journal version found in Crossref.",
    15: "Preprint only. Verified against arxiv.org/abs/2402.12715: Wenqian Ye, Luyang Jiang, "
        "Eric Xie, Guangtao Zheng, Yunsheng Ma, Xu Cao et al., posted 2024-02-20, revised 2025. "
        "No journal version found in Crossref.",
    35: "IEEE PHM'12 proceedings, not in Crossref. Verified against the authors' HAL record "
        "hal-00719503: seven authors, fifth is Brigitte Chebel-Morello (not 'Morello'), 2012.",
    37: "DOI 10.36001/ijphm.2010.v1i1.1336 resolves and the author list matches. Crossref "
        "carries the deposit year 2021; the article is IJPHM volume 1, issue 1, 2010.",
}


def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45) as r:
                out = r.read().decode("utf-8", "replace")
            time.sleep(0.4)
            return out
        except Exception as e:
            err = "ERROR {}: {}".format(type(e).__name__, e)
            time.sleep(1.5 * (attempt + 1))
    return err


def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s.replace("&", " and ")).strip()


def title_ok(a, b):
    a, b = norm(a), norm(b)
    if a == b or a in b or b in a:
        return True
    wa, wb = set(a.split()), set(b.split())
    return len(wa & wb) / max(1, len(wa | wb)) >= 0.8


def parse():
    block = PAPER.read_text("utf-8").split("## References", 1)[1].split("## Revision notes")[0]
    refs = {}
    for line in block.splitlines():
        m = re.match(r"^(\d+)\. (.*)$", line.strip())
        if not m:
            continue
        n, rest = int(m.group(1)), m.group(2)
        t = re.search(r"\*([^*]+)\*", rest)
        doi = re.search(r"DOI (10\.[^\s,]+?)\.?(?:\s|$)", rest)
        ax = re.search(r"arXiv:(\d{4}\.\d{4,5})", rest)
        yr = re.search(r"\b(19\d{2}|20\d{2}|26\d{2})\b", rest.split("DOI")[0])
        refs[n] = {"raw": rest, "authors": rest.split("*")[0].strip(" ."),
                   "title": (t.group(1) if t else "").strip(" .*"),
                   "doi": doi.group(1) if doi else None,
                   "arxiv": ax.group(1) if ax else None,
                   "year": yr.group(1) if yr else None}
    return refs


def check_crossref(r, m):
    """Compare one entry against a Crossref record. Returns a list of problems."""
    bad = []
    ct = (m.get("title") or [""])[0]
    fam = [a.get("family", "") for a in (m.get("author") or [])]
    yr = str((m.get("issued", {}).get("date-parts") or [[None]])[0][0])
    if not title_ok(r["title"], ct):
        bad.append("title: paper has {!r}, record has {!r}".format(r["title"][:60], ct[:60]))
    first = r["authors"].split(",")[0].strip()
    if not first:
        bad.append("no authors in the entry; record has " + ", ".join(fam[:5]))
    elif fam and not any(norm(f) == norm(first.split()[-1]) for f in fam[:2]):
        bad.append("first author: paper has {!r}, record has {!r}".format(first, fam[0]))
    if r["year"] and yr not in ("None", "") and r["year"] != yr:
        bad.append("year: paper has {}, record has {}".format(r["year"], yr))
    listed = r["raw"].replace("–", "-")
    for key in ("volume", "page"):
        v = str(m.get(key) or "").replace("–", "-")
        if v and v not in listed:
            bad.append("{}: record has {} which the entry does not print".format(key, v))
    return bad


def main():
    refs = parse()
    nums = sorted(refs)
    if nums != list(range(1, len(nums) + 1)):
        print("FAIL: reference numbering is not contiguous:", nums)
        return 1

    problems, checked, known = {}, 0, []
    for n in nums:
        r = refs[n]
        if n in KNOWN:
            known.append(n)
            continue
        if r["doi"]:
            j = fetch("https://api.crossref.org/works/" + urllib.parse.quote(r["doi"]))
            if j.startswith("ERROR"):
                problems[n] = ["DOI {} does not resolve in Crossref ({})".format(r["doi"], j[:40])]
                continue
            problems.setdefault(n, []).extend(check_crossref(r, json.loads(j)["message"]))
            checked += 1
        elif r["arxiv"]:
            x = fetch("http://export.arxiv.org/api/query?id_list=" + r["arxiv"])
            if x.startswith("ERROR"):
                problems[n] = ["arXiv lookup failed ({}); add an entry to KNOWN if verified "
                               "by hand".format(x[:40])]
                continue
            ns = {"a": "http://www.w3.org/2005/Atom"}
            e = ET.fromstring(x).find("a:entry", ns)
            if e is None:
                problems[n] = ["arXiv id {} not found".format(r["arxiv"])]
                continue
            ct = " ".join((e.findtext("a:title", "", ns) or "").split())
            names = [a.findtext("a:name", "", ns) or "" for a in e.findall("a:author", ns)]
            bad = []
            if not title_ok(r["title"], ct):
                bad.append("title: paper has {!r}, arXiv has {!r}".format(r["title"][:60], ct[:60]))
            first = r["authors"].split(",")[0].strip()
            if first and names and not any(norm(nm.split()[-1]) == norm(first.split()[-1])
                                           for nm in names[:3]):
                bad.append("first author: paper has {!r}, arXiv has {!r}".format(first, names[0]))
            problems.setdefault(n, []).extend(bad)
            checked += 1
        else:
            q = urllib.parse.urlencode({"query.bibliographic": r["title"], "rows": 3})
            j = fetch("https://api.crossref.org/works?" + q)
            if j.startswith("ERROR"):
                problems[n] = ["Crossref search failed " + j[:40]]
                continue
            hit = next((m for m in json.loads(j)["message"]["items"]
                        if title_ok(r["title"], (m.get("title") or [""])[0])), None)
            if hit is None:
                problems[n] = ["no DOI in the entry and no Crossref match by title; "
                               "add an entry to KNOWN if verified another way"]
                continue
            bad = check_crossref(r, hit)
            if hit.get("DOI"):
                bad.append("Crossref has a DOI for this entry that the paper does not print: "
                           + hit["DOI"])
            problems.setdefault(n, []).extend(bad)
            checked += 1

    problems = {n: v for n, v in problems.items() if v}
    print("{} entries · {} checked against Crossref/arXiv · {} verified by the routes in KNOWN {}"
          .format(len(nums), checked, len(known), known))
    for n in sorted(problems):
        print("\n  [{}] {}".format(n, refs[n]["title"][:70]))
        for p in problems[n]:
            print("       - " + p)
    print("\n" + ("PASS: every entry agrees with its record." if not problems
                  else "FAIL: {} entries disagree.".format(len(problems))))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
