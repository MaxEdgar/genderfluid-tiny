#!/usr/bin/env python3
"""Build a multinational name-gender association dataset from official sources.

Sources (all official government statistics):
  America:
    - U.S. Social Security Administration baby names (data/ssa_alldata.txt)
    - U.S. Census Bureau 2020 first names (data/census_2020_firstnames_sex.xlsx)
  Europe:
    - INSEE France "Base prénoms" 1900-2024 (downloaded from data.gouv.fr)

Each source contributes per-name counts by sex. Names are merged across
sources and labeled by statistical association: >=85% of recorded births
as female -> girl-associated, >=85% male -> boy-associated, otherwise
uncertain. Rows are weighted by frequency so common names dominate.

Outputs data/names.jsonl plus 80/10/10 train/validation/test splits that
are partitioned by normalized name (a name never appears in two splits).

Note on Asia/Africa: official government agencies in most Asian and
African countries do not publish per-name statistics broken down by sex
(Taiwan and Israel do, but their portals block automated downloads).
This script is structured so additional sources can be added as
functions that return {name: (girl_count, boy_count)}.
"""

import hashlib
import json
import os
import sys
import urllib.request

from genderfluid.preprocessing import normalize_name

ASSOC_THRESHOLD = 0.85
TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10

INSEE_DATASET_API = "https://www.data.gouv.fr/api/1/datasets/base-prenoms-2024-insee/"
SSA_ZIP_URL = "https://www.ssa.gov/oact/babynames/names.zip"


def _load_ssa(path: str, url: str = None) -> dict:
    """
    Parse SSA national baby names.

    Prefers the official names.zip (all years, names with 5+ occurrences,
    roughly a million unique spellings). Falls back to a local aggregated
    file if the download fails.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    cache_zip = os.path.join(base, "data", "raw", "ssa_names.zip")

    if url:
        try:
            import zipfile
            if not os.path.exists(cache_zip) or os.path.getsize(cache_zip) == 0:
                print("  Downloading SSA national data (names.zip)...")
                req = urllib.request.Request(url, headers={
                    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                    "Chrome/126.0 Safari/537.36"),
                    "Accept": "*/*",
                })
                with urllib.request.urlopen(req, timeout=600) as resp, \
                        open(cache_zip, "wb") as out:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
            counts: dict = {}
            with zipfile.ZipFile(cache_zip) as zf:
                for member in zf.namelist():
                    if not member.startswith("yob") or not member.endswith(".txt"):
                        continue
                    for line in zf.read(member).decode("utf-8", "replace").splitlines():
                        parts = line.split(",")
                        if len(parts) < 3:
                            continue
                        name = normalize_name(parts[0])
                        sex = parts[1].strip().upper()
                        try:
                            count = int(parts[2])
                        except ValueError:
                            continue
                        if not name or sex not in ("F", "M"):
                            continue
                        bucket = counts.setdefault(name, [0, 0])
                        bucket[0 if sex == "F" else 1] += count
            print(f"  SSA names.zip: {len(counts):,} unique names")
            return counts
        except Exception as exc:
            print(f"  WARNING: SSA names.zip failed ({exc}); using local file")

    counts: dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            name = normalize_name(parts[0])
            sex = parts[1].strip().upper()
            try:
                count = int(parts[2])
            except ValueError:
                continue
            if not name or sex not in ("F", "M"):
                continue
            bucket = counts.setdefault(name, [0, 0])
            bucket[0 if sex == "F" else 1] += count
    return counts


def _load_census_xlsx(path: str) -> dict:
    """Parse Census 2020 first names workbook: FIRST NAME, MALE, FEMALE."""
    try:
        import openpyxl
    except ImportError:
        print("WARNING: openpyxl not installed, skipping Census source")
        return {}
    counts: dict = {}
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < 3 or row is None or len(row) < 7:
            continue
        name = normalize_name(str(row[0]))
        if not name:
            continue
        try:
            male = int(row[5] or 0)
            female = int(row[6] or 0)
        except (ValueError, TypeError):
            continue
        if male + female == 0:
            continue
        bucket = counts.setdefault(name, [0, 0])
        bucket[0] += female
        bucket[1] += male
    wb.close()
    return counts


def _load_insee() -> dict:
    """Download the INSEE first-names parquet and aggregate counts by name."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("ERROR: pyarrow is required for the INSEE source. "
              "Install it with: pip install pyarrow")
        sys.exit(1)

    print("Resolving INSEE dataset resource...")
    with urllib.request.urlopen(INSEE_DATASET_API, timeout=60) as resp:
        meta = json.loads(resp.read().decode("utf-8"))
    resources = meta.get("resources", [])
    if not resources:
        print("ERROR: no resources found for INSEE dataset")
        sys.exit(1)
    # Prefer the newest parquet resource (the API returns them sorted).
    url = resources[0]["url"]
    print(f"  Downloading: {url}")

    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "raw", "insee_prenoms.parquet")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=600) as resp, \
                        open(local_path, "wb") as out:
                    total = int(resp.headers.get("Content-Length", 0) or 0)
                    done = 0
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  Downloaded {done/1048576:.1f} MB / {total/1048576:.1f} MB", end="")
                print()
            except Exception as exc:
                print(f"\n  Download attempt {attempt + 1} failed: {exc}")
                continue
            if total and os.path.getsize(local_path) < total:
                print(f"  Incomplete download ({os.path.getsize(local_path)} of {total} bytes), retrying...")
                continue
            break
        if os.path.getsize(local_path) == 0:
            print("ERROR: could not download INSEE dataset")
            sys.exit(1)

    print("  Aggregating records...")
    counts: dict = {}
    pf = pq.ParquetFile(local_path)
    for batch in pf.iter_batches(batch_size=100_000):
        cols = batch.to_pydict()
        sexe = cols.get("sexe") or cols.get("SEXE")
        preusuel = (cols.get("preusuel") or cols.get("prenom")
                    or cols.get("PREUSUEL"))
        nombre = (cols.get("nombre") or cols.get("valeur")
                  or cols.get("NOMBRE"))
        if sexe is None or preusuel is None or nombre is None:
            print("ERROR: unexpected INSEE columns:", list(cols.keys()))
            sys.exit(1)
        for s, p, n in zip(sexe, preusuel, nombre):
            name = normalize_name(str(p))
            if not name or s is None:
                continue
            try:
                c = int(n)
                sex = int(s)
            except (ValueError, TypeError):
                continue
            # INSEE: sexe 1 = male, 2 = female
            bucket = counts.setdefault(name, [0, 0])
            if sex == 2:
                bucket[0] += c
            elif sex == 1:
                bucket[1] += c
    return counts


def _weight_for(total: int) -> float:
    """Weight by frequency: names with 10k+ records get full weight."""
    w = min(1.0, max(0.05, (total + 1) ** 0.25 / 10.0))
    return round(w, 4)


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "data")

    sources = {
        "US_SSA": _load_ssa(os.path.join(data_dir, "ssa_alldata.txt"),
                             url=SSA_ZIP_URL),
        "US_CENSUS": _load_census_xlsx(
            os.path.join(data_dir, "census_2020_firstnames_sex.xlsx")),
    }
    sources["FR_INSEE"] = _load_insee()

    for source, counts in sources.items():
        total_records = sum(g + b for g, b in counts.values())
        print(f"  {source}: {len(counts):,} unique names, "
              f"{total_records:,} recorded births")

    # Merge across sources.
    merged: dict = {}
    for source, counts in sources.items():
        for name, (girl, boy) in counts.items():
            bucket = merged.setdefault(name, [0, 0, set()])
            bucket[0] += girl
            bucket[1] += boy
            bucket[2].add(source)

    entries = []
    for name, (girl, boy, srcs) in merged.items():
        total = girl + boy
        if total < 3:
            continue
        ratio = girl / total
        if ratio >= ASSOC_THRESHOLD:
            label = "girl-associated"
        elif ratio <= 1.0 - ASSOC_THRESHOLD:
            label = "boy-associated"
        else:
            label = "uncertain"
        entries.append({
            "name": name,
            "label": label,
            "weight": _weight_for(total),
            "girl_count": girl,
            "boy_count": boy,
            "total_records": total,
            "countries": ",".join(sorted(srcs)),
        })

    # Partition by normalized name so a name never leaks across splits.
    splits = {"train": [], "validation": [], "test": []}
    for entry in entries:
        h = hashlib.md5(entry["name"].encode("utf-8")).digest()[0] % 100
        if h < TRAIN_FRACTION * 100:
            splits["train"].append(entry)
        elif h < (TRAIN_FRACTION + VAL_FRACTION) * 100:
            splits["validation"].append(entry)
        else:
            splits["test"].append(entry)

    out_paths = {
        "train": os.path.join(data_dir, "train.jsonl"),
        "validation": os.path.join(data_dir, "validation.jsonl"),
        "test": os.path.join(data_dir, "test.jsonl"),
    }
    for key, path in out_paths.items():
        with open(path, "w", encoding="utf-8") as f:
            for entry in splits[key]:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(os.path.join(data_dir, "names.jsonl"), "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    report = {
        "sources": {s: {"unique_names": len(c)}
                    for s, c in sources.items()},
        "unique_names_total": len(entries),
        "total_recorded_births": sum(e["total_records"] for e in entries),
        "label_counts": {
            lbl: sum(1 for e in entries if e["label"] == lbl)
            for lbl in ("girl-associated", "boy-associated", "uncertain")
        },
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "association_threshold": ASSOC_THRESHOLD,
    }
    with open(os.path.join(data_dir, "dataset_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nDataset statistics")
    print("------------------")
    print(f"Total examples (unique names): {len(entries):,}")
    print(f"Total recorded births covered: {report['total_recorded_births']:,}")
    for lbl, n in report["label_counts"].items():
        print(f"{lbl}: {n:,}")
    print(f"Train: {len(splits['train']):,}  "
          f"Validation: {len(splits['validation']):,}  "
          f"Test: {len(splits['test']):,}")
    print(f"Report saved to {os.path.join(data_dir, 'dataset_report.json')}")


if __name__ == "__main__":
    main()