#!/usr/bin/env python3
"""Restore the pinned third-party inputs in an extracted GP-FTA supplement."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
SC = ROOT / "SC-OLH-KG"
OPSD = SC / "data/external/opsd_time_series_extended_v2.npz"
INGOLSTADT = SC / "data/ingolstadt21/data"
TEX = SC / "manuscript_v2_or"
RESCO_UPSTREAM = (
    "https://raw.githubusercontent.com/Pi-Star-Lab/RESCO/"
    "01b969ba8248a0f49c5456124ec76682785620a2/resco_benchmark/environments/ingolstadt21/"
)


def compute_npz_arrays_sha256(npz_bytes: bytes) -> str:
    import numpy as np

    digest = hashlib.sha256()
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
        for key in sorted(archive.files):
            arr = archive[key]
            digest.update(key.encode("utf-8") + b"\0")
            digest.update(str(arr.dtype).encode("utf-8") + b"\0")
            digest.update(repr(tuple(int(s) for s in arr.shape)).encode("utf-8") + b"\0")
            digest.update(arr.tobytes() + b"\0")
    return digest.hexdigest()


def install(path: Path, payload: bytes, expected: str) -> None:
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Input differs from the historical file: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    print(f"Verified {path.relative_to(ROOT)}")


def restore_resco(hashes: dict) -> None:
    for name in ("ingolstadt21.net.xml", "ingolstadt21.rou.xml"):
        with urlopen(RESCO_UPSTREAM + name, timeout=30) as response:
            source = response.read()
        source = source.replace(b"\r\n", b"\n")
        if name.endswith("net.xml"):
            # The experimental network disabled two upstream signal phases.
            for line in (b'<phase duration="1"  state="rrrr"/>',
                         b'<phase duration="25" state="rrrrrrGGGGrr"/>'):
                source = source.replace(b"        " + line + b"\n", b"        <!-- " + line + b" -->\n", 1)
        else:
            # The experimental routes changed the display color of random vehicles.
            source = re.sub(rb'(<vType id="random_[0-9]+" vClass="passenger" color=")blue("/>)',
                            rb'\g<1>red\2', source)
        path = INGOLSTADT / name
        install(path, source, hashes[path.relative_to(ROOT).as_posix()])


def restore_informs(archive_path: Path, hashes: dict) -> None:
    with ZipFile(archive_path) as archive:
        for name in ("informs4.cls", "informs_Logo.pdf"):
            members = [member for member in archive.namelist() if Path(member).name == name]
            if len(members) != 1:
                raise ValueError(f"Publisher template archive must contain one {name}")
            path = TEX / name
            expected = hashes[path.relative_to(ROOT).as_posix()]
            raw = archive.read(members[0])
            if name.endswith(".cls") and hashlib.sha256(raw).hexdigest() != expected:
                lf = raw.replace(b"\r\n", b"\n")
                crlf = lf.replace(b"\n", b"\r\n")
                for candidate in (lf, crlf):
                    if hashlib.sha256(candidate).hexdigest() == expected:
                        raw = candidate
                        break
            install(path, raw, expected)


def restore_opsd(csv: Path | None, prepared: Path | None, review_meta: dict) -> None:
    manifest = json.loads(OPSD.with_suffix(".npz.manifest.json").read_bytes())
    expected_outer = manifest.pop("output")["sha256"]
    expected_arrays = review_meta.get("opsd_uncompressed_arrays_sha256")
    if prepared is not None:
        install(OPSD, prepared.read_bytes(), expected_outer)
        return

    sys.path.insert(0, str(SC))
    import numpy as np
    from data.opsd import preprocess_opsd

    with TemporaryDirectory(prefix="gpfta-opsd-") as temp:
        output = Path(temp) / OPSD.name
        raw = preprocess_opsd(output, source=csv or manifest["official_url"],
                              markets=manifest["markets"], years=manifest["years"],
                              interpolation_limit=manifest["interpolation_limit_hours"])
        if raw["source"]["sha256"] != manifest["source"]["sha256"]:
            raise ValueError("OPSD source differs from the pinned official release")
        raw["source"] = manifest["source"]
        raw.pop("output")
        if raw != manifest:
            raise ValueError("OPSD processing metadata differs from the pinned archive")
        with np.load(output, allow_pickle=False) as archive:
            arrays = dict(archive)
        arrays["metadata_json"] = np.asarray(json.dumps(manifest, sort_keys=True), dtype="U")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        payload = buffer.getvalue()
        if expected_arrays is not None:
            if compute_npz_arrays_sha256(payload) != expected_arrays:
                raise ValueError("OPSD uncompressed array contents differ from the pinned archive")
            OPSD.parent.mkdir(parents=True, exist_ok=True)
            OPSD.write_bytes(payload)
            print(f"Verified {OPSD.relative_to(ROOT)} (exact array-level SHA-256)")
        else:
            install(OPSD, payload, expected_outer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_subparsers(dest="input", required=True)
    inputs.add_parser("resco")
    tex = inputs.add_parser("informs")
    tex.add_argument("--zip", type=Path, required=True, help="Official Operations Research LaTeX template ZIP")
    opsd = inputs.add_parser("opsd")
    opsd_source = opsd.add_mutually_exclusive_group()
    opsd_source.add_argument("--csv", type=Path, help="Local copy of the official CSV")
    opsd_source.add_argument("--prepared", type=Path, help="Existing authorized copy of the exact NPZ")
    args = parser.parse_args()
    review_meta = json.loads((ROOT / "REVIEW_INPUTS.json").read_bytes())
    hashes = review_meta["omitted_input_sha256"]
    if args.input == "resco":
        restore_resco(hashes)
    elif args.input == "informs":
        restore_informs(args.zip, hashes)
    else:
        restore_opsd(args.csv, args.prepared, review_meta)


if __name__ == "__main__":
    main()
