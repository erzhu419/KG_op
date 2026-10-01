#!/usr/bin/env python3
"""Render deterministic Figure 1 plot insets for later Visio composition.

The script intentionally produces only plot-like artwork.  Panel titles,
explanatory text, arrows, and numbering remain native Visio objects.
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
EXPORT_DIR = HERE / "exports"
DATA_DIR = HERE / "source_data"

sys.path.insert(0, str(PROJECT_ROOT))
from problems.randomized_profiles import (  # noqa: E402
    generate_structural_profile_library,
)


DECLARED_LIBRARY_SEED = 20260808
EFFECTIVE_LIBRARY_SEED = DECLARED_LIBRARY_SEED + 991
MAXIMUM_GENERATED_FREQUENCY = 40
CANDIDATE_POINT_SEED = 20260811
CONTRACT_DATE = "2026-08-11"
PNG_DPI = 600

FAMILY_ORDER = (
    "constant",
    "ramp",
    "low_frequency",
    "piecewise",
    "high_frequency",
)
EXPECTED_FAMILY_COUNTS = {
    "constant": 10,
    "ramp": 6,
    "low_frequency": 16,
    "piecewise": 16,
    "high_frequency": 16,
}

COLORS = {
    "navy": "#090D4E",
    "selected": "#3150EB",
    "candidate": "#7E8B9C",
    "border": "#A1B6D3",
    "wire": "#657386",
    "blue": "#2924C4",
    "green": "#1D6B22",
    "purple": "#6F45CA",
    "orange": "#FE8019",
}


def _configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 5.5,
        "axes.linewidth": 0.6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "sc-olh-kg-figure1-v1",
        "pdf.fonttype": 42,
        "savefig.transparent": True,
    })


def _mm(value: float) -> float:
    return float(value) / 25.4


def _new_figure(width_mm: float, height_mm: float) -> plt.Figure:
    fig = plt.figure(figsize=(_mm(width_mm), _mm(height_mm)), frameon=False)
    fig.patch.set_alpha(0.0)
    return fig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_asset(fig: plt.Figure, stem: str) -> list[Path]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [EXPORT_DIR / f"{stem}.{suffix}" for suffix in ("svg", "pdf", "png")]
    fig.savefig(
        paths[0],
        format="svg",
        transparent=True,
        metadata={
            "Creator": "SC-OLH-KG deterministic Figure 1 inset renderer",
            "Date": CONTRACT_DATE,
            "Title": stem,
        },
    )
    fixed_time = datetime(2026, 8, 11, tzinfo=timezone.utc)
    fig.savefig(
        paths[1],
        format="pdf",
        transparent=True,
        metadata={
            "Creator": "SC-OLH-KG deterministic Figure 1 inset renderer",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
            "Title": stem,
        },
    )
    fig.savefig(
        paths[2],
        format="png",
        dpi=PNG_DPI,
        transparent=True,
        metadata={
            "Software": "SC-OLH-KG deterministic Figure 1 inset renderer",
            "Creation Time": CONTRACT_DATE,
        },
    )
    plt.close(fig)
    return paths


def _write_csv(path: Path, header: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _library(seed: int):
    return generate_structural_profile_library(
        64,
        dimension=128,
        seed=int(seed),
        maximum_frequency=MAXIMUM_GENERATED_FREQUENCY,
    )


def _validate_library(library, *, label: str) -> dict:
    if len(library) != 64:
        raise RuntimeError(f"{label}: expected 64 profiles, found {len(library)}")
    counts = Counter(profile.family for profile in library)
    if dict(counts) != EXPECTED_FAMILY_COUNTS:
        raise RuntimeError(
            f"{label}: family counts drifted: {dict(counts)} != "
            f"{EXPECTED_FAMILY_COUNTS}"
        )
    matrix = np.asarray([profile.values for profile in library], dtype=float)
    nodes = np.asarray([profile.nodes for profile in library], dtype=float)
    if matrix.shape != (64, 128) or nodes.shape != (64, 128):
        raise RuntimeError(f"{label}: unexpected library array shape")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(nodes)):
        raise RuntimeError(f"{label}: non-finite profile data")
    if float(np.min(matrix)) < 0.0 or float(np.max(matrix)) > 1.0:
        raise RuntimeError(f"{label}: profile leaves [0,1]")
    if not np.all(np.diff(nodes, axis=1) > 0.0):
        raise RuntimeError(f"{label}: nodes are not strictly ordered")
    unique_rows = len({
        tuple(np.rint(row * 1e8).astype(np.int64)) for row in matrix
    })
    if unique_rows != 64:
        raise RuntimeError(f"{label}: duplicate profiles detected")
    return {
        "profile_count": 64,
        "family_counts": dict(counts),
        "unique_profiles_at_1e-8": unique_rows,
        "nodes_per_profile": 128,
        "strictly_increasing_nodes": True,
        "one_value_per_declared_node": True,
        "finite": True,
        "value_min": float(np.min(matrix)),
        "value_max": float(np.max(matrix)),
    }


def _representatives(library, family: str, count: int):
    members = [profile for profile in library if profile.family == family]
    if count > len(members):
        raise ValueError(f"requested too many representatives for {family}")
    matrix = np.asarray([profile.values for profile in members], dtype=float)
    distances = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
    selected = [int(np.argmin(np.mean(distances, axis=1)))]
    while len(selected) < count:
        nearest = np.min(distances[:, selected], axis=1)
        nearest[selected] = -np.inf
        selected.append(int(np.argmax(nearest)))
    return [members[index] for index in selected]


def _dominant_high_frequency(profile) -> int:
    """Infer the generator's dominant integer frequency for display QA."""
    nodes = np.asarray(profile.nodes, dtype=float)
    centered = np.asarray(profile.values, dtype=float)
    centered = centered - float(np.mean(centered))
    scores = []
    for frequency in range(9, MAXIMUM_GENERATED_FREQUENCY + 1):
        basis = np.column_stack([
            np.cos(np.pi * frequency * nodes),
            np.sin(np.pi * frequency * nodes),
        ])
        coefficients = np.linalg.lstsq(basis, centered, rcond=None)[0]
        scores.append((float(np.linalg.norm(coefficients)), frequency))
    return int(max(scores)[1])


def _display_representatives(library, family: str, count: int):
    if family != "high_frequency":
        return _representatives(library, family, count)
    members = [profile for profile in library if profile.family == family]
    ranked = sorted(
        members,
        key=lambda profile: (_dominant_high_frequency(profile), profile.profile_id),
    )
    # At manuscript width, profiles near k=40 rasterize into visually duplicate
    # bands.  Use the readable low end in the representative sheet and preserve
    # all 64 profiles, including the full frequency range, in the audit sheet.
    selected = []
    seen_frequencies = set()
    for profile in ranked:
        frequency = _dominant_high_frequency(profile)
        if frequency in seen_frequencies:
            continue
        selected.append(profile)
        seen_frequencies.add(frequency)
        if len(selected) == count:
            return selected
    for profile in ranked:
        if profile not in selected:
            selected.append(profile)
        if len(selected) == count:
            return selected
    raise RuntimeError("insufficient high-frequency display representatives")


def _profile_lookup(library) -> dict[str, object]:
    return {profile.profile_id: profile for profile in library}


def _draw_profile_card(
    fig: plt.Figure,
    rect,
    profile,
    *,
    line_color: str = COLORS["selected"],
    line_width: float = 0.8,
) -> None:
    ax = fig.add_axes(rect)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    card = FancyBboxPatch(
        (0.02, 0.03),
        0.96,
        0.94,
        transform=ax.transAxes,
        boxstyle="round,pad=0.006,rounding_size=0.10",
        facecolor="white",
        edgecolor=COLORS["border"],
        linewidth=0.48,
        zorder=0,
    )
    ax.add_patch(card)
    nodes = np.asarray(profile.nodes, dtype=float)
    values = np.asarray(profile.values, dtype=float)
    # Ordinary ordered-node polylines avoid the visual ambiguity of vertical
    # step-renderer joins while preserving one value at every declared node.
    ax.plot(
        0.09 + 0.82 * nodes,
        0.13 + 0.74 * values,
        color=line_color,
        linewidth=line_width,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
    )


def _draw_grid(
    profiles,
    *,
    rows: int,
    cols: int,
    width_mm: float,
    height_mm: float,
    gap_x: float,
    gap_y: float,
    line_width: float,
) -> plt.Figure:
    if len(profiles) != rows * cols:
        raise ValueError("profile count does not match grid")
    fig = _new_figure(width_mm, height_mm)
    margin_x = 0.015
    margin_y = 0.015
    cell_w = (1.0 - 2.0 * margin_x - (cols - 1) * gap_x) / cols
    cell_h = (1.0 - 2.0 * margin_y - (rows - 1) * gap_y) / rows
    for index, profile in enumerate(profiles):
        row, col = divmod(index, cols)
        left = margin_x + col * (cell_w + gap_x)
        bottom = 1.0 - margin_y - (row + 1) * cell_h - row * gap_y
        _draw_profile_card(
            fig,
            [left, bottom, cell_w, cell_h],
            profile,
            line_width=line_width,
        )
    return fig


def _render_candidate_points():
    fig = _new_figure(30.1, 36.3)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    rng = np.random.default_rng(CANDIDATE_POINT_SEED)
    central = np.clip(rng.normal(0.5, 0.18, size=(240, 3)), 0.025, 0.975)
    shell = rng.uniform(0.05, 0.95, size=(64, 3))
    points = np.vstack([central, shell])

    origin = np.asarray([0.15, 0.13])
    basis = np.asarray([
        [0.53, -0.01],
        [0.19, 0.18],
        [0.00, 0.62],
    ])

    def project(values):
        return origin + np.asarray(values, dtype=float) @ basis

    vertices = {
        vertex: project(vertex)
        for vertex in itertools.product((0.0, 1.0), repeat=3)
    }
    edges = []
    for vertex in vertices:
        for axis in range(3):
            other = list(vertex)
            other[axis] = 1.0 - other[axis]
            other = tuple(other)
            if vertex < other:
                edges.append((vertex, other))
    for start, stop in edges:
        back = start[1] == 1.0 and stop[1] == 1.0
        xy = np.vstack([vertices[start], vertices[stop]])
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=COLORS["wire"],
            linewidth=0.48,
            linestyle=(0, (2.2, 1.8)) if back else "-",
            alpha=0.70 if back else 0.88,
            zorder=1,
        )

    xy = project(points)
    order = np.argsort(points[:, 1])
    depth = points[order, 1]
    ax.scatter(
        xy[order, 0],
        xy[order, 1],
        s=2.3 + 2.7 * (1.0 - depth),
        c=COLORS["candidate"],
        alpha=0.48 + 0.34 * (1.0 - depth),
        linewidths=0.0,
        zorder=2,
    )

    for vector, scale in (
        (basis[0], 1.10),
        (basis[1], 1.40),
        (basis[2], 1.10),
    ):
        end = origin + scale * vector
        ax.annotate(
            "",
            xy=end,
            xytext=origin,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#111820",
                "lw": 0.62,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
                "mutation_scale": 5.0,
            },
            zorder=3,
        )
    return fig, points


def _render_example_profiles(declared_library):
    # Use the second farthest-first representative from each family.  Those
    # five profiles also occur in the 16-member panel-3 representative sheet,
    # while panel 5a uses the first representative and is therefore not a
    # duplicate strip.
    profiles = [
        _display_representatives(declared_library, family, 2)[1]
        for family in FAMILY_ORDER
    ]
    if len({profile.profile_id for profile in profiles}) != 5:
        raise RuntimeError("panel 2 example profiles are not unique")
    fig = _draw_grid(
        profiles,
        rows=5,
        cols=1,
        width_mm=15.7,
        height_mm=21.7,
        gap_x=0.0,
        gap_y=0.012,
        line_width=0.76,
    )
    return fig, profiles


def _render_library_representatives(declared_library):
    counts = {
        "constant": 2,
        "ramp": 2,
        "low_frequency": 4,
        "piecewise": 4,
        "high_frequency": 4,
    }
    profiles = []
    for family in FAMILY_ORDER:
        profiles.extend(
            _display_representatives(declared_library, family, counts[family])
        )
    if len({profile.profile_id for profile in profiles}) != 16:
        raise RuntimeError("panel 3 representative selection is not unique")
    fig = _draw_grid(
        profiles,
        rows=4,
        cols=4,
        width_mm=27.9,
        height_mm=26.7,
        gap_x=0.014,
        gap_y=0.012,
        line_width=0.78,
    )
    return fig, profiles


def _render_library_all64(declared_library):
    grouped = []
    for family in FAMILY_ORDER:
        grouped.extend(profile for profile in declared_library if profile.family == family)
    fig = _draw_grid(
        grouped,
        rows=8,
        cols=8,
        width_mm=55.0,
        height_mm=55.0,
        gap_x=0.007,
        gap_y=0.007,
        line_width=0.58,
    )
    return fig, grouped


def _render_representative_strip(declared_library):
    by_family = {
        family: _display_representatives(declared_library, family, 1)[0]
        for family in FAMILY_ORDER
    }
    order = (
        "low_frequency",
        "ramp",
        "piecewise",
        "high_frequency",
        "constant",
    )
    profiles = [by_family[family] for family in order]
    fig = _draw_grid(
        profiles,
        rows=1,
        cols=5,
        width_mm=27.1,
        height_mm=6.5,
        gap_x=0.020,
        gap_y=0.0,
        line_width=0.78,
    )
    return fig, profiles


def _render_representative10_tall(declared_library):
    pairs = {
        family: _display_representatives(declared_library, family, 2)
        for family in FAMILY_ORDER
    }
    profiles = (
        [pairs[family][0] for family in FAMILY_ORDER]
        + [pairs[family][1] for family in FAMILY_ORDER]
    )
    if len({profile.profile_id for profile in profiles}) != 10:
        raise RuntimeError("panel 5a tall representative set is not unique")
    fig = _draw_grid(
        profiles,
        rows=2,
        cols=5,
        width_mm=27.1,
        height_mm=13.0,
        gap_x=0.020,
        gap_y=0.025,
        line_width=0.74,
    )
    return fig, profiles


def _find_values_for_key(value, target_key: str) -> list:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == target_key:
                found.append(child)
            found.extend(_find_values_for_key(child, target_key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_values_for_key(child, target_key))
    return found


def _load_frozen_selected_ids() -> tuple[list[str], Path]:
    path = PROJECT_ROOT / "paper_artifacts/or_review/review_v2_supplemental_diagnostics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = _find_values_for_key(payload, "source_selected_profile_ids")
    if len(matches) != 1 or len(matches[0]) != 10:
        raise RuntimeError("could not identify one frozen selected-10 profile list")
    profile_ids = [str(value) for value in matches[0]]
    if len(set(profile_ids)) != 10:
        raise RuntimeError("frozen selected-10 list contains duplicates")
    return profile_ids, path


def _render_frozen_selection(effective_library, selected_ids):
    lookup = _profile_lookup(effective_library)
    missing = sorted(set(selected_ids) - set(lookup))
    if missing:
        raise RuntimeError(f"selected profile IDs missing from library: {missing}")
    profiles = [lookup[profile_id] for profile_id in selected_ids]
    fig = _draw_grid(
        profiles,
        rows=1,
        cols=10,
        width_mm=54.2,
        height_mm=13.0,
        gap_x=0.010,
        gap_y=0.0,
        line_width=0.72,
    )
    return fig, profiles


def _write_source_data(points, panel2_profiles, declared_library, effective_library):
    _write_csv(
        DATA_DIR / "fig1_p1_candidate_points.csv",
        ["point_id", "x1", "x2", "xd"],
        ([f"point_{index:04d}", *map(float, row)] for index, row in enumerate(points)),
    )
    _write_csv(
        DATA_DIR / "fig1_p2_example_profiles.csv",
        ["profile_id", "family", "node_index", "t", "h"],
        (
            [
                profile.profile_id,
                profile.family,
                index,
                float(profile.nodes[index]),
                float(profile.values[index]),
            ]
            for profile in panel2_profiles
            for index in range(len(profile.nodes))
        ),
    )
    for name, library, seed in (
        ("declared", declared_library, DECLARED_LIBRARY_SEED),
        ("effective", effective_library, EFFECTIVE_LIBRARY_SEED),
    ):
        _write_csv(
            DATA_DIR / f"profile_library_{name}_seed_{seed}.csv",
            ["profile_id", "family", "node_index", "t", "h"],
            (
                [
                    profile.profile_id,
                    profile.family,
                    index,
                    float(profile.nodes[index]),
                    float(profile.values[index]),
                ]
                for profile in library
                for index in range(len(profile.nodes))
            ),
        )


def _source_input(path: Path) -> dict:
    return {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _file_record(path: Path) -> dict:
    record = {
        "path": str(path.relative_to(HERE)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix.lower() == ".png":
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
            record["pixel_size"] = [int(image.width), int(image.height)]
            record["mode"] = image.mode
            record["dpi"] = [float(dpi[0]), float(dpi[1])] if dpi else None
    return record


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def main() -> None:
    _configure_matplotlib()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    declared_library = _library(DECLARED_LIBRARY_SEED)
    effective_library = _library(EFFECTIVE_LIBRARY_SEED)
    declared_qa = _validate_library(declared_library, label="declared library")
    effective_qa = _validate_library(effective_library, label="effective library")

    declared_matrix = np.asarray([p.values for p in declared_library], dtype=float)
    effective_matrix = np.asarray([p.values for p in effective_library], dtype=float)
    libraries_equal = bool(np.array_equal(declared_matrix, effective_matrix))
    if libraries_equal:
        raise RuntimeError("declared and effective seeds unexpectedly produced one library")

    generated = []
    fig, points = _render_candidate_points()
    generated.extend(_save_asset(fig, "fig1_p1_candidate_points"))

    fig, panel2_profiles = _render_example_profiles(declared_library)
    generated.extend(_save_asset(fig, "fig1_p2_example_profiles"))

    fig, panel3_profiles = _render_library_representatives(declared_library)
    generated.extend(_save_asset(fig, "fig1_p3_library_representatives"))

    fig, all64_profiles = _render_library_all64(declared_library)
    generated.extend(_save_asset(fig, "fig1_p3_library_all64_audit"))

    fig, panel5_profiles = _render_representative_strip(declared_library)
    generated.extend(_save_asset(fig, "fig1_p5a_representative_profiles"))

    fig, panel5_tall_profiles = _render_representative10_tall(declared_library)
    generated.extend(_save_asset(fig, "fig1_p5a_representative10_tall"))

    frozen_ids, frozen_artifact = _load_frozen_selected_ids()
    fig, frozen_profiles = _render_frozen_selection(effective_library, frozen_ids)
    generated.extend(_save_asset(fig, "fig1_p5a_aligned_selected10_audit"))

    _write_source_data(points, panel2_profiles, declared_library, effective_library)
    source_data = sorted(DATA_DIR.glob("*.csv"))

    input_paths = [
        PROJECT_ROOT / "problems/randomized_profiles.py",
        PROJECT_ROOT / "performance/benchmark_profile_stress_suite.py",
        PROJECT_ROOT / "performance/manifests/profile_atlas_v2_method_spec.json",
        PROJECT_ROOT / "manuscript/sections/04_atlas.tex",
        frozen_artifact,
    ]
    manifest = {
        "schema_version": 1,
        "contract_id": "figure1_visio_python_insets_v1",
        "contract_date": CONTRACT_DATE,
        "status": "complete",
        "backend": "python_matplotlib_only",
        "repository_commit": _git_commit(),
        "figure_contract": {
            "core_conclusion": (
                "Ordered bounded profiles admit a compact structural description; "
                "a fixed outcome-free library can be source-ranked and diversified "
                "before target evaluation, while search and verification stay separate."
            ),
            "archetype": "schematic-led composite",
            "empirical_result_panels": False,
            "final_full_figure_size_in": [6.5, 4.875],
            "editable_visio_elements": (
                "panel numbers, titles, prose, arrows, legends, and ellipses"
            ),
            "embedded_python_elements": (
                "candidate cloud, wireframe/axis arrows, curves, and profile "
                "thumbnail cards"
            ),
        },
        "seed_boundary": {
            "manuscript_declared_library_seed": DECLARED_LIBRARY_SEED,
            "frozen_benchmark_effective_library_seed": EFFECTIVE_LIBRARY_SEED,
            "effective_seed_expression": "family_seed + 991",
            "maximum_generated_frequency": MAXIMUM_GENERATED_FREQUENCY,
            "libraries_equal": libraries_equal,
            "recommended_editorial_assets_use": "manuscript_declared_library_seed",
            "frozen_selected10_audit_uses": "frozen_benchmark_effective_library_seed",
            "warning": (
                "Resolve the manuscript-versus-implementation seed drift before "
                "describing either generated library as the unique frozen library."
            ),
        },
        "panels": {
            "1": {
                "asset": "fig1_p1_candidate_points",
                "role": "illustrative 3-D projection of raw-space candidate samples",
                "point_count": int(len(points)),
                "rng_seed": CANDIDATE_POINT_SEED,
                "distribution": "240 clipped Gaussian central points plus 64 uniform shell points",
                "not_empirical": True,
            },
            "2": {
                "asset": "fig1_p2_example_profiles",
                "role": "bounded examples from every declared profile family",
                "function_count": 5,
                "profile_ids": [p.profile_id for p in panel2_profiles],
                "families": [p.family for p in panel2_profiles],
                "all_examples_are_members_of_panel3_representatives": all(
                    p.profile_id in {q.profile_id for q in panel3_profiles}
                    for p in panel2_profiles
                ),
                "domain": [0.0, 1.0],
                "range": [0.0, 1.0],
            },
            "3": {
                "recommended_asset": "fig1_p3_library_representatives",
                "audit_asset": "fig1_p3_library_all64_audit",
                "representative_profile_ids": [p.profile_id for p in panel3_profiles],
                "representative_families": [p.family for p in panel3_profiles],
                "displayed_high_frequencies": [
                    _dominant_high_frequency(p)
                    for p in panel3_profiles
                    if p.family == "high_frequency"
                ],
                "all64_layout_profile_ids": [p.profile_id for p in all64_profiles],
                "ordinary_ordered_node_polylines": True,
                "step_drawstyle_used": False,
            },
            "5a": {
                "recommended_asset": "fig1_p5a_representative_profiles",
                "recommended_status": "illustrative_only_composition_varies",
                "representative_profile_ids": [p.profile_id for p in panel5_profiles],
                "representative_families": [p.family for p in panel5_profiles],
                "recommended_tall_asset": "fig1_p5a_representative10_tall",
                "tall_representative_profile_ids": [
                    p.profile_id for p in panel5_tall_profiles
                ],
                "tall_representative_families": [
                    p.family for p in panel5_tall_profiles
                ],
                "audit_asset": "fig1_p5a_aligned_selected10_audit",
                "audit_regime": "aligned_low_frequency",
                "audit_profile_ids": frozen_ids,
                "audit_family_counts": dict(Counter(p.family for p in frozen_profiles)),
                "audit_depends_on_source_outcomes": True,
            },
        },
        "qa": {
            "declared_library": declared_qa,
            "effective_library": effective_qa,
            "panel3_representatives_unique": len({p.profile_id for p in panel3_profiles}) == 16,
            "panel2_representatives_unique": len({p.profile_id for p in panel2_profiles}) == 5,
            "panel5_representatives_unique": len({p.profile_id for p in panel5_profiles}) == 5,
            "panel5_tall_representatives_unique": len({
                p.profile_id for p in panel5_tall_profiles
            }) == 10,
            "panel2_all_values_in_unit_interval": all(
                float(np.min(profile.values)) >= 0.0
                and float(np.max(profile.values)) <= 1.0
                for profile in panel2_profiles
            ),
            "all_profile_artwork_uses_one_y_per_ordered_node": True,
            "fixed_canvas_no_tight_bbox": True,
            "png_dpi": PNG_DPI,
        },
        "source_inputs": [_source_input(path) for path in input_paths],
        "outputs": [_file_record(path) for path in sorted(generated + source_data)],
    }
    manifest_path = HERE / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete",
        "manifest": str(manifest_path),
        "exports": len(generated),
        "source_data_files": len(source_data),
        "declared_library_sha256": _sha256(
            DATA_DIR / f"profile_library_declared_seed_{DECLARED_LIBRARY_SEED}.csv"
        ),
        "effective_library_sha256": _sha256(
            DATA_DIR / f"profile_library_effective_seed_{EFFECTIVE_LIBRARY_SEED}.csv"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
