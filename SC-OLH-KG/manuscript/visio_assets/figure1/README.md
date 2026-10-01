# Figure 1 Python inset assets for Visio

This directory contains only the plot-like elements that are awkward to draw
reliably as native Visio shapes.  It does **not** install or call a Visio skill,
does not create a `.vsdx`, and does not overwrite the manuscript's current
`figures/figure1_profile_space.png`.

## Figure contract

```text
Core conclusion:
  Ordered bounded profiles admit a compact structural description; a fixed,
  outcome-free library can be source-ranked and diversified before target
  evaluation, while search and fresh verification remain separate.
Figure archetype:
  schematic-led composite
Target/output:
  Operations Research overview figure; Python artwork embedded into Visio
Backend:
  Python only (NumPy + Matplotlib)
Final size:
  The full figure is currently placed at 6.50 x 4.875 in in main.pdf.
Panel map:
  1: illustrative three-coordinate projection of raw-space candidate samples
  2: five bounded library examples, one from each declared profile family
  3: readable representatives from the declared 64-profile public library
  5a: illustrative profile thumbnails drawn by the same library renderer
Evidence hierarchy:
  hero evidence: the ordered-profile representation and public library
  validation evidence: mathematically valid, distinct curves and provenance
  controls/robustness: full 64-profile contact sheet and frozen-selection audit
Statistics needed:
  none; these are editorial schematics, not empirical result panels
Source data needed:
  deterministic CSV exports for every curve and point cloud
Image-integrity notes:
  no local retouching; fixed canvas; ordinary ordered-node polylines; no
  drawstyle='steps-*'; PNGs carry 600 dpi metadata
Reviewer risk:
  do not imply that illustrative 5a thumbnails are the realized selected set;
  do not hide the declared-seed versus effective-implementation-seed drift
```

## Recommended assets

The four recommended Visio insertions are:

- `fig1_p1_candidate_points`: a deterministic central sampling cloud with no
  AI-generated radial spokes.  It is explicitly an illustrative 3-D projection,
  not an empirical sample or a statement about the full `d`-dimensional law.
- `fig1_p2_example_profiles`: five actual functions in `[0,1]`, one from each
  declared family: constant, ramp, low-frequency, piecewise, and high-frequency.
  Each is also present in the panel-3 representative set.  This replaces the
  current invalid `[-1,1]` axis and avoids implying that every allowed profile
  is a smooth reconstruction from the retained coordinate modes.
- `fig1_p3_library_representatives`: 16 distinct members generated from the
  manuscript-declared public library.  It contains 2 constants, 2 ramps, and 4
  representatives from each of the low-frequency, piecewise, and high-frequency
  families.  The four high-frequency thumbnails are selected from the readable
  low end of the declared `9:40` range; otherwise `k` near 40 collapses into a
  solid band at the final slot width.  The full 64-profile sheet retains the
  entire spectrum as an audit asset rather than squeezing it into the manuscript
  slot.
- `fig1_p5a_representative_profiles`: five distinct, valid profiles, one per
  family, rendered by the same code and at the same `[0,1]` scale as panel 3.
  Keep the native Visio ellipsis and `n_0=10` text; if space permits, add
  `illustrative profiles` or state the schematic status in the caption.

If panel 5a can be made twice as tall, prefer
`fig1_p5a_representative10_tall`: it shows ten distinct, outcome-free library
members in two rows of five.  It is still schematic, but it avoids using five
icons plus an ellipsis to stand for `n_0=10`.

`fig1_p5a_aligned_selected10_audit` is deliberately **not** the recommended
overview insertion.  It reconstructs the ten profiles recorded for the frozen
aligned-low-frequency diagnostic and is therefore source-outcome-dependent.

## Seed audit boundary

The manuscript and method specification declare library seed `20260808`.
The frozen primary benchmark currently calls the generator with
`family_seed + 991`, i.e. effective PRNG seed `20261799`.  Those two libraries
are different even though their family counts are the same.

Accordingly:

- editorial Figure 1 assets use the manuscript-declared seed `20260808`;
- the aligned selected-10 audit uses effective implementation seed `20261799`;
- `asset_manifest.json` records both contracts and every displayed profile ID;
- the seed mismatch must be resolved in code/manuscript before any curve is
  described as the unique frozen 64-profile library.

## Visio placement

Use SVG first.  Use the transparent 600-dpi PNG only as a fallback.  Do not
stretch either format non-uniformly.

The user-provided 1448 x 1086 reference image has these Python-artwork slots:

| Panel | Pixel slot | Aspect | Insert size |
|---|---:|---:|---:|
| 1 | 264 x 318 | 0.830 | 30.1 x 36.3 mm |
| 2 | 138 x 190 | 0.726 | 15.7 x 21.7 mm |
| 3 | 245 x 234 | 1.047 | 27.9 x 26.7 mm |
| 5a | 238 x 57 | 4.175 | 27.1 x 6.5 mm |

The currently compiled manuscript graphic has a wider panel-3 slot
(approximately 243 x 193 px).  If that graphic becomes the Visio coordinate
master, center the near-square panel-3 asset with side whitespace or create a
new wider native frame; do not distort the curves.

Keep panel numbers, titles, explanatory prose, mathematical axis labels
(`x_1`, `x_2`, `x_d`), inter-panel arrows, and the ellipsis as native Visio
objects.  Only the plotting artwork should be embedded.

## Native Visio text corrections

The Python insets fix the malformed curves, but several nearby native labels
also need correction so that the finished Visio figure matches the method:

- replace `Ordered function ... with low-frequency modes` with
  `Bounded ordered profile h:[0,1] -> [0,1]`;
- replace `Low-frequency basis (DCT-II)` with
  `Retained cosine-coordinate modes, k=0,...,8` (the implementation uses exact
  cosine integrals over reconstruction cells, not a plotted DCT-II transform);
- do not say that the deployed profile is reconstructed as
  `h(t) ~ sum c_k phi_k(t)`: the 0:8 coefficients are a selection coordinate,
  while deployment interpolates the original 128-node profile;
- use the actual coordinate mapping
  `phi(h)=(c~_0,...,c~_8,c~_0^2,...,c~_8^2)` and state that it is standardized
  over the public library to obtain `z`;
- replace `Smooth, ordered, and compact` with wording that also permits the
  declared piecewise and high-frequency families, for example
  `Bounded, ordered, and represented in a fixed 18-D coordinate`.

## Regeneration

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
python3 SC-OLH-KG/manuscript/visio_assets/figure1/render_figure1_insets.py
```

The renderer writes:

- `exports/*.svg`, `exports/*.pdf`, and transparent `exports/*.png`;
- `source_data/*.csv`;
- `asset_manifest.json`, including input/output hashes and QA results.

The renderer fails closed if the library is not exactly 64 unique bounded
functions with the declared `10/6/16/16/16` family counts, if profile nodes are
not strictly ordered, or if a displayed function leaves `[0,1]`.
