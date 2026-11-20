# Rights and acquisition

- **OPSD hourly data.** The preparation manifest fixes the 2020-10-06 OPSD
  `time_series_60min_singleindex.csv`, 21 market labels and 2017–2019. The 63
  populated analytical columns are 21 actual loads, 21 forecast loads and 21
  day-ahead prices. Four market labels use OPSD-derived actual and forecast
  loads, and one uses an OPSD-derived day-ahead price. The publisher's metadata
  identifies ENTSO-E Transparency as the underlying source. The optional wind
  and solar columns named in the preparation manifest are unused. ENTSO-E's CC
  BY 4.0 list applies to selected data items and providers. The supplement
  provides the publisher's pinned download and the preparation program. [OPSD
  metadata](https://data.open-power-system-data.org/time_series/2020-10-06/datapackage.json)
  · [ENTSO-E reuse list](https://transparency.entsoe.eu/content/static_content/download?path=%2FStatic+content%2Fterms+and+conditions%2F231018_List_of_Data_available_for_reuse.pdf)
- **RESCO traffic scenario.** The network and route are reconstructed from the
  published benchmark at commit `01b969ba8248a0f49c5456124ec76682785620a2`.
  The experimental variant disables two signal phases and turns 22 random
  vehicle classes' display color from blue to red. The source directory uses
  GPL version 3 and the applicable notice is included. The preparation command
  verifies the reconstructed input files against the historical bytes.
  [Pinned RESCO scenario and license](https://github.com/Pi-Star-Lab/RESCO/tree/01b969ba8248a0f49c5456124ec76682785620a2/resco_benchmark/environments/ingolstadt21)
- **INFORMS template.** Obtain `informs4.cls` and `informs_Logo.pdf` directly from
  the publisher's Operations Research template. The manuscript used class
  v1.02 dated 2024-06-03. The bundled `informs2014.bst` states the LaTeX Project
  Public License version 1 or later and retains its notice. [INFORMS LaTeX style
  files](https://pubsonline.informs.org/authorportal/latex-style-files)
- **Runtimes.** Install Eclipse SUMO 1.22.0 and the named Python dependencies
  from upstream. Eclipse SUMO's release provides EPL 2.0 with its stated
  secondary GPL option. Lean and Mathlib use Apache 2.0 and are obtained
  separately at the pinned revisions. [Eclipse SUMO v1.22.0
  license](https://github.com/eclipse-sumo/sumo/blob/v1_22_0/LICENSE)
  · [Lean license](https://github.com/leanprover/lean4/blob/v4.31.0/LICENSE)
  · [Mathlib license](https://github.com/leanprover-community/mathlib4/blob/v4.31.0/LICENSE)

The proposed release licenses for author-owned software and author-owned
synthetic artifacts are MIT and CC BY 4.0, respectively. Public release remains
subject to rights-holder and attribution confirmation. The authors' proposed
terms do not change the upstream licenses or apply to the third-party inputs.
