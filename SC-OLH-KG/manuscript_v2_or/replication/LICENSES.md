# Provenance and licenses

- **OPSD time series.** The package contains the locally prepared compact hourly
  arrays for 2017–2019. The adjacent manifest `opsd_time_series_extended_v2.npz.manifest.json`
  records the source URL and hash, markets, columns, interpolation, and output hash.
  The 2020-10-06 upstream release attributes the raw data to transmission system
  operators and power exchanges via ENTSO-E Transparency. The compact file is
  subject to applicable upstream terms. ENTSO-E publishes a CC BY 4.0 list
  for specific categories and providers. [OPSD release and source attribution](https://doi.org/10.25832/time_series/2020-10-06)
  · [ENTSO-E free-reuse list](https://transparency.entsoe.eu/content/static_content/download?path=%2FStatic+content%2Fterms+and+conditions%2F231018_List_of_Data_available_for_reuse.pdf)
- **Ingolstadt21 network and route.** The XML files are the project's experimental
  copies of the RESCO Ingolstadt21 scenario (Ault and Sharon, 2021), imported in
  project commit `772b29cd`. They retain the scenario inputs used by the local
  adapter. RESCO publishes its Ingolstadt21 files under the GNU GPL v3. The
  supplied `RESCO_INGOLSTADT21_LICENSE.txt` is the upstream scenario license.
  The Python adapter and the decision-space and baseline JSONs are project files.
  [Pinned upstream RESCO scenario](https://github.com/Pi-Star-Lab/RESCO/tree/f1ed9a174f8de41fc9d8689373b836bc882570dc/resco_benchmark/environments/ingolstadt21)
- **SUMO, Lean, Mathlib, and Python libraries.** The package installs these from
  upstream on demand. Each dependency retains its upstream license. The SUMO
  runtime is provided by the wheels listed in `requirements-live-sumo.txt`.
  [Eclipse SUMO libsumo 1.22.0](https://pypi.org/project/libsumo/1.22.0/)
- **INFORMS manuscript template.** `informs2014.bst` carries the LaTeX Project
  Public License version 1 or later in its header. `informs4.cls` and
  `informs_Logo.pdf` are INFORMS template materials and retain the publisher's terms.
  [INFORMS author resources](https://pubsonline.informs.org/authorresources)

## Public release

This is a staged review artifact. The rights holders still need to confirm the
license for the new GP-FTA source and generated records, the OPSD redistribution
basis, and the terms for redistributing the INFORMS template materials in the
public archive.
