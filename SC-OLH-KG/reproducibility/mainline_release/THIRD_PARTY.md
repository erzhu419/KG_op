# Third-party components

The attachment redistributes the project's selected Python and Lean source
files and its generated synthetic assets. It does not vendor Python wheels,
Lean binaries, Mathlib source/cache, external Energy datasets or official
baseline implementations from the historical Atlas package.

| Component | Use | Upstream license / delivery |
| --- | --- | --- |
| Python | Runtime and standard library | Python Software Foundation license; installed separately |
| [NumPy](https://numpy.org/doc/stable/license.html) | Numerical arrays and random streams | BSD-3-Clause; installed from pinned requirements |
| [SciPy](https://github.com/scipy/scipy/blob/main/LICENSE.txt) | Statistical distributions and numerical routines | BSD-3-Clause; installed from pinned requirements |
| [Requests](https://github.com/psf/requests/blob/main/LICENSE) | Imported by shared production modules | Apache-2.0; installed from pinned requirements |
| [urllib3](https://github.com/urllib3/urllib3/blob/main/LICENSE.txt) | Requests dependency | MIT; installed separately |
| [certifi](https://github.com/certifi/python-certifi/blob/master/LICENSE) | Requests CA bundle | MPL-2.0; installed separately |
| [idna](https://github.com/kjd/idna/blob/master/LICENSE.md) | Requests dependency | BSD-3-Clause; installed separately |
| [charset-normalizer](https://github.com/jawah/charset_normalizer/blob/master/LICENSE) | Requests dependency | MIT; installed separately |
| [Lean 4](https://github.com/leanprover/lean4/blob/master/LICENSE) | Proof checker | Apache-2.0; installed separately |
| [Mathlib](https://github.com/leanprover-community/mathlib4/blob/master/LICENSE) | Locked mathematical dependency | Apache-2.0; fetched from the committed lock |

The current-only Lean export includes the project's dependency lock but no
dependency source or compiled artifacts. Its lock also identifies the
transitive Lean packages; their upstream terms apply when downloaded. The
runtime requirement pins retain the acceptance environment versions. Optional
code paths in the shared production source are not invoked by the registered
AFTER/TS small case and may need their own upstream packages if used separately.

The synthetic task names describe implemented benchmark models; the included
generated records are not real inventory, queue or financial/customer data.
Self-owned source and assets use the licenses next to this notice. No license
here relicenses third-party code, data, names or dependencies. Source asset
SHA256 entries allow recipients to confirm the frozen assets actually supplied.
