# Missing or Blocked Reference Notes

This folder contains the references cited from `markdown/GPT_plan.md` and
`markdown/GPT_roadmap.md` that could be downloaded without bypassing paywalls or
site protections. The notes below track items that were blocked or only partly
covered by an open equivalent.

## Covered by Open Equivalent

- **Stochastic Kriging for Simulation Metamodeling** (`10.1287/opre.1090.0754`):
  the Operations Research page was blocked by INFORMS/ACM, but the WSC 2008
  conference version and Staum's WSC 2009 stochastic-kriging tutorial were saved.
- **The Knowledge-Gradient Policy for Correlated Normal Beliefs**
  (`10.1287/ijoc.1080.0314`): INFORMS was blocked, but the Cornell author version
  was saved.
- **The Correlated Knowledge Gradient for Simulation Optimization of Continuous
  Parameters** (`10.1137/100801275`): SIAM was blocked, but the Cornell author
  version was saved.
- **Violation-aware contextual Bayesian optimization**: ScienceDirect was
  blocked, but arXiv and MERL technical-report PDFs were saved.
- **Comparison of Kriging-based algorithms for simulation optimization with
  heterogeneous noise**: ScienceDirect was blocked, but an open Semantic Scholar
  PDF was saved.
- **Preferential Multi-Objective Bayesian Optimization**: OpenReview direct PDF
  challenged command-line requests, but the arXiv equivalent was saved.
- **The Price of Robustness**: the INFORMS/RePEc landing page did not yield a PDF,
  but an open robustopt.com PDF was saved.
- **Gaussian Process Regression with Heteroscedastic or Non-Gaussian Residuals**:
  ResearchGate was blocked, but the arXiv PDF was saved.

## Still Needs Manual or Institutional Access

- **A Knowledge-Gradient Policy for Sequential Information Collection**
  (`10.1137/070693424`): SIAM was blocked. A Cornell presentation PDF was saved,
  but not the full article PDF.
- **Simulation Budget Allocation for Further Enhancing the Efficiency of Ordinal
  Optimization** (`10.1023/A:1008349927281`): Springer/DeepBlue pages were blocked
  by login or Cloudflare; only landing snapshots are saved.
- **Practical Heteroscedastic Gaussian Process Modeling for Large Simulation
  Experiments** (`10.1080/10618600.2018.1458625`): Taylor & Francis returned 403.
  Related `hetGP` JSS/package PDFs are saved, but not this article PDF.
- **Using the knowledge gradient acquisition function in Bayesian optimization
  when searching for robust solutions** (`10.1080/0305215X.2022.2145604`): Taylor
  & Francis returned 403.
- **Machine Learning-Assisted Stochastic Kriging Metamodel for Offline Simulation
  Online Application** (`10.1287/ijoc.2023.0130`): INFORMS was blocked; the SSRN
  page was also behind Cloudflare in command-line access.
- **A Heteroscedastic Robust Bayesian Optimization Method for Simulation-Based
  Transportation Problems** (`10.1287/trsc.2024.0840`): INFORMS was blocked.
- **Spatial variability-based sample size allocation for stratified sampling**
  (`S0341816221003672`): ScienceDirect returned 403.

To retry Open Access lookup for DOI-based references, run:

```bash
UNPAYWALL_EMAIL=your.email@example.com python3 reference/download_references.py
```

