#!/usr/bin/env python3
"""Run the legacy fresh-seed verifier against an explicit results root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
GPR_KG_CODE = REPO_ROOT / "Final_Submission" / "GPR_KG_Code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GPR_KG_CODE))

from experiments.ingolstadt21 import (  # noqa: E402
    validate_oos_feasibility as legacy_verifier,
)
from performance.execution_provenance import (  # noqa: E402
    attach_execution_provenance,
)


def _output_path(arguments):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", required=True)
    parsed, _ = parser.parse_known_args(arguments)
    return Path(parsed.out)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results-root", required=True)
    wrapper, verifier_arguments = parser.parse_known_args()
    results_root = Path(wrapper.results_root).resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(
            f"explicit traffic results root does not exist: {results_root}")
    output = _output_path(verifier_arguments)

    legacy_verifier.RESULTS_PATH = results_root
    sys.argv = [
        "experiments.ingolstadt21.validate_oos_feasibility",
        *verifier_arguments,
    ]
    legacy_verifier.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["explicit_results_root"] = str(results_root)
    payload["information_contract"] = {
        "target_oracle_used_for_search_or_selection": False,
        "verification_samples_update_optimizer": False,
        "verification_samples_used_to_reorder_shortlist": False,
    }
    attach_execution_provenance(payload)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


if __name__ == "__main__":
    main()
