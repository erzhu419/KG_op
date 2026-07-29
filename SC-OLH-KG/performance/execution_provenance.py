"""Fail-closed execution provenance for frozen paper experiments."""

from __future__ import annotations

import os
import re


ENV_FIELDS = {
    "repository_commit": "SCOLHKG_EXECUTION_COMMIT",
    "scolhkg_tree": "SCOLHKG_SCOLHKG_TREE",
    "proof_tree": "SCOLHKG_PROOF_TREE",
    "scripts_tree": "SCOLHKG_SCRIPTS_TREE",
    "method_contract_id": "SCOLHKG_METHOD_CONTRACT_ID",
    "theory_contract_id": "SCOLHKG_THEORY_CONTRACT_ID",
    "snapshot_root": "SCOLHKG_CODE_SNAPSHOT_ROOT",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def execution_provenance_from_env():
    """Read the immutable code snapshot contract passed by the submitter."""

    values = {
        field: os.environ.get(variable, "").strip()
        for field, variable in ENV_FIELDS.items()
    }
    required = os.environ.get(
        "SCOLHKG_EXECUTION_PROVENANCE_REQUIRED", "0"
    ).strip().lower() in {"1", "true", "yes"}
    supplied = any(values.values())
    if not supplied and not required:
        return {
            "status": "unregistered",
            "required": False,
        }

    missing = [
        ENV_FIELDS[field] for field, value in values.items() if not value
    ]
    if missing:
        raise RuntimeError(
            "incomplete frozen execution provenance: "
            + ", ".join(sorted(missing))
        )
    for field in (
        "repository_commit",
        "scolhkg_tree",
        "proof_tree",
        "scripts_tree",
    ):
        if HEX40.fullmatch(values[field]) is None:
            raise RuntimeError(
                f"{ENV_FIELDS[field]} must be a full 40-character Git hash"
            )
    return {
        "status": "frozen",
        "required": required,
        **values,
        "target_outcomes_used_to_select_snapshot": False,
        "runtime_checkpoints_or_model_weights_in_snapshot": False,
    }


def attach_execution_provenance(payload):
    """Attach provenance without changing the experiment's result schema."""

    payload["execution_provenance"] = execution_provenance_from_env()
    return payload
