"""Export the first registered Stage85 Inventory/observation_1 release case.

Read existing JSON and the small neighbor NPZ only. No model restoration,
source fitting, target simulation, checkpoint access or outcome-based selection.
The source pickle files are referenced at their release-relative destinations.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from pathlib import PurePosixPath
import resource
import sys
import time


MANIFEST = "performance/manifests/mainline_precision_confirmation_v1.json"
DOMAIN = "InventorySupplyChain"
ARM = "legacy_role_after_init_point_only"
SEED = 2026120001
STREAM = "observation_1"


def export(project, output, source_commit, asset_prefix="../assets"):
    import numpy as np

    project, output = Path(project).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, wall = time.process_time(), time.perf_counter()
    receipt = dict(schema="mainline-small-preparation-v1", status="failed",
        kind="release_acceptance_preparation_only", automatic_retries=0,
        new_source_calls=0, new_source_fits=0, new_target_initial_calls=0,
        new_target_search_calls=0, new_verification_calls=0,
        posterior_restorations=0, model_files_opened=0, truth_used=False,
        source_commit=source_commit, input_files=[],
        started_utc=datetime.now(timezone.utc).isoformat())

    def read(path):
        path = Path(path)
        if not path.is_absolute():
            path = project / path
        receipt["input_files"].append(dict(path=str(path.relative_to(project)),
                                            bytes=path.stat().st_size))
        return json.loads(path.read_text())

    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    try:
        plan = read(MANIFEST)
        registered = next(task for task in plan["tasks"] if task["domain"] == DOMAIN)
        descriptor = next(stream for stream in plan["streams"] if stream["name"] == STREAM)
        if registered["geometry_task_seed"] != SEED or plan["streams"][0] != descriptor:
            raise ValueError("The first registered Inventory case/stream differs from preselection")
        seal_file = Path(plan["output_dir"]) / "execution_status.json"
        seal = read(seal_file)
        if seal["status"] != "sealed":
            raise ValueError("Stage85 execution status is not sealed")
        rows = [row for row in seal["cases"] if
                (row["domain"], row["seed"], row["stream"], row["arm"]) == (DOMAIN, SEED, STREAM, ARM)]
        if len(rows) != 1:
            raise ValueError("The preselected registered cell is not unique")
        cell = rows[0]
        if cell["old"]["input_file"] is None:
            raise ValueError("The preselected frozen terminal input is missing; do not choose another case")
        source = read(cell["search_manifest"])
        terminal = read(cell["terminal_manifest"])
        old_input = read(cell["old"]["input_file"])
        child = read(descriptor["manifest"])
        seeds = child["replicates"][0]["seed_by_task"][str(SEED)]
        observation_seed = int(seeds["observation_seed"])
        verification_seed = int(seeds["verification_seed"])
        if (source["observation_seed_by_task"][str(SEED)] != observation_seed
                or terminal["verification_seed_by_task"][str(SEED)] != verification_seed
                or old_input["verification_seed"] != verification_seed):
            raise ValueError("Frozen initialization and terminal stream registration disagree")
        if source["domain"] != DOMAIN or old_input["domain"] != DOMAIN or old_input["seed"] != SEED:
            raise ValueError("The saved input belongs to another registered geometry")
        geometry = source["geometry_by_seed"][str(SEED)]["problem_kwargs"]
        if geometry != registered["geometry"]["problem_kwargs"] or old_input["geometry"] != geometry:
            raise ValueError("The saved geometry differs from the registered task")
        if old_input["protocol"] != "same_policy_distinct_challenger":
            raise ValueError("Release acceptance must use the frozen old terminal")
        directory = Path(cell["source_case"]).parent
        initialization_file = directory / "initialization.json"
        neighbors_file = directory / "neighbors.npz"
        metadata_file = directory / "neighbors.json"
        initialization = read(initialization_file)
        metadata = read(metadata_file)
        if initialization["observation_seed"] != observation_seed or initialization["geometry"] != geometry:
            raise ValueError("The paid initialization comes from another target stream")
        initial = initialization["history"]
        if len(initial) != 10 or any(len(row["point"]) != 1000 or len(row["raw"]) != 2 for row in initial):
            raise ValueError("The paid initial10 raw schema differs")
        receipt["input_files"].append(dict(path=str(neighbors_file),
            bytes=(project / neighbors_file).stat().st_size))
        with np.load(project / neighbors_file, allow_pickle=False) as payload:
            points = payload["points"].tolist()
        if len(points) != 64 or metadata["count"] != 64 or any(len(point) != 1000 for point in points):
            raise ValueError("The registered shared64 physical bank differs")
        if (metadata["seed"] != SEED or metadata["radius"] != source["neighbor_radius"]
                or metadata["safe_margin_scale"] != source["neighbor_safe_margin_scale"]):
            raise ValueError("The shared64 metadata differs from the frozen generator recipe")
        assets = source["frozen_assets"]
        resolved_file = Path(assets["native_source_prior_file"]).parent / "native_resolved_task.json"
        task = deepcopy(read(resolved_file))
        if task["heldout"] != DOMAIN or task["args"]["d"] != 1000:
            raise ValueError("The frozen native resolved task belongs to another family/dimension")
        if (task["args"]["N"], task["args"]["n0"]) != (13, 10):
            raise ValueError("The frozen resolved search budget changed")
        if task["args"]["initial_design_points"] != [row["point"] for row in initial]:
            raise ValueError("The registered physical initialization differs from source design80")
        task["seed"], task["initial_design_seed"] = SEED, 80
        asset_directory = PurePosixPath(asset_prefix) / DOMAIN
        case = dict(schema="mainline-small-v1", kind="release_acceptance_only",
            domain=DOMAIN, geometry_task_seed=SEED, stream=STREAM,
            observation_seed=observation_seed, verification_seed=verification_seed,
            problem_kwargs=geometry, source_commit=source_commit,
            source_models_file=str(asset_directory / "source_models.pkl"),
            native_source_prior_file=str(asset_directory / "native_source_prior.pkl"),
            resolved_task_file="resolved_task.json", initial_rows_file="initial_rows.json",
            shared_neighbors_file="neighbors.json",
            preselection="First registered Stage85 Inventory geometry and first observation stream; no outcome screening",
            initialization_semantics="Reuse these ten already paid paired raw rows; condition the real AFTER model",
            acceptance_semantics="New release acceptance only: three fresh production search calls and independent old terminal; sealed scientific cohorts unchanged",
            provenance=dict(stage85_manifest=MANIFEST, execution_status=str(seal_file),
                source_case=cell["source_case"], search_manifest=cell["search_manifest"],
                terminal_manifest=cell["terminal_manifest"], old_input_file=cell["old"]["input_file"],
                source_models_original=assets["source_models_file"],
                native_prior_original=assets["native_source_prior_file"],
                resolved_task_original=str(resolved_file), initialization_original=str(initialization_file),
                neighbors_original=str(neighbors_file), neighbor_metadata_original=str(metadata_file)))
        save("initial_rows.json", initial)
        save("neighbors.json", dict(points=points, metadata=metadata))
        save("resolved_task.json", task)
        save("case.json", case)
        receipt.update(status="complete", selected_domain=DOMAIN, geometry_task_seed=SEED,
            selected_stream=STREAM, observation_seed=observation_seed, verification_seed=verification_seed,
            reused_initial_rows=10, reused_shared_neighbors=64,
            output_files=["case.json", "initial_rows.json", "neighbors.json", "resolved_task.json"],
            source_assets_copied=False, numpy_version=np.__version__)
        return receipt
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        receipt.update(cpu_seconds=time.process_time()-started, wall_seconds=time.perf_counter()-wall,
            peak_rss_bytes=int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
            finished_utc=datetime.now(timezone.utc).isoformat())
        save("preparation_receipt.json", receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "SC-OLH-KG")
    parser.add_argument("--output", type=Path, required=True,
                        help="New case directory; existing extraction attempts are retained")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--asset-prefix", default="../assets",
                        help="Source asset directory relative to case.json")
    args = parser.parse_args(argv)
    print(json.dumps(export(args.project_root, args.output, args.source_commit, args.asset_prefix)))


if __name__ == "__main__":
    main()
