"""Exact accepted-edge removal selection and vertex-cover dataset manifests."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import file_io, manifest
from .clustering import UnionFind


def _validated_review(reviewed_path: Path) -> tuple[list[dict[str, str]], str]:
    sidecar_path = reviewed_path.with_suffix(".json")
    if not reviewed_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("Finalized review and sidecar are required")
    sidecar = file_io.read_json(sidecar_path)
    snapshot = file_io.sha256_file(reviewed_path)
    if sidecar.get("reviewed_pairs_sha256") != snapshot:
        raise ValueError("Finalized review CSV does not match its sidecar")
    rows = file_io.read_csv_rows(reviewed_path)
    if not sidecar.get("complete") or any(
        row.get("decision") == manifest.DECISION_PENDING for row in rows
    ):
        raise ValueError("Minimum vertex cover requires a complete finalized review")
    return rows, snapshot


def _component_key(nodes: list[str], edges: list[tuple[str, str]]) -> str:
    payload = {"nodes": nodes, "edges": [list(edge) for edge in edges]}
    return hashlib.sha256(file_io.canonical_json_bytes(payload)).hexdigest()


def _solve_component(
    nodes: list[str],
    edges: list[tuple[str, str]],
    part_path: Path,
    *,
    workers: int,
    time_limit: float,
) -> dict[str, Any]:
    key = _component_key(nodes, edges)
    if part_path.is_file():
        part = file_io.read_json(part_path)
        selected = set(part.get("selected_image_ids", []))
        if (
            part.get("component_sha256") == key
            and part.get("status") == "OPTIMAL"
            and selected <= set(nodes)
            and all(a in selected or b in selected for a, b in edges)
        ):
            return part

    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError("select-removals requires the 'ortools' package") from exc

    model = cp_model.CpModel()
    variables = {image_id: model.new_bool_var(f"remove_{index}") for index, image_id in enumerate(nodes)}
    for a, b in edges:
        model.add(variables[a] + variables[b] >= 1)
    model.minimize(sum(variables.values()))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.solve(model)
    status_name = solver.status_name(status)
    if status_name != "OPTIMAL":
        raise RuntimeError(
            f"Vertex-cover component with {len(nodes)} images was not proven optimal "
            f"within {time_limit}s (status={status_name}, bound={solver.best_objective_bound})"
        )
    selected = sorted(image_id for image_id in nodes if solver.value(variables[image_id]))
    part = {
        "component_sha256": key,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "status": status_name,
        "objective": len(selected),
        "best_objective_bound": solver.best_objective_bound,
        "wall_time_seconds": solver.wall_time,
        "selected_image_ids": selected,
    }
    file_io.atomic_write_json(part_path, part)
    return part


def select_minimum_vertex_cover(
    images_path: Path,
    reviewed_path: Path,
    output: Path,
    *,
    workers: int = 16,
    time_limit: float = 900.0,
) -> dict[str, Any]:
    """Select a certified minimum set touching every accepted review edge."""
    if workers <= 0 or time_limit <= 0:
        raise ValueError("workers and time_limit must be positive")
    images = manifest.load_images(images_path)
    reviewed, snapshot = _validated_review(reviewed_path)
    accepted_rows = [row for row in reviewed if row.get("decision") == manifest.DECISION_ACCEPTED]
    accepted_edges: dict[tuple[str, str], dict[str, str]] = {}
    uf = UnionFind(images)
    for row in accepted_rows:
        a, b = row["image_a_id"], row["image_b_id"]
        if a not in images or b not in images:
            raise ValueError(f"Accepted pair {row['pair_id']} refers to an unknown image")
        edge = tuple(sorted((a, b)))
        if edge in accepted_edges:
            raise ValueError(f"Duplicate accepted endpoint pair: {edge}")
        accepted_edges[edge] = row
        uf.union(a, b)

    edge_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    node_groups: dict[str, set[str]] = defaultdict(set)
    for edge in sorted(accepted_edges):
        root = uf.find(edge[0])
        edge_groups[root].append(edge)
        node_groups[root].update(edge)
    components = sorted(
        ((sorted(node_groups[root]), edge_groups[root]) for root in edge_groups),
        key=lambda item: (len(item[0]), item[0]),
    )

    parts_dir = output.parent / f"{output.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    selected: set[str] = set()
    reports = []
    for nodes, edges in components:
        component_sha = _component_key(nodes, edges)
        part = _solve_component(
            nodes,
            edges,
            parts_dir / f"{component_sha}.json",
            workers=workers,
            time_limit=time_limit,
        )
        selected.update(part["selected_image_ids"])
        reports.append({key: value for key, value in part.items() if key != "selected_image_ids"})

    if any(a not in selected and b not in selected for a, b in accepted_edges):
        raise RuntimeError("Computed selection does not cover every accepted pair")
    adjacency: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    for (a, b), row in accepted_edges.items():
        adjacency[a].append((b, row))
        adjacency[b].append((a, row))

    rows = []
    for image_id in sorted(selected):
        witnesses = sorted(
            ((neighbor, row) for neighbor, row in adjacency[image_id] if neighbor not in selected),
            key=lambda item: (item[0], item[1]["pair_id"]),
        )
        if not witnesses:
            raise RuntimeError(
                f"Optimal cover image {image_id} has no retained accepted neighbor; solution is not minimal"
            )
        witness, supporting = witnesses[0]
        entry = images[image_id]
        rows.append(
            {
                "image_id": image_id,
                "shared_rel_path": entry.shared_rel_path,
                "sha256": entry.sha256,
                "witness_keeper_image_id": witness,
                "supporting_accepted_pair_id": supporting["pair_id"],
                "review_snapshot_sha256": snapshot,
            }
        )
    file_io.atomic_write_csv(output, manifest.MINIMUM_VERTEX_COVER_FIELDS, rows)
    raw_deletions_path = output.with_name("raw_files_to_delete.csv")
    raw_deletion_rows = [
        {
            "image_id": image_id,
            "source_name": images[image_id].source_name,
            "raw_relative_path": images[image_id].raw_relative_path,
            "shared_rel_path": images[image_id].shared_rel_path,
            "sha256": images[image_id].sha256,
        }
        for image_id in sorted(selected)
    ]
    file_io.atomic_write_csv(
        raw_deletions_path, manifest.RAW_FILES_TO_DELETE_FIELDS, raw_deletion_rows
    )
    raw_deletions_txt_path = output.with_name("raw_files_to_delete.txt")
    file_io.atomic_write_text(
        raw_deletions_txt_path,
        "".join(f"{images[image_id].source_name}/{images[image_id].raw_relative_path}\n" for image_id in sorted(selected)),
    )
    summary = {
        "strategy": "minimum_vertex_cover",
        "status": "OPTIMAL",
        "review_snapshot_sha256": snapshot,
        "accepted_edge_count": len(accepted_edges),
        "involved_image_count": len({item for edge in accepted_edges for item in edge}),
        "component_count": len(components),
        "selected_image_count": len(selected),
        "retained_image_count": len(images) - len(selected),
        "selection_sha256": file_io.sha256_file(output),
        "raw_files_to_delete_csv_sha256": file_io.sha256_file(raw_deletions_path),
        "raw_files_to_delete_txt_sha256": file_io.sha256_file(raw_deletions_txt_path),
        "all_edges_covered": True,
        "components": reports,
    }
    file_io.atomic_write_json(output.with_suffix(".json"), summary)
    return summary


def prepare_vertex_cover_export(
    selection_path: Path,
    manifest_path: Path,
    excluded_path: Path,
) -> dict[str, Any]:
    """Prepare one-class-per-retained-image manifests from a certified vertex cover."""
    work = file_io.work_dir()
    reviewed_path = work / "review" / "reviewed_pairs.csv"
    reviewed, review_snapshot = _validated_review(reviewed_path)
    summary_path = selection_path.with_suffix(".json")
    if not selection_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Minimum vertex-cover CSV and sidecar are required")
    selection_summary = file_io.read_json(summary_path)
    if selection_summary.get("status") != "OPTIMAL":
        raise ValueError("Minimum vertex cover is not certified optimal")
    if selection_summary.get("selection_sha256") != file_io.sha256_file(selection_path):
        raise ValueError("Minimum vertex-cover CSV does not match its sidecar")
    if selection_summary.get("review_snapshot_sha256") != review_snapshot:
        raise ValueError("Minimum vertex cover is stale relative to finalized review")

    images = manifest.load_images(work / "images.csv")
    selection_rows = file_io.read_csv_rows(selection_path)
    removed = {row["image_id"]: row for row in selection_rows}
    if len(removed) != len(selection_rows) or not set(removed) <= set(images):
        raise ValueError("Minimum vertex cover contains duplicate or unknown image IDs")
    for image_id, row in removed.items():
        entry = images[image_id]
        if (
            row.get("review_snapshot_sha256") != review_snapshot
            or row.get("shared_rel_path") != entry.shared_rel_path
            or row.get("sha256") != entry.sha256
        ):
            raise ValueError(f"Minimum vertex-cover provenance mismatch for {image_id}")
    accepted_by_id = {
        row["pair_id"]: row for row in reviewed if row.get("decision") == manifest.DECISION_ACCEPTED
    }
    if any(
        row["image_a_id"] not in removed and row["image_b_id"] not in removed
        for row in accepted_by_id.values()
    ):
        raise ValueError("Minimum vertex cover leaves an accepted pair entirely retained")
    for image_id, row in removed.items():
        pair = accepted_by_id.get(row.get("supporting_accepted_pair_id", ""))
        if pair is None or {pair["image_a_id"], pair["image_b_id"]} != {
            image_id,
            row.get("witness_keeper_image_id", ""),
        }:
            raise ValueError(f"Removed image {image_id} has invalid accepted witness evidence")
        if row["witness_keeper_image_id"] in removed:
            raise ValueError(f"Removed image {image_id} has a removed witness")
    retained_ids = sorted(set(images) - set(removed))
    class_key_by_image = {
        image_id: hashlib.sha256(f"mvc-retained\0{image_id}".encode()).hexdigest()
        for image_id in retained_ids
    }
    manifest_rows = []
    for class_id, image_id in enumerate(retained_ids, start=1):
        entry = images[image_id]
        file_id = f"dedup_{hashlib.sha256(image_id.encode()).hexdigest()[:24]}"
        suffix = Path(entry.shared_filename).suffix.lower() or ".jpg"
        manifest_rows.append(
            {
                "class_id": class_id,
                "class_key": class_key_by_image[image_id],
                "file_id": file_id,
                "source_name": entry.source_name,
                "shared_rel_path": entry.shared_rel_path,
                "output_rel_path": f"images/{file_id}{suffix}",
                "mime_type": entry.mime_type,
                "width": entry.width,
                "height": entry.height,
                "bytes_on_disk": entry.bytes_on_disk,
                "sha256": entry.sha256,
            }
        )
    excluded_rows = []
    for image_id in sorted(removed):
        row = removed[image_id]
        witness_id = row["witness_keeper_image_id"]
        if witness_id not in class_key_by_image:
            raise ValueError(f"Removed image {image_id} has a missing/removed witness {witness_id}")
        entry, witness = images[image_id], images[witness_id]
        excluded_rows.append(
            {
                "cluster_id": class_key_by_image[witness_id],
                "excluded_image_id": image_id,
                "excluded_shared_rel_path": entry.shared_rel_path,
                "excluded_sha256": entry.sha256,
                "keeper_image_id": witness_id,
                "keeper_shared_rel_path": witness.shared_rel_path,
                "supporting_accepted_pair_ids": row["supporting_accepted_pair_id"],
            }
        )
    file_io.atomic_write_csv(manifest_path, manifest.DATASET_MANIFEST_FIELDS, manifest_rows)
    file_io.atomic_write_csv(excluded_path, manifest.EXCLUDED_IMAGES_FIELDS, excluded_rows)
    export_summary = {
        "selection_strategy": "minimum_vertex_cover",
        "solver_status": selection_summary["status"],
        "review_snapshot_sha256": review_snapshot,
        "accepted_edge_count": selection_summary["accepted_edge_count"],
        "minimum_removal_count": len(excluded_rows),
        "retained_image_count": len(manifest_rows),
        "selection_sha256": file_io.sha256_file(selection_path),
        "dataset_manifest_sha256": file_io.sha256_file(manifest_path),
        "excluded_images_sha256": file_io.sha256_file(excluded_path),
    }
    file_io.atomic_write_json(manifest_path.with_suffix(".json"), export_summary)
    return export_summary
