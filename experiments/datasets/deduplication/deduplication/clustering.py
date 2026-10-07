"""Stages 8–9 — accepted-edge clustering, keeper selection, and export manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from . import file_io, manifest
from .config import Config


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        if a not in self.parent or b not in self.parent:
            raise ValueError(f"Reviewed pair refers to unknown image: {a}, {b}")
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def components(self) -> list[list[str]]:
        groups: dict[str, list[str]] = {}
        for item in sorted(self.parent):
            groups.setdefault(self.find(item), []).append(item)
        return sorted((sorted(group) for group in groups.values()), key=lambda group: group[0])


def make_cluster_id(image_ids: list[str]) -> str:
    return hashlib.sha256("\0".join(sorted(image_ids)).encode("utf-8")).hexdigest()


def _keeper_key(entry: manifest.ImageEntry, priority: tuple[str, ...]) -> tuple[Any, ...]:
    source_rank = priority.index(entry.source_name) if entry.source_name in priority else len(priority)
    return (
        source_rank,
        -(entry.width * entry.height),
        -min(entry.width, entry.height),
        -entry.bytes_on_disk,
        entry.image_id,
    )


def _load_overrides(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Keeper override file not found: {path}")
    overrides: dict[str, str] = {}
    for row in file_io.iter_csv_rows(path):
        cluster_id, keeper = row.get("cluster_id", ""), row.get("keeper_image_id", "")
        if not cluster_id or not keeper or cluster_id in overrides:
            raise ValueError(f"Malformed or duplicate keeper override for cluster {cluster_id!r}")
        overrides[cluster_id] = keeper
    return overrides


def build_clusters(
    config: Config,
    images_path: Path,
    reviewed_pairs_path: Path,
    output: Path,
    *,
    overrides_path: Path | None = None,
) -> dict[str, Any]:
    images = manifest.load_images(images_path)
    sidecar_path = reviewed_pairs_path.with_suffix(".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError("Finalized review sidecar is missing; run finalize-review")
    sidecar = file_io.read_json(sidecar_path)
    if sidecar.get("reviewed_pairs_sha256") != file_io.sha256_file(reviewed_pairs_path):
        raise ValueError("Finalized review CSV does not match its sidecar")
    if not sidecar.get("complete"):
        raise ValueError("Cannot cluster an incomplete finalized review")
    reviewed = file_io.read_csv_rows(reviewed_pairs_path)
    pending = [row["pair_id"] for row in reviewed if row.get("decision") == manifest.DECISION_PENDING]
    if pending:
        raise ValueError(f"Cannot cluster incomplete review: {len(pending)} pending pair(s)")
    uf = UnionFind(images)
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for row in reviewed:
        decision = row.get("decision")
        if decision == manifest.DECISION_ACCEPTED:
            uf.union(row["image_a_id"], row["image_b_id"])
            accepted.append(row)
        elif decision == manifest.DECISION_REJECTED:
            rejected.append(row)
        else:
            raise ValueError(f"Invalid finalized decision for {row.get('pair_id')}: {decision!r}")

    snapshot = file_io.sha256_file(reviewed_pairs_path)
    components = uf.components()
    component_by_image: dict[str, tuple[str, list[str]]] = {}
    for members in components:
        cluster_id = make_cluster_id(members)
        for image_id in members:
            component_by_image[image_id] = (cluster_id, members)

    conflicts: list[dict[str, Any]] = []
    for row in rejected:
        a, b = row["image_a_id"], row["image_b_id"]
        if a not in component_by_image or b not in component_by_image:
            raise ValueError(f"Rejected pair {row['pair_id']} refers to an unknown image")
        cluster_id, members = component_by_image[a]
        if component_by_image[b][0] == cluster_id:
            conflicts.append(
                {
                    "cluster_id": cluster_id,
                    "rejected_pair_id": row["pair_id"],
                    "image_a_id": a,
                    "image_b_id": b,
                    "component_size": len(members),
                }
            )

    overrides = _load_overrides(overrides_path)
    known_clusters = {make_cluster_id(members): set(members) for members in components}
    for cluster_id, keeper in overrides.items():
        if cluster_id not in known_clusters:
            raise ValueError(f"Keeper override refers to unknown cluster {cluster_id}")
        if keeper not in known_clusters[cluster_id]:
            raise ValueError(f"Override keeper {keeper} is not in cluster {cluster_id}")

    rows: list[dict[str, Any]] = []
    for members in components:
        cluster_id = make_cluster_id(members)
        recommended = min((images[item] for item in members), key=lambda entry: _keeper_key(entry, config.keeper_source_priority)).image_id
        keeper = overrides.get(cluster_id, recommended)
        reason = "explicit_override" if cluster_id in overrides else "source_priority_then_dimensions_size_id"
        for image_id in members:
            entry = images[image_id]
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "image_id": image_id,
                    "component_size": len(members),
                    "source_name": entry.source_name,
                    "shared_rel_path": entry.shared_rel_path,
                    "sha256": entry.sha256,
                    "recommended_keeper": int(image_id == recommended),
                    "is_keeper": int(image_id == keeper),
                    "keeper_reason": reason if image_id == keeper else "",
                    "review_snapshot_sha256": snapshot,
                }
            )
    rows.sort(key=lambda row: (row["cluster_id"], row["image_id"]))
    conflicts.sort(key=lambda row: (row["cluster_id"], row["rejected_pair_id"]))
    file_io.atomic_write_csv(output, manifest.CLUSTERS_FIELDS, rows)
    conflicts_path = output.with_name("cluster_conflicts.csv")
    file_io.atomic_write_csv(conflicts_path, manifest.CLUSTER_CONFLICTS_FIELDS, conflicts)
    return {
        "cluster_count": len(components),
        "non_singleton_cluster_count": sum(len(group) > 1 for group in components),
        "conflict_count": len(conflicts),
        "keeper_count": sum(int(row["is_keeper"]) for row in rows),
        "review_snapshot_sha256": snapshot,
    }


def prepare_dataset_export(clusters_path: Path, manifest_path: Path, excluded_path: Path) -> dict[str, Any]:
    work = file_io.work_dir()
    reviewed_path = work / "review" / "reviewed_pairs.csv"
    if not reviewed_path.is_file():
        raise FileNotFoundError("Finalized review is missing")
    reviewed_sidecar_path = reviewed_path.with_suffix(".json")
    if not reviewed_sidecar_path.is_file():
        raise FileNotFoundError("Finalized review sidecar is missing")
    reviewed_sidecar = file_io.read_json(reviewed_sidecar_path)
    if reviewed_sidecar.get("reviewed_pairs_sha256") != file_io.sha256_file(reviewed_path):
        raise ValueError("Finalized review CSV does not match its sidecar")
    reviewed = file_io.read_csv_rows(reviewed_path)
    if not reviewed_sidecar.get("complete") or any(row.get("decision") == manifest.DECISION_PENDING for row in reviewed):
        raise ValueError("Cannot prepare export while review is incomplete")
    snapshot = file_io.sha256_file(reviewed_path)
    conflicts_path = clusters_path.with_name("cluster_conflicts.csv")
    if not conflicts_path.is_file():
        raise FileNotFoundError("Cluster conflict report is missing; rebuild clusters")
    if file_io.read_csv_rows(conflicts_path):
        raise ValueError("Cannot prepare export: cluster conflicts must be resolved")

    images = manifest.load_images(work / "images.csv")
    cluster_rows = file_io.read_csv_rows(clusters_path)
    groups: dict[str, list[dict[str, str]]] = {}
    for row in cluster_rows:
        if row.get("review_snapshot_sha256") != snapshot:
            raise ValueError("Cluster output is stale relative to the finalized review")
        image_id = row["image_id"]
        entry = images.get(image_id)
        if entry is None or entry.sha256 != row["sha256"]:
            raise ValueError(f"Cluster image is missing or changed: {image_id}")
        path = entry.absolute_path()
        if not path.is_file() or file_io.sha256_file(path) != entry.sha256:
            raise ValueError(f"Shared image bytes changed: {image_id}")
        groups.setdefault(row["cluster_id"], []).append(row)
    clustered_ids = [row["image_id"] for row in cluster_rows]
    if len(clustered_ids) != len(set(clustered_ids)) or set(images) != set(clustered_ids):
        raise ValueError("Clusters do not contain exactly one row per current valid image")

    accepted_by_cluster: dict[str, list[str]] = {cluster_id: [] for cluster_id in groups}
    image_cluster = {row["image_id"]: cluster_id for cluster_id, rows in groups.items() for row in rows}
    for row in reviewed:
        if row["decision"] == manifest.DECISION_ACCEPTED:
            cluster_id = image_cluster.get(row["image_a_id"])
            if cluster_id and image_cluster.get(row["image_b_id"]) == cluster_id:
                accepted_by_cluster[cluster_id].append(row["pair_id"])

    output_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for class_id, cluster_id in enumerate(sorted(groups), start=1):
        rows = groups[cluster_id]
        keepers = [row for row in rows if row["is_keeper"] in {"1", "true", "True"}]
        if len(keepers) != 1:
            raise ValueError(f"Cluster {cluster_id} has {len(keepers)} keepers; expected exactly one")
        keeper_row = keepers[0]
        keeper = images[keeper_row["image_id"]]
        file_id = f"dedup_{hashlib.sha256(keeper.image_id.encode()).hexdigest()[:24]}"
        suffix = Path(keeper.shared_filename).suffix.lower() or ".jpg"
        output_rel_path = f"images/{file_id}{suffix}"
        output_rows.append(
            {
                "class_id": class_id,
                "class_key": cluster_id,
                "file_id": file_id,
                "source_name": keeper.source_name,
                "shared_rel_path": keeper.shared_rel_path,
                "output_rel_path": output_rel_path,
                "mime_type": keeper.mime_type,
                "width": keeper.width,
                "height": keeper.height,
                "bytes_on_disk": keeper.bytes_on_disk,
                "sha256": keeper.sha256,
            }
        )
        support = ";".join(sorted(accepted_by_cluster[cluster_id]))
        for row in rows:
            if row is keeper_row:
                continue
            entry = images[row["image_id"]]
            excluded_rows.append(
                {
                    "cluster_id": cluster_id,
                    "excluded_image_id": entry.image_id,
                    "excluded_shared_rel_path": entry.shared_rel_path,
                    "excluded_sha256": entry.sha256,
                    "keeper_image_id": keeper.image_id,
                    "keeper_shared_rel_path": keeper.shared_rel_path,
                    "supporting_accepted_pair_ids": support,
                }
            )
    file_io.atomic_write_csv(manifest_path, manifest.DATASET_MANIFEST_FIELDS, output_rows)
    file_io.atomic_write_csv(excluded_path, manifest.EXCLUDED_IMAGES_FIELDS, excluded_rows)
    return {
        "keeper_count": len(output_rows),
        "excluded_count": len(excluded_rows),
        "review_snapshot_sha256": snapshot,
        "dataset_manifest_sha256": file_io.sha256_file(manifest_path),
        "excluded_images_sha256": file_io.sha256_file(excluded_path),
    }
