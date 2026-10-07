"""CSV/JSON schema contracts, file identities, and manifest loaders.

Two related identities (TECHNICAL_PLAN 8):
  * ``image_id``: stable identity from ``source_name`` + normalized raw-relative path.
  * ``sha256``: identity of the copied shared image bytes.

A reviewed pair is specific to both endpoints' identities *and* their current bytes, so changing a
source file yields a new ``pair_id`` and cannot silently inherit an old human decision.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import file_io

# ---------------------------------------------------------------------------
# CSV schemas (explicit, ordered field lists)
# ---------------------------------------------------------------------------

SOURCE_FILES_FIELDS = [
    "image_id",
    "source_name",
    "raw_relative_path",
    "original_filename",
    "shared_filename",
    "shared_rel_path",
    "bytes_on_disk",
    "sha256",
]

IMAGES_FIELDS = [
    "image_id",
    "source_name",
    "raw_relative_path",
    "shared_filename",
    "shared_rel_path",
    "mime_type",
    "width",
    "height",
    "bytes_on_disk",
    "sha256",
]

INVALID_IMAGES_FIELDS = [
    "image_id",
    "source_name",
    "raw_relative_path",
    "shared_filename",
    "shared_rel_path",
    "bytes_on_disk",
    "sha256",
    "error",
]

ORPHANED_SHARED_IMAGES_FIELDS = ["shared_filename", "shared_rel_path", "bytes_on_disk", "sha256"]

CANDIDATES_FIELDS = [
    "pair_id",
    "image_a_id",
    "image_b_id",
    "cosine_similarity",
    "rank_a_to_b",
    "rank_b_to_a",
    "retrieved_from_a",
    "retrieved_from_b",
    "candidate_reasons",
    "dino_threshold_match",
    "dino_window_match",
    "exact_sha256_match",
    "model_fingerprint",
    "retrieval_config_sha256",
]

VERIFICATION_FIELDS = [
    "pair_id",
    "status",
    "lightglue_prediction",
    "lightglue_confidence",
    "match_count",
    "classifier_sha256",
    "superpoint_fingerprint",
    "error",
]

REVIEW_PAIRS_FIELDS = [
    "pair_id",
    "image_a_id",
    "image_a_shared_rel_path",
    "image_a_source",
    "image_a_raw_relative_path",
    "image_a_width",
    "image_a_height",
    "image_b_id",
    "image_b_shared_rel_path",
    "image_b_source",
    "image_b_raw_relative_path",
    "image_b_width",
    "image_b_height",
    "cosine_similarity",
    "candidate_reasons",
    "dino_threshold_match",
    "dino_window_match",
    "exact_sha256_match",
    "lightglue_prediction",
    "lightglue_confidence",
    "match_count",
]

REVIEW_DECISIONS_FIELDS = ["pair_id", "decision", "reviewed_at", "reviewer"]

RAW_FILES_TO_DELETE_FIELDS = [
    "image_id",
    "source_name",
    "raw_relative_path",
    "shared_rel_path",
    "sha256",
]

MINIMUM_VERTEX_COVER_FIELDS = [
    "image_id",
    "shared_rel_path",
    "sha256",
    "witness_keeper_image_id",
    "supporting_accepted_pair_id",
    "review_snapshot_sha256",
]

REVIEWED_PAIRS_FIELDS = REVIEW_PAIRS_FIELDS + ["decision", "reviewed_at", "reviewer"]

CLUSTERS_FIELDS = [
    "cluster_id",
    "image_id",
    "component_size",
    "source_name",
    "shared_rel_path",
    "sha256",
    "recommended_keeper",
    "is_keeper",
    "keeper_reason",
    "review_snapshot_sha256",
]

CLUSTER_CONFLICTS_FIELDS = [
    "cluster_id",
    "rejected_pair_id",
    "image_a_id",
    "image_b_id",
    "component_size",
]

KEEPER_OVERRIDES_FIELDS = ["cluster_id", "keeper_image_id"]

DATASET_MANIFEST_FIELDS = [
    "class_id",
    "class_key",
    "file_id",
    "source_name",
    "shared_rel_path",
    "output_rel_path",
    "mime_type",
    "width",
    "height",
    "bytes_on_disk",
    "sha256",
]

EXCLUDED_IMAGES_FIELDS = [
    "cluster_id",
    "excluded_image_id",
    "excluded_shared_rel_path",
    "excluded_sha256",
    "keeper_image_id",
    "keeper_shared_rel_path",
    "supporting_accepted_pair_ids",
]

DECISION_ACCEPTED = "accepted"
DECISION_REJECTED = "rejected"
DECISION_PENDING = "pending"
VALID_DECISIONS = frozenset({DECISION_ACCEPTED, DECISION_REJECTED, DECISION_PENDING})


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------


def make_image_id(source_name: str, raw_relative_path: str) -> str:
    """Stable, byte-independent identity from source + normalized relative path."""
    normalized = raw_relative_path.replace("\\", "/").strip("/")
    return f"{source_name}/{normalized}"


def make_pair_id(image_a_id: str, image_a_sha256: str, image_b_id: str, image_b_sha256: str) -> str:
    """Content-specific pair id. Endpoints must already be ordered (a_id < b_id)."""
    payload = "\0".join([image_a_id, image_a_sha256, image_b_id, image_b_sha256])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def order_endpoints(id1: str, sha1: str, id2: str, sha2: str) -> tuple[str, str, str, str]:
    """Return endpoints ordered so the first image_id sorts before the second."""
    if id1 < id2:
        return id1, sha1, id2, sha2
    if id2 < id1:
        return id2, sha2, id1, sha1
    raise ValueError(f"Self-pair is not allowed: {id1}")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageEntry:
    image_id: str
    source_name: str
    raw_relative_path: str
    shared_filename: str
    shared_rel_path: str
    mime_type: str
    width: int
    height: int
    bytes_on_disk: int
    sha256: str

    def absolute_path(self) -> Path:
        return file_io.resolve_from_project(self.shared_rel_path)


def load_images(images_csv: Path | str) -> dict[str, ImageEntry]:
    """Load ``images.csv`` into an ordered dict keyed by ``image_id``."""
    entries: dict[str, ImageEntry] = {}
    for row in file_io.iter_csv_rows(images_csv):
        entry = ImageEntry(
            image_id=row["image_id"],
            source_name=row["source_name"],
            raw_relative_path=row["raw_relative_path"],
            shared_filename=row["shared_filename"],
            shared_rel_path=row["shared_rel_path"],
            mime_type=row.get("mime_type", ""),
            width=int(row["width"] or 0),
            height=int(row["height"] or 0),
            bytes_on_disk=int(row["bytes_on_disk"] or 0),
            sha256=row["sha256"],
        )
        entries[entry.image_id] = entry
    return entries


def ordered_image_ids(images: dict[str, ImageEntry]) -> list[str]:
    return sorted(images)


def latest_decisions(decisions_csv: Path | str) -> dict[str, str]:
    """Replay the append-only decision log; last row per pair wins."""
    state: dict[str, str] = {}
    if not Path(decisions_csv).exists():
        return state
    for row in file_io.iter_csv_rows(decisions_csv):
        pair_id = row.get("pair_id", "")
        decision = row.get("decision", "")
        if pair_id:
            state[pair_id] = decision
    return state


def validate_decision(decision: str) -> None:
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Invalid decision {decision!r}; expected one of {sorted(VALID_DECISIONS)}")
