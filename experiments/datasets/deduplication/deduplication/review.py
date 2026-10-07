"""Stages 5–7 — review handover, minimal Flask UI, and decision finalization."""

from __future__ import annotations

import csv
import hashlib
import os
import threading
from pathlib import Path
from typing import Any

from . import file_io, manifest

INSTRUCTION = (
    "Accept only if these two files should represent one image identity in the synthetic-data "
    "source corpus, such that retaining both would constitute duplicate dataset content. Reject "
    "visually similar but distinct artworks, copies, studies, or separate works."
)


def _is_positive(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "positive"}


def prepare_review(output: Path, *, include_negatives: bool = False) -> dict[str, Any]:
    work = file_io.work_dir()
    images = manifest.load_images(work / "images.csv")
    candidates_path = work / "candidates" / "candidates.csv"
    verification_path = work / "verification" / "verification.csv"
    verification_summary = file_io.read_json(work / "verification" / "summary.json")
    if verification_summary.get("candidates_sha256") != file_io.sha256_file(candidates_path):
        raise ValueError("Verification results are stale relative to candidates.csv")
    if verification_summary.get("verification_sha256") != file_io.sha256_file(verification_path):
        raise ValueError("verification.csv hash does not match its summary")
    # Exact byte matches are review-worthy even if the geometric classifier is negative. Keep
    # those pair ids alongside classifier positives, while still avoiding millions of ordinary
    # negative rows in memory.
    exact_pair_ids = {
        row["pair_id"]
        for row in file_io.iter_csv_rows(candidates_path)
        if row.get("exact_sha256_match", "0") == "1"
    }
    verification: dict[str, dict[str, str]] = {}
    for verified in file_io.iter_csv_rows(verification_path):
        if verified.get("status") != "ok":
            continue
        if (
            include_negatives
            or verified.get("pair_id") in exact_pair_ids
            or _is_positive(verified.get("lightglue_prediction", ""))
        ):
            verification[verified["pair_id"]] = verified
    rows: list[dict[str, Any]] = []
    found: set[str] = set()
    for candidate in file_io.iter_csv_rows(candidates_path):
        pair_id = candidate["pair_id"]
        verified = verification.get(pair_id)
        if verified is None:
            continue
        found.add(pair_id)
        a = images.get(candidate["image_a_id"])
        b = images.get(candidate["image_b_id"])
        if a is None or b is None:
            raise ValueError(f"Candidate {pair_id} refers to an unknown image")
        rows.append(
            {
                "pair_id": pair_id,
                "image_a_id": a.image_id,
                "image_a_shared_rel_path": a.shared_rel_path,
                "image_a_source": a.source_name,
                "image_a_raw_relative_path": a.raw_relative_path,
                "image_a_width": a.width,
                "image_a_height": a.height,
                "image_b_id": b.image_id,
                "image_b_shared_rel_path": b.shared_rel_path,
                "image_b_source": b.source_name,
                "image_b_raw_relative_path": b.raw_relative_path,
                "image_b_width": b.width,
                "image_b_height": b.height,
                "cosine_similarity": candidate["cosine_similarity"],
                "candidate_reasons": candidate.get("candidate_reasons", "dino_threshold"),
                "dino_threshold_match": candidate.get("dino_threshold_match", "1"),
                "dino_window_match": candidate.get("dino_window_match", "0"),
                "exact_sha256_match": candidate.get("exact_sha256_match", "0"),
                "lightglue_prediction": verified["lightglue_prediction"],
                "lightglue_confidence": verified["lightglue_confidence"],
                "match_count": verified["match_count"],
            }
        )
    missing = sorted(set(verification) - found)
    if missing:
        raise ValueError(f"Verification contains {len(missing)} review outcomes absent from candidates.csv")
    rows.sort(key=lambda row: (-float(row["cosine_similarity"]), row["pair_id"]))
    file_io.atomic_write_csv(output, manifest.REVIEW_PAIRS_FIELDS, rows)
    return {"review_pair_count": len(rows), "review_pairs_sha256": file_io.sha256_file(output)}


def load_review_pairs(path: Path | str) -> list[dict[str, str]]:
    rows = file_io.read_csv_rows(path)
    ids: set[str] = set()
    for row in rows:
        pair_id = row.get("pair_id", "")
        if not pair_id or pair_id in ids:
            raise ValueError(f"Missing or duplicate pair_id in review queue: {pair_id!r}")
        ids.add(pair_id)
    return rows


def replay_decisions(path: Path | str, active_pair_ids: set[str] | None = None) -> dict[str, dict[str, str]]:
    decision_path = Path(path)
    if not decision_path.exists():
        return {}
    # csv.DictReader silently accepts a truncated final row, so validate every field explicitly.
    state: dict[str, dict[str, str]] = {}
    with decision_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != manifest.REVIEW_DECISIONS_FIELDS:
            raise ValueError(f"Malformed decision CSV header in {decision_path}")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(field) is None for field in manifest.REVIEW_DECISIONS_FIELDS):
                raise ValueError(f"Interrupted or malformed decision row at line {line_number}")
            pair_id = row["pair_id"]
            manifest.validate_decision(row["decision"])
            if not pair_id or not row["reviewed_at"]:
                raise ValueError(f"Incomplete decision row at line {line_number}")
            if active_pair_ids is not None and pair_id not in active_pair_ids:
                raise ValueError(f"Decision at line {line_number} refers to unknown pair {pair_id}")
            state[pair_id] = dict(row)
    return state


def append_decision(path: Path, pair_id: str, decision: str, reviewer: str) -> dict[str, str]:
    manifest.validate_decision(decision)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "pair_id": pair_id,
        "decision": decision,
        "reviewed_at": file_io.utc_now_iso(),
        "reviewer": reviewer,
    }
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest.REVIEW_DECISIONS_FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    return row


def _safe_image_path(row: dict[str, str], side: str) -> Path:
    if side not in {"a", "b"}:
        raise ValueError("Image side must be a or b")
    resolved = file_io.resolve_from_project(row[f"image_{side}_shared_rel_path"])
    allowed = (file_io.work_dir() / "images").resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("Review image path escapes work/images") from exc
    return resolved


def _thumbnail_path(
    source: Path, cache_dir: Path, max_size: tuple[int, int] = (1600, 1200)
) -> Path:
    """Return a bounded JPEG preview, cached by source identity, metadata, and size."""
    from PIL import Image, ImageOps

    stat = source.stat()
    cache_key = (
        f"{source.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}\0{max_size[0]}x{max_size[1]}"
    ).encode()
    target = cache_dir / f"{hashlib.sha256(cache_key).hexdigest()}.jpg"
    if target.is_file():
        return target

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with Image.open(source) as opened:
            preview = ImageOps.exif_transpose(opened).convert("RGB")
            preview.thumbnail(max_size, Image.Resampling.LANCZOS)
            preview.save(temporary, format="JPEG", quality=85)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _review_components(pairs: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build deterministic connected components, largest first, from all review edges."""
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            if left_root > right_root:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root

    for pair in pairs:
        union(pair["image_a_id"], pair["image_b_id"])

    pair_groups: dict[str, list[dict[str, str]]] = {}
    image_groups: dict[str, set[str]] = {}
    for pair in pairs:
        root = find(pair["image_a_id"])
        pair_groups.setdefault(root, []).append(pair)
        image_groups.setdefault(root, set()).update((pair["image_a_id"], pair["image_b_id"]))

    components = []
    for root, component_pairs in pair_groups.items():
        image_ids = sorted(image_groups[root])
        cluster_id = hashlib.sha256("\0".join(image_ids).encode()).hexdigest()
        components.append(
            {
                "cluster_id": cluster_id,
                "image_ids": image_ids,
                "pairs": sorted(component_pairs, key=lambda row: row["pair_id"]),
                "size": len(image_ids),
                "edge_count": len(component_pairs),
            }
        )
    return sorted(components, key=lambda item: (-item["size"], -item["edge_count"], item["cluster_id"]))


def _component_images(pairs: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    images: dict[str, dict[str, str]] = {}
    for pair in pairs:
        for side in ("a", "b"):
            image_id = pair[f"image_{side}_id"]
            row = {
                "image_id": image_id,
                "shared_rel_path": pair[f"image_{side}_shared_rel_path"],
                "source": pair[f"image_{side}_source"],
                "raw_relative_path": pair[f"image_{side}_raw_relative_path"],
                "width": pair[f"image_{side}_width"],
                "height": pair[f"image_{side}_height"],
            }
            previous = images.setdefault(image_id, row)
            if previous != row:
                raise ValueError(f"Inconsistent review metadata for image {image_id}")
    return images


def append_cluster_decision(
    path: Path, pair_ids: list[str], decision: str, reviewer: str
) -> dict[str, dict[str, str]]:
    """Append one ordinary decision event per cluster edge in one flushed batch."""
    manifest.validate_decision(decision)
    path.parent.mkdir(parents=True, exist_ok=True)
    reviewed_at = file_io.utc_now_iso()
    events = {
        pair_id: {
            "pair_id": pair_id,
            "decision": decision,
            "reviewed_at": reviewed_at,
            "reviewer": reviewer,
        }
        for pair_id in pair_ids
    }
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest.REVIEW_DECISIONS_FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerows(events.values())
        handle.flush()
        os.fsync(handle.fileno())
    return events


def resolve_pending(
    pairs_path: Path, decisions_path: Path, decision: str, reviewer: str
) -> dict[str, Any]:
    """Resolve every currently pending pair through append-only ordinary decision events."""
    if decision not in {manifest.DECISION_ACCEPTED, manifest.DECISION_REJECTED}:
        raise ValueError("Pending pairs can only be bulk-resolved as accepted or rejected")
    pairs = load_review_pairs(pairs_path)
    active = {row["pair_id"] for row in pairs}
    state = replay_decisions(decisions_path, active)
    pending = [
        row["pair_id"]
        for row in pairs
        if state.get(row["pair_id"], {}).get("decision", manifest.DECISION_PENDING)
        == manifest.DECISION_PENDING
    ]
    if pending:
        append_cluster_decision(decisions_path, pending, decision, reviewer)
    final_state = replay_decisions(decisions_path, active)
    counts = {item: 0 for item in manifest.VALID_DECISIONS}
    for pair_id in active:
        counts[final_state.get(pair_id, {}).get("decision", manifest.DECISION_PENDING)] += 1
    return {
        "resolved_count": len(pending),
        "resolved_as": decision,
        "counts": counts,
        "review_pairs_sha256": file_io.sha256_file(pairs_path),
        "review_decisions_sha256": file_io.sha256_file(decisions_path),
    }


def create_app(pairs_path: Path, decisions_path: Path, reviewer: str = "local"):
    from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for

    pairs = load_review_pairs(pairs_path)
    pair_ids = {row["pair_id"] for row in pairs}
    components = _review_components(pairs)
    component_by_id = {item["cluster_id"]: item for item in components}
    positions = {item["cluster_id"]: index for index, item in enumerate(components)}
    pair_list = [row["pair_id"] for row in pairs]
    pair_by_id = {row["pair_id"]: row for row in pairs}
    pair_positions = {pair_id: index for index, pair_id in enumerate(pair_list)}
    images = _component_images(pairs)
    state = replay_decisions(decisions_path, pair_ids)
    decision_lock = threading.Lock()
    thumbnail_dir = decisions_path.parent / "thumbnails" / "cluster_gallery"
    app = Flask(__name__)

    def counts(component: dict[str, Any]) -> dict[str, int]:
        result = {decision: 0 for decision in manifest.VALID_DECISIONS}
        for pair in component["pairs"]:
            decision = state.get(pair["pair_id"], {}).get("decision", manifest.DECISION_PENDING)
            result[decision] += 1
        return result

    def resolved(component: dict[str, Any]) -> bool:
        component_counts = counts(component)
        return (
            component_counts[manifest.DECISION_ACCEPTED] == component["edge_count"]
            or component_counts[manifest.DECISION_REJECTED] == component["edge_count"]
        )

    def pair_is_pending(pair_id: str) -> bool:
        return state.get(pair_id, {}).get("decision", manifest.DECISION_PENDING) == manifest.DECISION_PENDING

    def destination_after_pair(pair_id: str) -> str:
        start = pair_positions[pair_id]
        for offset in range(1, len(pair_list) + 1):
            candidate = pair_list[(start + offset) % len(pair_list)]
            if pair_is_pending(candidate):
                return url_for("show_pair", pair_id=candidate)
        return url_for("pair_index")

    @app.get("/")
    def index():
        first = next((item for item in components if not resolved(item)), None)
        if first is not None:
            return redirect(url_for("show_cluster", cluster_id=first["cluster_id"]))
        return render_template_string(
            "<h1>Cluster review complete</h1><p>{{ n }} / {{ n }} clusters reviewed.</p>",
            n=len(components),
        )

    @app.get("/cluster/<cluster_id>")
    def show_cluster(cluster_id: str):
        component = component_by_id.get(cluster_id)
        if component is None:
            abort(404)
        index = positions[cluster_id]
        incident: dict[str, list[str]] = {image_id: [] for image_id in component["image_ids"]}
        for pair in component["pairs"]:
            incident[pair["image_a_id"]].append(pair["pair_id"])
            incident[pair["image_b_id"]].append(pair["pair_id"])
        gallery = []
        for image_index, image_id in enumerate(component["image_ids"]):
            pair_ids_for_image = incident[image_id]
            image_counts = {decision: 0 for decision in manifest.VALID_DECISIONS}
            for pair_id in pair_ids_for_image:
                decision = state.get(pair_id, {}).get("decision", manifest.DECISION_PENDING)
                image_counts[decision] += 1
            gallery.append(
                {
                    **images[image_id],
                    "index": image_index,
                    "edge_count": len(pair_ids_for_image),
                    "counts": image_counts,
                }
            )
        return render_template_string(
            _CLUSTER_PAGE,
            cluster=component,
            gallery=gallery,
            counts=counts(component),
            reviewed=sum(resolved(item) for item in components),
            position=index + 1,
            total=len(components),
            previous_id=components[index - 1]["cluster_id"] if index else None,
            next_id=components[index + 1]["cluster_id"] if index + 1 < len(components) else None,
        )

    @app.get("/cluster/<cluster_id>/image/<int:image_index>")
    def cluster_image(cluster_id: str, image_index: int):
        component = component_by_id.get(cluster_id)
        if component is None or not 0 <= image_index < component["size"]:
            abort(404)
        image = images[component["image_ids"][image_index]]
        try:
            path = file_io.resolve_from_project(image["shared_rel_path"])
            path.relative_to((file_io.work_dir() / "images").resolve())
        except ValueError:
            abort(400)
        if not path.is_file():
            abort(404)
        if request.args.get("original") == "1":
            return send_file(path, conditional=True, max_age=86400)
        try:
            preview = _thumbnail_path(path, thumbnail_dir, (420, 420))
        except (OSError, ValueError):
            abort(404)
        return send_file(preview, mimetype="image/jpeg", conditional=True, max_age=31536000)

    @app.post("/cluster/<cluster_id>/image/<int:image_index>/decision")
    def decide_cluster_image(cluster_id: str, image_index: int):
        component = component_by_id.get(cluster_id)
        if component is None or not 0 <= image_index < component["size"]:
            abort(404)
        decision = request.form.get("decision", "")
        if decision not in {manifest.DECISION_ACCEPTED, manifest.DECISION_REJECTED}:
            abort(400)
        image_id = component["image_ids"][image_index]
        incident_pair_ids = [
            pair["pair_id"]
            for pair in component["pairs"]
            if image_id in {pair["image_a_id"], pair["image_b_id"]}
        ]
        with decision_lock:
            state.update(
                append_cluster_decision(decisions_path, incident_pair_ids, decision, reviewer)
            )
        return redirect(url_for("show_cluster", cluster_id=cluster_id) + f"#image-{image_index}")

    @app.post("/cluster/<cluster_id>/decision")
    def decide_cluster(cluster_id: str):
        component = component_by_id.get(cluster_id)
        if component is None:
            abort(404)
        decision = request.form.get("decision", "")
        if decision not in {
            manifest.DECISION_ACCEPTED,
            manifest.DECISION_REJECTED,
            manifest.DECISION_PENDING,
        }:
            abort(400)
        with decision_lock:
            events = append_cluster_decision(
                decisions_path,
                [pair["pair_id"] for pair in component["pairs"]],
                decision,
                reviewer,
            )
            state.update(events)
        return redirect(url_for("show_cluster", cluster_id=cluster_id))

    @app.get("/pairs")
    def pair_index():
        first = next((pair_id for pair_id in pair_list if pair_is_pending(pair_id)), None)
        if first is not None:
            return redirect(url_for("show_pair", pair_id=first))
        return render_template_string(
            "<h1>Pair review complete</h1><p>{{ n }} / {{ n }} pairs reviewed.</p>"
            "<p><a href='{{ url_for(\"index\") }}'>Cluster review</a></p>",
            n=len(pair_list),
        )

    @app.get("/pair/<pair_id>")
    def show_pair(pair_id: str):
        row = pair_by_id.get(pair_id)
        if row is None:
            abort(404)
        index = pair_positions[pair_id]
        return render_template_string(
            _PAIR_PAGE,
            row=row,
            instruction=INSTRUCTION,
            reviewed=sum(not pair_is_pending(item) for item in pair_list),
            total=len(pair_list),
            current=state.get(pair_id, {}).get("decision", manifest.DECISION_PENDING),
            previous_id=pair_list[index - 1] if index else None,
            next_id=pair_list[index + 1] if index + 1 < len(pair_list) else None,
        )

    @app.get("/pair/<pair_id>/image/<side>")
    def pair_image(pair_id: str, side: str):
        row = pair_by_id.get(pair_id)
        if row is None:
            abort(404)
        try:
            path = _safe_image_path(row, side)
        except ValueError:
            abort(400)
        if not path.is_file():
            abort(404)
        if request.args.get("original") == "1":
            return send_file(path, conditional=True, max_age=86400)
        try:
            preview = _thumbnail_path(path, decisions_path.parent / "thumbnails" / "pair", (1600, 1200))
        except (OSError, ValueError):
            abort(404)
        return send_file(preview, mimetype="image/jpeg", conditional=True, max_age=31536000)

    @app.post("/pair/<pair_id>/decision")
    def decide_pair(pair_id: str):
        if pair_id not in pair_by_id:
            abort(404)
        decision = request.form.get("decision", "")
        try:
            manifest.validate_decision(decision)
        except ValueError:
            abort(400)
        with decision_lock:
            state[pair_id] = append_decision(decisions_path, pair_id, decision, reviewer)
        return redirect(destination_after_pair(pair_id))

    return app


_CLUSTER_PAGE = r"""
<!doctype html><html><head><meta charset="utf-8"><title>Cluster review</title>
<style>
body{font:15px sans-serif;margin:1rem;background:#f3f3f3;color:#222}.top{position:sticky;top:0;z-index:2;background:#fff;padding:.8rem;box-shadow:0 2px 8px #999}.summary,.actions{text-align:center}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:.7rem;margin-top:1rem}.card{background:#fff;padding:.5rem;overflow:hidden;scroll-margin-top:9rem}.card img{display:block;width:100%;height:240px;object-fit:contain;background:#222}.meta{overflow-wrap:anywhere;font-size:.82rem}.image-actions{display:flex;margin-top:.4rem}.image-actions form{display:flex;width:100%;gap:.25rem}.image-actions button{flex:1;font-size:.78rem;padding:.4rem .2rem;margin:0}button,a{font-size:1rem;padding:.65rem 1rem;margin:.2rem}.accept{background:#267d45;color:#fff}.reject{background:#a93636;color:#fff}.nav{color:#174ea6}
</style></head><body>
<div class="top"><div class="summary"><strong>Cluster {{ position }} / {{ total }}</strong> · {{ reviewed }} resolved<br>
<strong>{{ cluster.size }} images</strong> · {{ cluster.edge_count }} proposal edges · accepted {{ counts.accepted }} · rejected {{ counts.rejected }} · pending {{ counts.pending }}</div>
<div class="actions"><form method="post" action="{{ url_for('decide_cluster',cluster_id=cluster.cluster_id) }}">
<button id="accept" class="accept" name="decision" value="accepted" onclick="return confirm('Accept all {{ cluster.edge_count }} edges? This will keep one image and remove the other {{ cluster.size - 1 }} images.')">Accept all — keep one (A)</button>
<button id="reject" class="reject" name="decision" value="rejected" onclick="return confirm('Reject all {{ cluster.edge_count }} edges and keep all {{ cluster.size }} images?')">Reject all — keep all (R)</button>
<button name="decision" value="pending" onclick="return confirm('Reset all {{ cluster.edge_count }} decisions in this cluster to pending?')">Reset all decisions</button></form>
<a class="nav" href="{{ url_for('pair_index') }}">Single-pair review</a>
{% if previous_id %}<a id="previous" class="nav" href="{{ url_for('show_cluster',cluster_id=previous_id) }}">← Previous</a>{% endif %}{% if next_id %}<a id="next" class="nav" href="{{ url_for('show_cluster',cluster_id=next_id) }}">Next →</a>{% endif %}</div></div>
<div class="gallery">{% for image in gallery %}<div class="card" id="image-{{ image.index }}"><a target="_blank" href="{{ url_for('cluster_image',cluster_id=cluster.cluster_id,image_index=image.index,original=1) }}"><img loading="lazy" src="{{ url_for('cluster_image',cluster_id=cluster.cluster_id,image_index=image.index) }}"></a><div class="meta"><strong>{{ image.source }}</strong> · {{ image.width }}×{{ image.height }}<br>{{ image.raw_relative_path }}<br><strong>{{ image.edge_count }} pairs</strong> · A {{ image.counts.accepted }} · R {{ image.counts.rejected }} · P {{ image.counts.pending }}</div><div class="image-actions"><form method="post" action="{{ url_for('decide_cluster_image',cluster_id=cluster.cluster_id,image_index=image.index) }}"><button class="accept" name="decision" value="accepted" onclick="return confirm('Accept all {{ image.edge_count }} pairs involving this image?')">Accept its pairs</button><button class="reject" name="decision" value="rejected" onclick="return confirm('Reject all {{ image.edge_count }} pairs involving this image?')">Reject its pairs</button></form></div></div>{% endfor %}</div>
<script>addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const k=e.key.toLowerCase();if(k==='a')document.querySelector('#accept').click();if(k==='r')document.querySelector('#reject').click();if((k==='p'||e.key==='ArrowLeft')&&document.querySelector('#previous'))document.querySelector('#previous').click();if((k==='n'||e.key==='ArrowRight')&&document.querySelector('#next'))document.querySelector('#next').click()})</script></body></html>
"""


_PAIR_PAGE = r"""
<!doctype html><html><head><meta charset="utf-8"><title>Single-pair review</title>
<style>body{font:16px sans-serif;margin:1rem;background:#f5f5f5}.notice{max-width:1000px;margin:auto;padding:.8rem;background:#fff3cd}.pair{display:flex;gap:1rem;margin:1rem 0}.card{flex:1;background:white;padding:1rem}.card img{width:100%;height:65vh;object-fit:contain;background:#222}.actions{text-align:center}button,a{font-size:1rem;padding:.7rem 1rem;margin:.2rem}.accept{background:#2d8a4e;color:white}.reject{background:#a93636;color:white}</style>
</head><body><p class="notice"><strong>Instruction:</strong> {{ instruction }}</p>
<p style="text-align:center">{{ reviewed }} / {{ total }} reviewed · current: <strong>{{ current }}</strong> · <a href="{{ url_for('index') }}">Cluster review</a></p>
<div class="pair"><div class="card"><a target="_blank" href="{{ url_for('pair_image',pair_id=row.pair_id,side='a',original=1) }}"><img src="{{ url_for('pair_image',pair_id=row.pair_id,side='a') }}"></a><p>{{ row.image_a_source }} · {{ row.image_a_width }}×{{ row.image_a_height }}<br>{{ row.image_a_raw_relative_path }}</p></div>
<div class="card"><a target="_blank" href="{{ url_for('pair_image',pair_id=row.pair_id,side='b',original=1) }}"><img src="{{ url_for('pair_image',pair_id=row.pair_id,side='b') }}"></a><p>{{ row.image_b_source }} · {{ row.image_b_width }}×{{ row.image_b_height }}<br>{{ row.image_b_raw_relative_path }}</p></div></div>
<p style="text-align:center">DINO {{ row.cosine_similarity }} · reasons {{ row.candidate_reasons }} · exact SHA {{ row.exact_sha256_match }} · LightGlue {{ row.lightglue_confidence }} · matches {{ row.match_count }}</p>
<div class="actions"><form method="post" action="{{ url_for('decide_pair',pair_id=row.pair_id) }}"><button id="accept" class="accept" name="decision" value="accepted">Accept duplicate (A)</button><button id="reject" class="reject" name="decision" value="rejected">Reject duplicate (R)</button><button name="decision" value="pending">Reset</button></form>
{% if previous_id %}<a id="previous" href="{{ url_for('show_pair',pair_id=previous_id) }}">← Previous</a>{% endif %}{% if next_id %}<a id="next" href="{{ url_for('show_pair',pair_id=next_id) }}">Next →</a>{% endif %}</div>
<script>addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key.toLowerCase()==='a')document.querySelector('#accept').click();if(e.key.toLowerCase()==='r')document.querySelector('#reject').click();if(e.key==='ArrowLeft'&&document.querySelector('#previous'))document.querySelector('#previous').click();if(e.key==='ArrowRight'&&document.querySelector('#next'))document.querySelector('#next').click()})</script></body></html>
"""


def finalize_review(pairs_path: Path, decisions_path: Path, output: Path, *, require_complete: bool) -> dict[str, Any]:
    pairs = load_review_pairs(pairs_path)
    active = {row["pair_id"] for row in pairs}
    state = replay_decisions(decisions_path, active)
    rows: list[dict[str, str]] = []
    pending: list[str] = []
    counts = {decision: 0 for decision in manifest.VALID_DECISIONS}
    for pair in sorted(pairs, key=lambda row: row["pair_id"]):
        event = state.get(pair["pair_id"], {"decision": manifest.DECISION_PENDING, "reviewed_at": "", "reviewer": ""})
        decision = event["decision"]
        counts[decision] += 1
        if decision == manifest.DECISION_PENDING:
            pending.append(pair["pair_id"])
        rows.append({**pair, "decision": decision, "reviewed_at": event.get("reviewed_at", ""), "reviewer": event.get("reviewer", "")})
    if require_complete and pending:
        raise ValueError(f"Review is incomplete: {len(pending)} pair(s) remain pending")
    file_io.atomic_write_csv(output, manifest.REVIEWED_PAIRS_FIELDS, rows)
    sidecar = {
        "pairs_sha256": file_io.sha256_file(pairs_path),
        "decisions_sha256": file_io.sha256_file(decisions_path) if decisions_path.exists() else None,
        "reviewed_pairs_sha256": file_io.sha256_file(output),
        "counts": counts,
        "complete": not pending,
    }
    file_io.atomic_write_json(output.with_suffix(".json"), sidecar)
    return sidecar
