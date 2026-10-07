from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from deduplication import file_io, manifest
from deduplication.review import (
    _review_components,
    append_decision,
    create_app,
    finalize_review,
    replay_decisions,
    resolve_pending,
)


def _review_row(project: Path) -> dict[str, str]:
    image_dir = project / "work/images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (4, 5), "red").save(image_dir / "a.jpg")
    Image.new("RGB", (6, 7), "blue").save(image_dir / "b.jpg")
    return {
        "pair_id": "pair1",
        "image_a_id": "alpha/a.jpg",
        "image_a_shared_rel_path": "work/images/a.jpg",
        "image_a_source": "alpha",
        "image_a_raw_relative_path": "a.jpg",
        "image_a_width": "4",
        "image_a_height": "5",
        "image_b_id": "beta/b.jpg",
        "image_b_shared_rel_path": "work/images/b.jpg",
        "image_b_source": "beta",
        "image_b_raw_relative_path": "b.jpg",
        "image_b_width": "6",
        "image_b_height": "7",
        "cosine_similarity": "0.9",
        "lightglue_prediction": "1",
        "lightglue_confidence": "0.8",
        "match_count": "10",
    }


def test_append_replay_finalize_and_flask(project):
    pair_path = project / "work/review/review_pairs.csv"
    decisions = project / "work/review/review_decisions.csv"
    output = project / "work/review/reviewed_pairs.csv"
    file_io.atomic_write_csv(pair_path, manifest.REVIEW_PAIRS_FIELDS, [_review_row(project)])

    with pytest.raises(ValueError, match="incomplete"):
        finalize_review(pair_path, decisions, output, require_complete=True)
    app = create_app(pair_path, decisions, "tester")
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 302
    cluster_url = response.headers["Location"]
    page = client.get(cluster_url)
    assert page.status_code == 200
    assert b"2 images" in page.data
    pair_response = client.get("/pairs")
    assert pair_response.status_code == 302
    assert pair_response.headers["Location"].endswith("/pair/pair1")
    assert client.get("/pair/pair1").status_code == 200
    assert client.get("/pair/pair1/image/a").status_code == 200
    assert client.get("/pair/unknown").status_code == 404
    assert client.get(f"{cluster_url}/image/0").status_code == 200
    assert client.get(f"{cluster_url}/image/99").status_code == 404
    assert client.post(f"{cluster_url}/decision", data={"decision": "accepted"}).status_code == 302
    assert replay_decisions(decisions, {"pair1"})["pair1"]["decision"] == "accepted"

    summary = finalize_review(pair_path, decisions, output, require_complete=True)
    assert summary["complete"] is True
    append_decision(decisions, "pair1", "pending", "tester")
    assert replay_decisions(decisions, {"pair1"})["pair1"]["decision"] == "pending"


def test_review_components_are_largest_first_and_bulk_decisions_override_all_edges(project):
    row1 = _review_row(project)
    row2 = dict(row1)
    row2.update(
        {
            "pair_id": "pair2",
            "image_a_id": "beta/b.jpg",
            "image_a_shared_rel_path": "work/images/b.jpg",
            "image_a_source": "beta",
            "image_a_raw_relative_path": "b.jpg",
            "image_a_width": "6",
            "image_a_height": "7",
            "image_b_id": "gamma/c.jpg",
            "image_b_shared_rel_path": "work/images/c.jpg",
            "image_b_source": "gamma",
            "image_b_raw_relative_path": "c.jpg",
            "image_b_width": "8",
            "image_b_height": "9",
        }
    )
    Image.new("RGB", (8, 9), "green").save(project / "work/images/c.jpg")
    row3 = dict(row1)
    row3.update({"pair_id": "pair3", "image_a_id": "other/a.jpg", "image_b_id": "other/b.jpg"})

    components = _review_components([row3, row2, row1])
    assert [(item["size"], item["edge_count"]) for item in components] == [(3, 2), (2, 1)]

    pair_path = project / "work/review/review_pairs.csv"
    decisions = project / "work/review/review_decisions.csv"
    file_io.atomic_write_csv(pair_path, manifest.REVIEW_PAIRS_FIELDS, [row1, row2])
    append_decision(decisions, "pair1", "rejected", "metadata")
    client = create_app(pair_path, decisions, "tester").test_client()
    cluster_url = client.get("/").headers["Location"]
    page = client.get(cluster_url)
    assert b"Accept its pairs" in page.data

    response = client.post(f"{cluster_url}/image/0/decision", data={"decision": "accepted"})
    assert response.status_code == 302
    state = replay_decisions(decisions, {"pair1", "pair2"})
    assert state["pair1"]["decision"] == "accepted"
    assert "pair2" not in state

    response = client.post(f"{cluster_url}/decision", data={"decision": "accepted"})
    assert response.status_code == 302
    assert response.headers["Location"] == cluster_url
    state = replay_decisions(decisions, {"pair1", "pair2"})
    assert {event["decision"] for event in state.values()} == {"accepted"}

    response = client.post(f"{cluster_url}/decision", data={"decision": "pending"})
    assert response.status_code == 302
    assert response.headers["Location"] == cluster_url
    state = replay_decisions(decisions, {"pair1", "pair2"})
    assert {event["decision"] for event in state.values()} == {"pending"}


def test_resolve_pending_appends_complete_bulk_decisions(project):
    pair_path = project / "work/review/review_pairs.csv"
    decisions = project / "work/review/review_decisions.csv"
    file_io.atomic_write_csv(pair_path, manifest.REVIEW_PAIRS_FIELDS, [_review_row(project)])
    result = resolve_pending(pair_path, decisions, "accepted", "bulk-test")
    assert result["resolved_count"] == 1
    assert result["counts"]["accepted"] == 1
    assert resolve_pending(pair_path, decisions, "accepted", "bulk-test")["resolved_count"] == 0


def test_unknown_and_truncated_decisions_are_rejected(project):
    decisions = project / "decisions.csv"
    append_decision(decisions, "unknown", "accepted", "tester")
    with pytest.raises(ValueError, match="unknown pair"):
        replay_decisions(decisions, {"known"})
    decisions.write_text("pair_id,decision,reviewed_at,reviewer\npair,accepted\n")
    with pytest.raises(ValueError, match="malformed"):
        replay_decisions(decisions)
