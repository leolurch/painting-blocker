from __future__ import annotations

from pathlib import Path

from deduplication import file_io
from deduplication.config import load_config, write_resolved_config


PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_CLASSIFIER_SHA = "f9b313a44d86b7352583cd365d2af9181ec018708768aeaeaa4c762ebe7ff24d"


def test_classifier_is_referenced_not_shipped():
    note = (PROJECT / "UPSTREAM_SOURCES.md").read_text(encoding="utf-8")
    example = (PROJECT / "config/deduplication.example.toml").read_text(encoding="utf-8")
    assert "https://github.com/HPI-Information-Systems/smARTmatch" in note
    assert "https://github.com/HPI-Information-Systems/smARTmatch" in example
    assert list(PROJECT.rglob("*.pkl")) == []


def test_example_config_is_pinned_and_resolved_paths_are_portable(tmp_path):
    config = load_config(PROJECT / "config/deduplication.example.toml")
    assert config.model.revision == "b80367753773648a6793235ab9c65cdbb029506f"
    assert config.model.output_dim == 4096
    assert config.retrieval.cosine_threshold == 0.11395263671875
    assert config.retrieval.comparison == ">="
    assert "https://github.com/HPI-Information-Systems/smARTmatch" in (PROJECT / "config/deduplication.example.toml").read_text()
    output = write_resolved_config(config, tmp_path)
    text = output.read_text()
    assert str(PROJECT.resolve()) not in text


def test_alternate_calibrated_threshold_is_allowed_with_provenance(tmp_path):
    text = (PROJECT / "config/deduplication.example.toml").read_text()
    text = text.replace("cosine_threshold = 0.11395263671875", "cosine_threshold = 0.11395263671875")
    text = text.replace("target_pc = 0.995", "target_pc = 0.995")
    path = tmp_path / "pc995.toml"
    path.write_text(text)
    config = load_config(path)
    assert config.retrieval.cosine_threshold == 0.11395263671875
    assert config.calibration["target_pc"] == 0.995


def test_only_export_module_mentions_sqlite():
    mentions = []
    for path in (PROJECT / "deduplication").glob("*.py"):
        if "import sqlite3" in path.read_text(encoding="utf-8"):
            mentions.append(path.name)
    assert mentions == ["export_dataset.py"]
