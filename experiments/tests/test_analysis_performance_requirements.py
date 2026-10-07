import json
from pathlib import Path

import pytest

from experiments.core.analysis_charts import (
    AnalysisRenderError,
    _blocking_performance_metrics,
)
from experiments.core.run_figures import ModelSpec


def _spec(model_id: str, storage_key: str) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        storage_key=storage_key,
        label=storage_key,
        color="#000000",
        linestyle="-",
        group="frozen",
        is_baseline=False,
        model_dir=Path("."),
    )


def test_performance_metrics_are_loaded_directly_from_completed_run(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.yml"
    config_path.write_text("analysis_id: test\n", encoding="utf-8")
    run_dir = tmp_path / "benchmark"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps({
            "models": [{
                "model_id": "benchmark/model-a",
                "peak_allocated_bytes": 2 * 1073741824,
                "sequential_images_per_second": 12.34,
            }]
        }),
        encoding="utf-8",
    )
    addon_dir = tmp_path / "addon"
    addon_dir.mkdir()
    (addon_dir / "run.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (addon_dir / "summary.json").write_text(
        json.dumps({
            "models": [{
                "model_id": "addon/model-b",
                "peak_allocated_bytes": 3 * 1073741824,
                "sequential_images_per_second": 10.0,
            }]
        }),
        encoding="utf-8",
    )
    chart = {
        "performance_requirements": {
            "source_runs": ["benchmark", "addon"],
            "missing_data": "render_dash",
            "model_id_aliases": {"analysis-a": "benchmark/model-a"},
            "columns": [
                {
                    "key": "peak_vram_gib",
                    "label": "Peak VRAM (GiB)",
                    "source_field": "peak_allocated_bytes",
                    "divisor": 1073741824,
                    "objective": "min",
                    "decimals": 2,
                },
                {
                    "key": "images_per_second",
                    "label": "Images/s",
                    "source_field": "sequential_images_per_second",
                    "objective": "max",
                    "decimals": 1,
                },
            ],
        }
    }

    columns, values, metadata = _blocking_performance_metrics(
        config_path,
        chart,
        [
            _spec("unused/model-a", "analysis-a"),
            _spec("addon/model-b", "analysis-b"),
            _spec("missing/model", "analysis-c"),
        ],
    )

    assert [column["key"] for column in columns] == [
        "peak_vram_gib",
        "images_per_second",
    ]
    assert values["analysis-a"] == {
        "peak_vram_gib": 2.0,
        "images_per_second": 12.34,
    }
    assert values["analysis-b"] == {
        "peak_vram_gib": 3.0,
        "images_per_second": 10.0,
    }
    assert values["analysis-c"] == {}
    assert metadata is not None
    assert [source["summary_path"] for source in metadata["sources"]] == [
        run_dir / "summary.json",
        addon_dir / "summary.json",
    ]
    assert metadata["missing_models"] == ["analysis-c"]

    (run_dir / "run.json").write_text(
        json.dumps({"status": "running"}), encoding="utf-8"
    )
    with pytest.raises(AnalysisRenderError, match="not completed"):
        _blocking_performance_metrics(
            config_path, chart, [_spec("unused/model-a", "analysis-a")]
        )
