from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from experiments.core.config_schema import _validate_retrieval_tasks, load_experiment_config
from experiments.core.generation_plan import (
    make_image_directory_generation_plan,
    partition_assets,
)
from experiments.core.split_schema import resolve_retrieval_tasks
from experiments.core.validate_hard_synth_benchmark import validate_hard_synth_benchmark

DISTORTER = Path(__file__).resolve().parents[1] / "datasets" / "image-distorter"
sys.path.insert(0, str(DISTORTER))
from utils import compose_image_with_frame, resize_to_max_dimension  # noqa: E402


def _generator_module():
    spec = importlib.util.spec_from_file_location(
        "hard_synth_generator", DISTORTER / "generate_synth_dataset.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("size", "cap", "expected"),
    [
        ((3000, 2000), 1024, (1024, 683)),
        ((2000, 3000), 1024, (683, 1024)),
        ((3000, 3000), 900, (900, 900)),
        ((400, 300), 1024, (400, 300)),
    ],
)
def test_resize_to_max_dimension(size, cap, expected):
    source = Image.new("RGB", size)
    output, metadata = resize_to_max_dimension(source, cap)
    assert output.size == expected
    assert max(output.size) <= cap
    assert metadata["resized"] is (max(size) > cap)


def test_resize_rejects_invalid_cap():
    with pytest.raises(ValueError, match=">= 1"):
        resize_to_max_dimension(Image.new("RGB", (2, 2)), 0)


def test_source_raster_is_capped_once_and_uses_jpeg_draft(tmp_path):
    module = _generator_module()
    source_path = tmp_path / "large.jpg"
    Image.new("RGB", (4096, 3072), "red").save(source_path, quality=90)

    image = Image.open(source_path)
    raster, metadata = module.prepare_source_raster(image, image.size, 1024)
    try:
        assert raster.mode == "RGBA"
        assert raster.size == (1024, 768)
        assert metadata["input_size"] == [4096, 3072]
        assert metadata["output_size"] == [1024, 768]
        assert metadata["decoder_draft_applied"] is True
        assert max(metadata["decoder_size"]) < 4096
    finally:
        raster.close()


def test_source_working_cap_uses_strictest_bound():
    module = _generator_module()
    base = module.PostProcessConfig(max_source_side=2048)
    assert module.source_working_cap(base, 1024) == 1024
    assert module.source_working_cap(base, 4096) == 2048
    assert (
        module.source_working_cap(module.PostProcessConfig(max_source_side=0), 1024)
        == 1024
    )


def test_frame_composition_respects_working_dimension_cap():
    photo = Image.new("RGBA", (1200, 100), "blue")
    frame = Image.new("RGBA", (2500, 2500), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, 2499, 2499), outline="black", width=250)
    result = compose_image_with_frame(
        photo, frame, rotate_portrait=False, max_frame_dimension=1200
    )
    try:
        assert max(result.size) <= 1200
    finally:
        result.close()


def test_dimension_cap_is_stable_and_bounded():
    module = _generator_module()
    first = module.realized_dimension_cap(
        1024, seed=42, parent_file_id="painting-1", variant_key="historic:print:0"
    )
    assert round(0.85 * 1024) <= first <= 1024
    assert first == module.realized_dimension_cap(
        1024, seed=42, parent_file_id="painting-1", variant_key="historic:print:0"
    )
    assert first != module.realized_dimension_cap(
        1024, seed=42, parent_file_id="painting-1", variant_key="historic:print:1"
    )


def test_historic_schedule_has_exact_equal_quota():
    module = _generator_module()
    schedule = module.role_profile_schedule("historic", 8)
    assert {profile: schedule.count(profile) for profile in set(schedule)} == {
        "archival": 2,
        "print": 2,
        "cropped_record": 2,
        "framed_photo": 2,
    }
    assert module.expand_profile_counts(module.HISTORIC_PROFILE_COUNTS) == [
        ("archival", 0), ("archival", 1),
        ("print", 0), ("print", 1),
        ("cropped_record", 0), ("cropped_record", 1),
        ("framed_photo", 0), ("framed_photo", 1),
    ]


def _split():
    return {
        "schema_version": 2,
        "split_id": "toy",
        "dataset": {"dataset_id": "toy", "image_root": "/tmp"},
        "subsets": {
            "val": {
                "roles": {
                    "modern_original": ["q.jpg"],
                    "historic_archival": ["a.jpg"],
                    "historic_print": ["p.jpg"],
                }
            }
        },
        "images": {
            "q.jpg": {"class_id": 1},
            "a.jpg": {"class_id": 1},
            "p.jpg": {"class_id": 1},
        },
    }


def test_explicit_original_and_generated_roles_do_not_overlap():
    module = _generator_module()
    original = set(module.generated_tags(["modern"], "modern_original", "modern_query", {}))
    generated = set(module.generated_tags(["modern"], "modern_generated", "catalogue", {}))
    assert {"modern_original", "retained_original", "source_original"} <= original
    assert "modern_generated" not in original
    assert "modern_generated" in generated
    assert "modern_original" not in generated
    assert "retained_original" not in generated


def test_raw_image_directory_plan_uses_stable_filename_ids(tmp_path):
    for name in ("Alpha One.jpg", "beta.png", "beta.jpg"):
        Image.new("RGB", (8, 8)).save(tmp_path / name)
    recipe = Path(__file__).resolve().parents[1] / "configs" / "synthetic" / "hard_synth_v1.yml"
    plan = make_image_directory_generation_plan(tmp_path, recipe, seed=42)
    assert plan["assignment_key"] == "parent_file_id"
    assert plan["source_kind"] == "image_dir_one_file_per_class"
    assert plan["source_files"] == {
        "Alpha_One": "Alpha One.jpg",
        "beta": "beta.jpg",
        "beta_002": "beta.png",
    }
    assert set(plan["class_assignments"].values()) == {"train", "val", "test"}


def test_asset_partitioning_is_deterministic_and_disjoint():
    registry = {
        f"sha256:{index}": {"type": "frame" if index < 6 else "overlay", "name": str(index)}
        for index in range(12)
    }
    first = partition_assets(registry, 42)
    assert first == partition_assets(registry, 42)
    assert set(first.values()) == {"train", "val", "test"}
    assert set(first) == set(registry)


def test_multirole_retrieval_selector():
    task = {
        "task_id": "combined",
        "query": {"subset": "val", "role": "modern_original"},
        "candidates": {
            "subset": "val",
            "roles": ["historic_archival", "historic_print"],
        },
        "positive_policy": "same_painting",
        "exclude_self": False,
    }
    resolved = resolve_retrieval_tasks(_split(), [task])[0]
    assert resolved["query_ids"] == ["q.jpg"]
    assert resolved["candidate_ids"] == ["a.jpg", "p.jpg"]
    _validate_retrieval_tasks([task], Path("config.yml"))


def _valid_benchmark_fixture():
    roles = ("modern_original", "historic_archival", "historic_print", "historic_cropped_record", "historic_framed_photo")
    subsets = {}
    images = {}
    for class_id, subset in enumerate(("train", "val", "test"), start=1):
        subset_roles = {role: [] for role in roles}
        query = f"{subset}-query.jpg"
        subset_roles["modern_original"].append(query)
        images[query] = {
            "class_id": class_id, "role": "modern_original", "origin": "source_original",
            "profile": "modern_query", "parent_file_id": f"source-{class_id}",
            "dimensions": {"configured_max_output_dimension": 1024, "realized_cap": 900, "final_size": [900, 600]},
            "transforms": {"crop": {"applied": True}},
        }
        for profile in ("archival", "print", "cropped_record", "framed_photo"):
            role = f"historic_{profile}"
            for index in range(2):
                image_id = f"{subset}-{profile}-{index}.jpg"
                subset_roles[role].append(image_id)
                transforms = {
                    "crop": {"applied": True, "retained_area_fraction": 0.55},
                    "resolution_loss": {"applied": True, "longest_side_pixels": 80},
                    "perspective": {"applied": True, "visible_fraction": 0.7},
                    "vintage": {"applied": True},
                    "paper_document": {"applied": True},
                    "annotations": {"applied": True},
                    "frame": {"name": "frame"},
                    "occlusion": {"applied": True, "coverage": 0.1},
                }
                images[image_id] = {
                    "class_id": class_id, "role": role, "origin": "synthetic",
                    "profile": profile, "profile_version": 1,
                    "parent_file_id": f"source-{class_id}",
                    "dimensions": {"configured_max_output_dimension": 1024, "realized_cap": 900, "final_size": [900, 600]},
                    "severity": {"preset": "extreme" if index else "hard", "bucket": "extreme", "score": 0.795},
                    "transforms": transforms,
                }
        subsets[subset] = {"roles": subset_roles}
    repro = {key: "x" for key in (
        "generator_commit", "generator_version", "recipe_sha256", "generation_plan_sha256",
        "source_dataset_sha256", "asset_registry_sha256", "random_seed", "max_output_dimension",
    )}
    repro["dimension_jitter_fraction"] = 0.15
    repro["max_output_dimension"] = 1024
    split = {
        "schema_version": 2, "split_id": "fixture",
        "dataset": {"dataset_id": "fixture", "image_root": "/unused"},
        "subsets": subsets, "images": images, "build_reproducibility": repro,
    }
    plan = {
        "asset_assignments": {},
        "profile_policy": {"profile_version": 1},
    }
    return split, plan


def test_benchmark_validator_passes_and_detects_count_error():
    split, plan = _valid_benchmark_fixture()
    report = validate_hard_synth_benchmark(split, plan, inspect_headers=False)
    assert report["num_test_classes"] == 1
    split["subsets"]["test"]["roles"]["historic_print"].pop()
    with pytest.raises(ValueError, match="expected 2"):
        validate_hard_synth_benchmark(split, plan, inspect_headers=False)


def test_retrieval_selector_rejects_role_and_roles():
    task = {
        "task_id": "bad",
        "query": {"subset": "val", "role": "modern_original", "roles": ["modern_original"]},
        "candidates": {"subset": "val", "role": "historic_print"},
        "positive_policy": "same_painting",
        "exclude_self": False,
    }
    with pytest.raises(ValueError, match="exactly one"):
        resolve_retrieval_tasks(_split(), [task])
    with pytest.raises(ValueError, match="exactly one"):
        _validate_retrieval_tasks([task], Path("config.yml"))


def test_hard_synth_experiment_configs_parse_and_use_explicit_roles():
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    paths = sorted(
        path for path in config_dir.rglob("*.yml")
        if "hard_synth" in str(path.relative_to(config_dir))
    )
    assert paths
    for path in paths:
        experiment = load_experiment_config(path)
        assert "hard_synth" in experiment.experiment_id
        finetuning = experiment.raw.get("finetuning")
        if not isinstance(finetuning, dict):
            continue
        train_roles = finetuning["data"]["train"]["roles"]
        assert "ALL_ROLES" not in train_roles
        assert "modern_original" in train_roles


def test_hard_synth_wikidata_preprocessing_ablation_pairs_every_frozen_model():
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    source = load_experiment_config(
        config_dir / "eval_hard_synth_finetuned_wikidata_calibrated_v1.yml"
    )
    ablation = load_experiment_config(
        config_dir / "eval_hard_synth_preprocessing_wikidata_calibrated_v1.yml"
    )

    source_frozen = {
        model.model_id: model
        for model in source.models
        if not model.model_id.startswith("local/")
    }
    source_finetuned = {
        model.model_id
        for model in source.models
        if model.model_id.startswith("local/")
    }
    ablation_by_id = {model.model_id: model for model in ablation.models}

    assert source_frozen
    for model_id, source_model in source_frozen.items():
        current = ablation_by_id[model_id]
        native = ablation_by_id[f"model-default/{model_id}"]
        assert current.adapter.model_id == native.adapter.model_id == model_id
        assert current.raw.get("preprocessing") == source_model.raw.get("preprocessing")
        assert native.raw["preprocessing"] == {"resize": {"mode": "model_default"}}
        assert current.raw["color"] == "#E45756"
        assert native.raw["color"] == "#F58518"

    ablation_finetuned = {
        model.model_id
        for model in ablation.models
        if model.model_id.startswith("local/")
    }
    assert ablation_finetuned == source_finetuned
    assert all(ablation_by_id[model_id].raw["color"] == "#4C78A8" for model_id in ablation_finetuned)

    qwen = ablation_by_id["Qwen/Qwen3-VL-Embedding-8B"]
    assert qwen.raw["preprocessing"] == {"resize": {"mode": "model_default"}}
    assert qwen.raw["color"] == "#F58518"
    assert len(ablation.models) == 2 * len(source_frozen) + len(source_finetuned) + 1
    assert ablation.split_file == source.split_file
    assert ablation.raw["evaluation"]["threshold_calibration"] == source.raw["evaluation"]["threshold_calibration"]
    assert ablation.raw["evaluation"]["retrieval_tasks"] == source.raw["evaluation"]["retrieval_tasks"]


def test_hard_synth_pk_sweeps_match_filename_and_batch_composition():
    config_dir = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "finetune_dinov3_vitb_hard_synth_batch_composition_sweep"
    )
    expected_classes_per_batch = {
        ("lora_mlp", 4): {16, 32, 64, 96, 248},
        ("lora_mlp", 2): {112, 128, 192, 224, 248},
        ("lora_linear", 4): {16, 32, 64, 96, 248},
        ("lora_linear", 2): {112, 128, 192, 224, 248},
        ("cls_mlp", 4): {16, 32, 64, 96, 248},
        ("cls_mlp", 2): {112, 128, 192, 224, 248},
    }
    for (variant, expected_k), expected_p_values in expected_classes_per_batch.items():
        configs = sorted(config_dir.glob(f"{variant}_compound_p*_k{expected_k}_b*.yml"))
        assert len(configs) == len(expected_p_values)
        observed_p_values = set()
        for path in configs:
            match = re.fullmatch(
                rf"{variant}_compound_p(?P<p>\d+)_k(?P<k>\d+)_b(?P<b>\d+)",
                path.stem,
            )
            assert match is not None
            filename_p, filename_k, filename_batch = (
                int(match["p"]), int(match["k"]), int(match["b"])
            )
            experiment = load_experiment_config(path)
            sampler = experiment.raw["finetuning"]["sampler"]
            assert sampler["classes_per_batch"] == filename_p
            assert sampler["images_per_class"] == filename_k == expected_k
            assert filename_batch == filename_p * filename_k
            assert f"_p{filename_p}_k{filename_k}_b{filename_batch}_" in experiment.experiment_id
            observed_p_values.add(filename_p)
        assert observed_p_values == expected_p_values


def test_leave_one_profile_out_configs_exclude_profile_from_train_and_calibration():
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    for profile in ("archival", "print", "cropped_record", "framed_photo"):
        path = config_dir / f"finetune_dinov3_vitb_lora_mlp_hard_synth_loo_{profile}_v1.yml"
        experiment = load_experiment_config(path)
        data = experiment.raw["finetuning"]["data"]
        held_role = f"historic_{profile}"
        assert held_role not in data["train"]["roles"]
        validation = data["validation"][0]
        assert held_role not in validation["roles"]
        assert held_role not in validation["retrieval_tasks"][0]["candidates"]["roles"]
        assert held_role in data["test"]["roles"]
