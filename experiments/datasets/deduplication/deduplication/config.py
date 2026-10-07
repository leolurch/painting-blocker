"""TOML configuration loading, model fingerprints, and run provenance."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import file_io

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")
VALID_COMPARISONS = (">=", ">")
CALIBRATED_MODEL_ID = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
CALIBRATED_MODEL_REVISION = "b80367753773648a6793235ab9c65cdbb029506f"


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    revision: str | None
    pooling: str
    resize_mode: str
    resize_size: int
    normalize: bool
    dtype: str
    batch_size: int
    preprocess_workers: int
    output_dim: int | None

    def fingerprint(self) -> dict[str, Any]:
        """Complete, calibration-coupled model + preprocessing fingerprint."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "pooling": self.pooling,
            "preprocessing": {"resize_mode": self.resize_mode, "resize_size": self.resize_size},
            "normalize": self.normalize,
            "dtype": self.dtype,
        }

    def fingerprint_sha256(self) -> str:
        return file_io.sha256_json(self.fingerprint())


@dataclass(frozen=True)
class RetrievalConfig:
    min_retrieval_window: int
    cosine_threshold: float
    comparison: str
    query_batch_size: int

    def compare(self, score: float) -> bool:
        if self.comparison == ">=":
            return score >= self.cosine_threshold
        return score > self.cosine_threshold


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, Any]
    images_dir: Path
    extensions: tuple[str, ...]
    model: ModelConfig
    retrieval: RetrievalConfig
    superpoint_max_num_keypoints: int
    classifier_path: Path
    classifier_sha256: str
    keeper_source_priority: tuple[str, ...]
    calibration: dict[str, Any] = field(default_factory=dict)

    def config_sha256(self) -> str:
        return file_io.sha256_json(self.raw)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _require(section: dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise ValueError(f"Missing required config key [{where}].{key}")
    return section[key]


def load_config(path: Path | str) -> Config:
    config_path = Path(path).expanduser().resolve()
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    images = raw.get("images", {})
    images_dir = file_io.resolve_from_project(str(images.get("dir", "work/images")))
    allowed_roots = (file_io.PROJECT_DIR.resolve(), file_io.WORK_DIR.resolve())
    if not any(_is_relative_to(images_dir, root) for root in allowed_roots):
        raise ValueError(
            "[images].dir must remain inside the deduplication project or active work directory"
        )
    extensions = tuple(str(e).lower() for e in images.get("extensions", DEFAULT_EXTENSIONS))
    if not extensions or any(not e.startswith(".") for e in extensions):
        raise ValueError("[images].extensions must be a non-empty list of dotted extensions")

    m = raw.get("model", {})
    model = ModelConfig(
        model_id=str(_require(m, "id", "model")),
        revision=(str(m["revision"]) if m.get("revision") else None),
        pooling=str(m.get("pooling", "default")),
        resize_mode=str(m.get("resize_mode", "letterbox")),
        resize_size=int(m.get("resize_size", 512)),
        normalize=bool(m.get("normalize", True)),
        dtype=str(m.get("dtype", "float16")),
        batch_size=int(m.get("batch_size", 8)),
        preprocess_workers=int(m.get("preprocess_workers", 4)),
        output_dim=(int(m["output_dim"]) if m.get("output_dim") is not None else None),
    )

    r = raw.get("retrieval", {})
    comparison = str(r.get("comparison", ">="))
    if comparison not in VALID_COMPARISONS:
        raise ValueError(f"retrieval.comparison must be one of {VALID_COMPARISONS}, got {comparison!r}")
    if "cosine_threshold" not in r:
        # Retrieval must fail rather than silently use a default/top-k-only behaviour.
        raise ValueError(
            "Missing [retrieval].cosine_threshold. Candidate generation requires the calibrated "
            "cosine threshold and its model provenance."
        )
    # ``top_k`` used to cap retrieval before thresholding. The current contract is exhaustive
    # threshold retrieval OR a minimum nearest-neighbour window. Accepting top_k here would make
    # old configs appear to work while silently retaining the obsolete, lossy semantics.
    if "top_k" in r:
        raise ValueError(
            "[retrieval].top_k is obsolete; replace it with min_retrieval_window. "
            "Candidates now satisfy threshold OR the minimum retrieval window."
        )
    retrieval = RetrievalConfig(
        min_retrieval_window=int(r.get("min_retrieval_window", 200)),
        cosine_threshold=float(r["cosine_threshold"]),
        comparison=comparison,
        query_batch_size=int(r.get("query_batch_size", 512)),
    )
    if retrieval.min_retrieval_window <= 0 or retrieval.query_batch_size <= 0:
        raise ValueError(
            "retrieval.min_retrieval_window and retrieval.query_batch_size must be positive"
        )
    if model.batch_size <= 0 or model.preprocess_workers <= 0:
        raise ValueError("model.batch_size and model.preprocess_workers must be positive")
    calibrated_contract = {
        "model.id": (model.model_id, CALIBRATED_MODEL_ID),
        "model.revision": (model.revision, CALIBRATED_MODEL_REVISION),
        "model.pooling": (model.pooling, "default"),
        "model.resize_mode": (model.resize_mode, "letterbox"),
        "model.resize_size": (model.resize_size, 512),
        "model.normalize": (model.normalize, True),
        "model.output_dim": (model.output_dim, 4096),
        "retrieval.comparison": (retrieval.comparison, ">="),
    }
    mismatches = [
        f"{key}={actual!r} (calibrated value: {expected!r})"
        for key, (actual, expected) in calibrated_contract.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "Configuration does not match the calibrated DINO retrieval contract:\n  - "
            + "\n  - ".join(mismatches)
        )
    if not math.isfinite(retrieval.cosine_threshold) or not -1.0 <= retrieval.cosine_threshold <= 1.0:
        raise ValueError("retrieval.cosine_threshold must be a finite cosine value in [-1, 1]")
    calibration = dict(raw.get("calibration", {}))
    for key in ("run_id", "artifact_path", "threshold_source", "target_pc"):
        _require(calibration, key, "calibration")
    target_pc = float(calibration["target_pc"])
    if not 0.0 < target_pc <= 1.0:
        raise ValueError("calibration.target_pc must be in (0, 1]")

    sp = raw.get("superpoint", {})
    clf = raw.get("classifier", {})
    classifier_path = Path(str(clf.get("path", "classifier.pkl")))
    if not classifier_path.is_absolute():
        classifier_path = (file_io.PROJECT_DIR / classifier_path).resolve()

    keeper = raw.get("keeper", {})

    return Config(
        path=config_path,
        raw=raw,
        images_dir=images_dir,
        extensions=extensions,
        model=model,
        retrieval=retrieval,
        superpoint_max_num_keypoints=int(sp.get("max_num_keypoints", 2048)),
        classifier_path=classifier_path,
        classifier_sha256=str(_require(clf, "sha256", "classifier")),
        keeper_source_priority=tuple(str(s) for s in keeper.get("source_priority", ())),
        calibration=calibration,
    )


def write_resolved_config(config: Config, work: Path) -> Path:
    """Write ``resolved_config.json`` recording the fully resolved configuration."""
    resolved = {
        "config_path": file_io.rel_to_project(config.path),
        "config_sha256": config.config_sha256(),
        "images_dir": file_io.rel_to_project(config.images_dir),
        "extensions": list(config.extensions),
        "model": config.model.fingerprint(),
        "model_execution": {
            "batch_size": config.model.batch_size,
            "preprocess_workers": config.model.preprocess_workers,
        },
        "model_fingerprint_sha256": config.model.fingerprint_sha256(),
        "retrieval": {
            "min_retrieval_window": config.retrieval.min_retrieval_window,
            "cosine_threshold": config.retrieval.cosine_threshold,
            "comparison": config.retrieval.comparison,
            "query_batch_size": config.retrieval.query_batch_size,
        },
        "superpoint": {"max_num_keypoints": config.superpoint_max_num_keypoints},
        "classifier": {
            "path": file_io.rel_to_project(config.classifier_path),
            "sha256": config.classifier_sha256,
        },
        "keeper_source_priority": list(config.keeper_source_priority),
        "calibration": config.calibration,
    }
    path = work / "resolved_config.json"
    file_io.atomic_write_json(path, resolved)
    return path


def write_run_metadata(config: Config, work: Path, extra: dict[str, Any] | None = None) -> Path:
    """Write/refresh ``run.json`` (source-code revision, model fingerprints, timestamps)."""
    path = work / "run.json"
    existing = file_io.read_json(path) if path.exists() else {}
    run = {
        "started_at": existing.get("started_at", file_io.utc_now_iso()),
        "updated_at": file_io.utc_now_iso(),
        "config_sha256": config.config_sha256(),
        "model_fingerprint_sha256": config.model.fingerprint_sha256(),
        "classifier_sha256": config.classifier_sha256,
        "calibration": config.calibration,
        "git": file_io.git_info(),
    }
    if extra:
        run.update(extra)
    merged = {**existing, **run}
    file_io.atomic_write_json(path, merged)
    return path
