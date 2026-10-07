"""Dataset loading helpers for class-balanced metric-learning finetuning."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .emulsion import historic_emulsion

from experiments.core.config_schema import DatasetConfig
from experiments.core.split_schema import (
    SplitImage,
    ids_for_multirole_selector,
    is_all_roles,
    is_all_subsets,
    records_from_split,
)

ImageTransform = Callable[..., torch.Tensor]


def _transform_accepts_role(transform: object) -> bool:
    try:
        signature = inspect.signature(transform)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name == "role"
        for parameter in signature.parameters.values()
    )


@dataclass(frozen=True)
class MetricImageItem:
    image_id: str
    path: Path
    class_id: str
    role: str | None = None


def _default_transform(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _record_to_item(
    record: SplitImage,
    dataset: DatasetConfig,
    *,
    role: str | None = None,
) -> MetricImageItem:
    path = record.absolute_path(dataset.image_root)
    if not path.is_file():
        raise FileNotFoundError(f"Image for file_id={record.file_id} not found: {path}")
    return MetricImageItem(
        image_id=record.file_id,
        path=path,
        class_id=str(record.class_id),
        role=role,
    )


def load_items_by_ids(
    dataset: DatasetConfig,
    split: dict[str, object],
    image_ids: Sequence[str],
    *,
    source: str,
) -> list[MetricImageItem]:
    """Load same-painting items selected by explicit image IDs."""
    if not image_ids:
        raise ValueError(f"{source} resolved to no image IDs")
    records = records_from_split(split)
    missing = sorted(set(str(image_id) for image_id in image_ids) - set(records))
    if missing:
        raise ValueError(f"{source} references IDs absent from images: {missing[:10]}")
    return [_record_to_item(records[str(file_id)], dataset) for file_id in sorted({str(value) for value in image_ids})]


def _selected_role_memberships(
    split: dict[str, object],
    subset_name: str,
    roles: object,
) -> dict[str, set[str]]:
    """Return selected split-role memberships without relying on image metadata."""
    subsets = split.get("subsets")
    if not isinstance(subsets, dict) or not subsets:
        raise ValueError("Split contains no subsets")
    if is_all_subsets(subset_name):
        selected_subsets = sorted(str(value) for value in subsets)
    else:
        selected_subsets = [subset_name]

    configured_roles = [roles] if isinstance(roles, str) else list(roles)  # type: ignore[arg-type]
    memberships: dict[str, set[str]] = {}
    for selected_subset in selected_subsets:
        subset = subsets.get(selected_subset)
        available = subset.get("roles") if isinstance(subset, dict) else None
        if not isinstance(available, dict) or not available:
            raise ValueError(f"Split subset {selected_subset!r} has no roles")
        if any(is_all_roles(role) for role in configured_roles):
            selected_roles = sorted(str(value) for value in available)
        else:
            selected_roles = [str(value) for value in configured_roles]
        for role in selected_roles:
            if role not in available:
                choices = ", ".join(sorted(str(value) for value in available))
                raise ValueError(
                    f"Split subset {selected_subset!r} does not contain role {role!r}; available: {choices}"
                )
            for image_id in available[role]:
                memberships.setdefault(str(image_id), set()).add(role)
    return memberships


def load_subset_items(
    dataset: DatasetConfig,
    split: dict[str, object],
    selector: dict[str, object],
) -> list[MetricImageItem]:
    """Load same-painting items selected by explicit subset + roles config."""
    if not isinstance(selector, dict):
        raise ValueError("Finetuning data selector must be a mapping")
    if "subset" not in selector or "roles" not in selector:
        raise ValueError("Finetuning data selector must contain subset and roles")
    subset_name = str(selector["subset"])
    roles = selector["roles"]
    image_ids = ids_for_multirole_selector(split, subset_name, roles)  # type: ignore[arg-type]
    memberships = _selected_role_memberships(split, subset_name, roles)
    items = load_items_by_ids(dataset, split, image_ids, source=f"Selector for subset {subset_name!r}")
    return [
        MetricImageItem(
            image_id=item.image_id,
            path=item.path,
            class_id=item.class_id,
            role=next(iter(memberships[item.image_id])) if len(memberships.get(item.image_id, ())) == 1 else None,
        )
        for item in items
    ]


class MetricLearningImageDataset(Dataset):
    """Image dataset returning the dict contract used by the finetuning loop."""

    def __init__(self, items: Sequence[MetricImageItem], transform: ImageTransform | None = None) -> None:
        if not items:
            raise ValueError("MetricLearningImageDataset requires at least one item")
        self.items = list(items)
        self.transform = transform or _default_transform
        self._passes_role = _transform_accepts_role(self.transform)
        self.class_to_label = {class_id: idx for idx, class_id in enumerate(sorted({item.class_id for item in self.items}))}
        self.labels = [self.class_to_label[item.class_id] for item in self.items]
        self.class_ids = [item.class_id for item in self.items]
        self.roles = [item.role for item in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.items[index]
        image = Image.open(item.path).convert("RGB")
        if item.role == "historic_online":
            seed = int(np.random.randint(0, 2**31 - 1))
            array = historic_emulsion(np.asarray(image), np.random.default_rng(seed))
            image = Image.fromarray(array)
        if self._passes_role:
            pixels = self.transform(image, role=item.role)
        else:
            pixels = self.transform(image)
        return {
            "image_id": item.image_id,
            "path": item.path,
            "class_id": item.class_id,
            "label": int(self.labels[index]),
            "role": item.role,
            "index": index,
            "pixel_values": pixels,
        }


def append_online_historic_views(items: Sequence[MetricImageItem]) -> list[MetricImageItem]:
    """Add one historic view per painting, generated from its modern original.

    The pixels are the modern original. The dataset applies a fresh emulsion
    when the role is ``historic_online``.
    """
    originals: dict[str, MetricImageItem] = {}
    for item in items:
        if item.role == "modern_original" and item.class_id not in originals:
            originals[item.class_id] = item
    classes = {item.class_id for item in items}
    missing = sorted(classes - set(originals))
    if missing:
        raise ValueError(
            "Online historic views need a modern_original for every training painting; "
            f"missing {missing[:10]}"
        )
    extra = [
        MetricImageItem(
            image_id=f"{original.image_id}#historic_online",
            path=original.path,
            class_id=class_id,
            role="historic_online",
        )
        for class_id, original in sorted(originals.items())
    ]
    return [*items, *extra]


def append_hard_historic_views(
    items: Sequence[MetricImageItem],
    manifest_path: Path | str,
) -> list[MetricImageItem]:
    """Add hard and extreme historic files listed in a manifest.

    The manifest's ``class_id`` values are the training split's class ids, so
    a hard view joins the painting it depicts. Files live under the manifest's
    ``image_root``, which can differ from the medium catalog.
    """
    path = Path(manifest_path).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    image_root = Path(str(payload["image_root"]))
    known = {item.class_id for item in items}
    extra: list[MetricImageItem] = []
    for view in payload["views"]:
        class_id = str(view["class_id"])
        if class_id not in known:
            continue
        file_id = str(view["file_id"])
        image_path = image_root / file_id
        if not image_path.is_file():
            raise FileNotFoundError(f"Hard historic view not found: {image_path}")
        extra.append(
            MetricImageItem(
                image_id=f"hard::{file_id}",
                path=image_path,
                class_id=class_id,
                role=str(view["role"]),
            )
        )
    if not extra:
        raise ValueError(f"Hard historic manifest {path} added no training views")
    return [*items, *extra]


def origin_code(role: str) -> int | None:
    """Map a dataset role onto the historic or modern origin tag.

    Generated profiles such as ``historic_print`` and ``modern_generated``
    share an origin. The cross-origin loss keeps a pair only when the two
    images have different origins. Roles outside those two groups, such as a
    split that stores every image under ``all``, have no origin.
    """
    name = str(role)
    if name == "historic" or name.startswith("historic_"):
        return 0
    if name == "modern" or name.startswith("modern_"):
        return 1
    return None


def collate_metric_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    roles = [item.get("role") for item in batch]
    origins = None
    if all(isinstance(role, str) and role for role in roles):
        codes = [origin_code(str(role)) for role in roles]
        if all(code is not None for code in codes):
            origins = torch.tensor(codes, dtype=torch.long)
    return {
        "image_id": [str(item["image_id"]) for item in batch],
        "path": [item["path"] for item in batch],
        "class_id": [str(item["class_id"]) for item in batch],
        "labels": torch.tensor([int(item["label"]) for item in batch], dtype=torch.long),
        "origins": origins,
        "indices": torch.tensor([int(item["index"]) for item in batch], dtype=torch.long)
        if all("index" in item for item in batch)
        else None,
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
    }
