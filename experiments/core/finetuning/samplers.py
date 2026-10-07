"""Class-balanced batch samplers for metric-learning finetuning."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Sampler


@dataclass(frozen=True)
class _HardNegativeCandidate:
    negative_class: int | str
    anchor_index: int | None
    negative_index: int | None
    score: float


class PKBatchSampler(Sampler[list[int]]):
    """Yield P x K class-balanced batches.

    Classes are sampled without replacement when enough classes exist, and with
    replacement otherwise. Images within classes are sampled with replacement
    when a class contains fewer than K images.
    """

    def __init__(
        self,
        labels: Sequence[int | str],
        classes_per_batch: int,
        images_per_class: int,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        if classes_per_batch <= 0 or images_per_class <= 0:
            raise ValueError("classes_per_batch and images_per_class must be positive")
        grouped: dict[int | str, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            grouped[label].append(index)
        if not grouped:
            raise ValueError("PKBatchSampler requires at least one labeled sample")
        self.class_to_indices = {label: indices for label, indices in grouped.items()}
        self.classes = sorted(self.class_to_indices, key=str)
        self.classes_per_batch = int(classes_per_batch)
        self.images_per_class = int(images_per_class)
        self.seed = int(seed)
        default_batches = math.ceil(len(labels) / float(self.classes_per_batch * self.images_per_class))
        self.batches_per_epoch = int(batches_per_epoch or max(1, default_batches))
        self.epoch = 0

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            if len(self.classes) >= self.classes_per_batch:
                selected_classes = rng.sample(self.classes, self.classes_per_batch)
            else:
                selected_classes = [rng.choice(self.classes) for _ in range(self.classes_per_batch)]
            for class_id in selected_classes:
                indices = self.class_to_indices[class_id]
                if len(indices) >= self.images_per_class:
                    batch.extend(rng.sample(indices, self.images_per_class))
                else:
                    batch.extend(rng.choice(indices) for _ in range(self.images_per_class))
            rng.shuffle(batch)
            yield batch


def _origin_name(role: str | None) -> str | None:
    """Collapse a dataset role to the historic or modern origin tag."""
    if role is None:
        return None
    name = str(role).strip()
    if name == "historic" or name.startswith("historic_"):
        return "historic"
    if name == "modern" or name.startswith("modern_"):
        return "modern"
    return None


class OriginBalancedPKBatchSampler(Sampler[list[int]]):
    """Yield P-class batches with a fixed number of images from each origin.

    ``per_origin`` names how many historic and modern views each selected
    painting contributes. Sampling is without replacement inside an origin, so
    asking for every available historic view and every available modern view
    puts the whole painting in the batch. A painting that cannot fill the
    requested counts is left out of the sampler.
    """

    def __init__(
        self,
        labels: Sequence[int | str],
        roles: Sequence[str | None],
        classes_per_batch: int,
        per_origin: dict[str, int],
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        if classes_per_batch <= 0:
            raise ValueError("classes_per_batch must be positive")
        if len(labels) != len(roles):
            raise ValueError("roles must have the same length as labels")
        if not isinstance(per_origin, dict) or not per_origin:
            raise ValueError("per_origin must name at least one origin")
        normalized: dict[str, int] = {}
        for origin, count in per_origin.items():
            name = str(origin).strip()
            if name not in {"historic", "modern"}:
                raise ValueError("per_origin keys must be 'historic' or 'modern'")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"per_origin.{name} must be a positive integer")
            normalized[name] = count
        grouped: dict[int | str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, (label, role) in enumerate(zip(labels, roles)):
            origin = _origin_name(role)
            if origin is None or origin not in normalized:
                continue
            grouped[label][origin].append(index)
        eligible: dict[int | str, dict[str, list[int]]] = {}
        for label, by_origin in grouped.items():
            if all(len(by_origin.get(origin, ())) >= count for origin, count in normalized.items()):
                eligible[label] = {origin: list(indices) for origin, indices in by_origin.items()}
        if len(eligible) < classes_per_batch:
            raise ValueError(
                "Origin-balanced sampling needs at least "
                f"{classes_per_batch} paintings with {normalized} views; found {len(eligible)}"
            )
        self.class_to_origin_indices = eligible
        self.classes = sorted(eligible, key=str)
        self.classes_per_batch = int(classes_per_batch)
        self.per_origin = dict(normalized)
        self.images_per_class = sum(normalized.values())
        self.seed = int(seed)
        default_batches = math.ceil(
            len(labels) / float(self.classes_per_batch * self.images_per_class)
        )
        self.batches_per_epoch = int(batches_per_epoch or max(1, default_batches))
        self.epoch = 0

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            selected_classes = rng.sample(self.classes, self.classes_per_batch)
            for class_id in selected_classes:
                by_origin = self.class_to_origin_indices[class_id]
                for origin, count in self.per_origin.items():
                    batch.extend(rng.sample(by_origin[origin], count))
            rng.shuffle(batch)
            yield batch


class RoleStratifiedPKBatchSampler(Sampler[list[int]]):
    """Yield P x K batches with one anchor role and distinct positive roles.

    For every selected class, one image is sampled from ``anchor_role`` and
    ``K - 1`` distinct roles are sampled from ``positive_roles``. One image is
    then sampled from each selected positive role. Samples in other roles do
    not enter batches, but still count toward the default epoch length so a
    role-stratified run can use the same training population and update budget
    as its ordinary-PK control.
    """

    def __init__(
        self,
        labels: Sequence[int | str],
        roles: Sequence[str | None],
        classes_per_batch: int,
        images_per_class: int,
        *,
        anchor_role: str,
        positive_roles: Sequence[str],
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        if classes_per_batch <= 0:
            raise ValueError("classes_per_batch must be positive")
        if images_per_class < 2:
            raise ValueError("RoleStratifiedPKBatchSampler requires images_per_class >= 2")
        if len(labels) != len(roles):
            raise ValueError("roles must have the same length as labels")
        if not labels:
            raise ValueError("RoleStratifiedPKBatchSampler requires at least one labeled sample")

        if isinstance(positive_roles, str):
            raise ValueError("positive_roles must be a sequence of role names, not a string")
        normalized_anchor = str(anchor_role).strip()
        normalized_positives = tuple(str(role).strip() for role in positive_roles)
        if not normalized_anchor:
            raise ValueError("anchor_role must be a non-empty string")
        if not normalized_positives or any(not role for role in normalized_positives):
            raise ValueError("positive_roles must contain non-empty strings")
        if len(set(normalized_positives)) != len(normalized_positives):
            raise ValueError("positive_roles must not contain duplicates")
        if normalized_anchor in normalized_positives:
            raise ValueError("anchor_role must not also appear in positive_roles")
        if images_per_class - 1 > len(normalized_positives):
            raise ValueError(
                "images_per_class - 1 cannot exceed the number of distinct positive_roles"
            )

        grouped: dict[int | str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        role_counts: dict[str, int] = defaultdict(int)
        ambiguous_indices: list[int] = []
        for index, (label, role) in enumerate(zip(labels, roles)):
            if role is None or not str(role).strip():
                ambiguous_indices.append(index)
                continue
            normalized_role = str(role).strip()
            grouped[label][normalized_role].append(index)
            role_counts[normalized_role] += 1
        if ambiguous_indices:
            raise ValueError(
                "RoleStratifiedPKBatchSampler requires one unambiguous role per sample; "
                f"missing or ambiguous roles at indices {ambiguous_indices[:10]}"
            )

        required_roles = (normalized_anchor, *normalized_positives)
        missing_by_class: dict[int | str, list[str]] = {}
        for class_id in sorted(set(labels), key=str):
            missing = [role for role in required_roles if not grouped[class_id].get(role)]
            if missing:
                missing_by_class[class_id] = missing
        if missing_by_class:
            preview = "; ".join(
                f"{class_id}: {', '.join(missing)}"
                for class_id, missing in list(missing_by_class.items())[:10]
            )
            raise ValueError(
                "Every class must contain the anchor role and every configured positive role; "
                f"missing roles by class: {preview}"
            )

        self.class_to_role_indices = {
            class_id: {role: list(indices) for role, indices in by_role.items()}
            for class_id, by_role in grouped.items()
        }
        self.classes = sorted(self.class_to_role_indices, key=str)
        self.classes_per_batch = int(classes_per_batch)
        self.images_per_class = int(images_per_class)
        self.anchor_role = normalized_anchor
        self.positive_roles = normalized_positives
        self.role_counts = dict(sorted(role_counts.items()))
        configured = {normalized_anchor, *normalized_positives}
        self.ignored_role_counts = {
            role: count for role, count in self.role_counts.items() if role not in configured
        }
        self.seed = int(seed)
        default_batches = math.ceil(len(labels) / float(self.classes_per_batch * self.images_per_class))
        self.batches_per_epoch = int(batches_per_epoch or max(1, default_batches))
        self.epoch = 0

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            if len(self.classes) >= self.classes_per_batch:
                selected_classes = rng.sample(self.classes, self.classes_per_batch)
            else:
                selected_classes = [rng.choice(self.classes) for _ in range(self.classes_per_batch)]
            batch: list[int] = []
            for class_id in selected_classes:
                by_role = self.class_to_role_indices[class_id]
                batch.append(rng.choice(by_role[self.anchor_role]))
                selected_roles = rng.sample(self.positive_roles, self.images_per_class - 1)
                batch.extend(rng.choice(by_role[role]) for role in selected_roles)
            rng.shuffle(batch)
            yield batch


class HardNegativePKBatchSampler(Sampler[list[int]]):
    """Yield P x K batches that co-locate mined hard-negative classes.

    ``classes_per_batch`` remains the total number of classes in a batch. The
    sampler picks random anchor classes, then fills available class slots with
    up to ``hard_negatives_per_anchor_class`` mined negative classes for each
    anchor. When mined image IDs are available, those images are forced into the
    K samples for their classes. If a class has no mined negatives, random
    classes fill the remaining slots.
    """

    def __init__(
        self,
        labels: Sequence[int | str],
        classes_per_batch: int,
        images_per_class: int,
        *,
        hard_negative_parquet: Path | str | None = None,
        hard_negatives_per_anchor_class: int = 1,
        image_ids: Sequence[str] | None = None,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        if classes_per_batch <= 0 or images_per_class <= 0:
            raise ValueError("classes_per_batch and images_per_class must be positive")
        if hard_negatives_per_anchor_class < 0:
            raise ValueError("hard_negatives_per_anchor_class must be non-negative")
        if image_ids is not None and len(image_ids) != len(labels):
            raise ValueError("image_ids must have the same length as labels")

        grouped: dict[int | str, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            grouped[label].append(index)
        if not grouped:
            raise ValueError("HardNegativePKBatchSampler requires at least one labeled sample")

        self.class_to_indices = {label: indices for label, indices in grouped.items()}
        self.classes = sorted(self.class_to_indices, key=str)
        self.index_to_class = {index: label for label, indices in self.class_to_indices.items() for index in indices}
        self.image_id_to_index = self._image_lookup(image_ids)
        self.classes_per_batch = int(classes_per_batch)
        self.images_per_class = int(images_per_class)
        self.hard_negatives_per_anchor_class = int(hard_negatives_per_anchor_class)
        self.seed = int(seed)
        default_batches = math.ceil(len(labels) / float(self.classes_per_batch * self.images_per_class))
        self.batches_per_epoch = int(batches_per_epoch or max(1, default_batches))
        self.epoch = 0
        self.hard_candidates = self._load_hard_negatives(hard_negative_parquet) if hard_negative_parquet else {}

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            selected_classes, forced_indices = self._select_classes(rng)
            batch: list[int] = []
            for class_id in selected_classes:
                batch.extend(self._sample_class(class_id, forced_indices.get(class_id, set()), rng))
            rng.shuffle(batch)
            yield batch

    @staticmethod
    def _image_lookup(image_ids: Sequence[str] | None) -> dict[str, int] | None:
        if image_ids is None:
            return None
        lookup = {str(image_id): index for index, image_id in enumerate(image_ids)}
        if len(lookup) != len(image_ids):
            raise ValueError("image_ids must be unique when hard-negative image forcing is enabled")
        return lookup

    def _class_lookup(self) -> dict[str, int | str]:
        lookup: dict[str, int | str] = {}
        for class_id in self.classes:
            key = str(class_id)
            if key in lookup:
                raise ValueError("Class IDs must be unique after string conversion for hard-negative lookup")
            lookup[key] = class_id
        return lookup

    def _load_hard_negatives(self, parquet_path: Path | str | None) -> dict[int | str, list[_HardNegativeCandidate]]:
        if parquet_path is None:
            return {}
        path = Path(parquet_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Hard-negative parquet not found: {path}")
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - dependency is declared for this package
            raise ImportError("HardNegativePKBatchSampler requires pandas/pyarrow to read parquet files") from exc

        rows = pd.read_parquet(path)
        missing = {"anchor_class_id", "negative_class_id"} - set(rows.columns)
        if missing:
            raise ValueError(f"Hard-negative parquet is missing required columns: {', '.join(sorted(missing))}")

        class_lookup = self._class_lookup()
        candidates: dict[int | str, list[_HardNegativeCandidate]] = defaultdict(list)
        for row in rows.to_dict("records"):
            anchor_class = class_lookup.get(str(row["anchor_class_id"]))
            negative_class = class_lookup.get(str(row["negative_class_id"]))
            if anchor_class is None or negative_class is None or anchor_class == negative_class:
                continue
            candidates[anchor_class].append(
                _HardNegativeCandidate(
                    negative_class=negative_class,
                    anchor_index=self._image_index(row.get("anchor_image_id"), anchor_class),
                    negative_index=self._image_index(row.get("negative_image_id"), negative_class),
                    score=self._score(row.get("score", 0.0)),
                )
            )
        for values in candidates.values():
            values.sort(key=lambda candidate: candidate.score, reverse=True)
        return dict(candidates)

    @staticmethod
    def _score(value: object) -> float:
        try:
            score = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if score != score else score

    def _image_index(self, value: object, expected_class: int | str) -> int | None:
        if value is None or self.image_id_to_index is None:
            return None
        index = self.image_id_to_index.get(str(value))
        if index is None or self.index_to_class[index] != expected_class:
            return None
        return index

    def _select_classes(self, rng: random.Random) -> tuple[list[int | str], dict[int | str, set[int]]]:
        selected: list[int | str] = []
        selected_set: set[int | str] = set()
        forced_indices: dict[int | str, set[int]] = defaultdict(set)
        max_attempts = max(self.classes_per_batch * 4, len(self.classes) * 4)
        attempts = 0
        while len(selected) < self.classes_per_batch and attempts < max_attempts:
            attempts += 1
            anchor_class = self._fallback_class(rng, selected_set)
            self._add_class(anchor_class, selected, selected_set)
            for candidate in self._choose_candidates(anchor_class, rng):
                if candidate.anchor_index is not None:
                    forced_indices[anchor_class].add(candidate.anchor_index)
                if candidate.negative_index is not None:
                    forced_indices[candidate.negative_class].add(candidate.negative_index)
                self._add_class(candidate.negative_class, selected, selected_set)
                if len(selected) >= self.classes_per_batch:
                    break
        while len(selected) < self.classes_per_batch:
            self._add_class(self._fallback_class(rng, selected_set), selected, selected_set)
        return selected[: self.classes_per_batch], forced_indices

    def _add_class(self, class_id: int | str, selected: list[int | str], selected_set: set[int | str]) -> bool:
        if len(selected) >= self.classes_per_batch:
            return False
        if len(self.classes) >= self.classes_per_batch and class_id in selected_set:
            return False
        selected.append(class_id)
        selected_set.add(class_id)
        return True

    def _fallback_class(self, rng: random.Random, selected_set: set[int | str]) -> int | str:
        if len(self.classes) >= self.classes_per_batch:
            choices = [class_id for class_id in self.classes if class_id not in selected_set]
            if choices:
                return rng.choice(choices)
        return rng.choice(self.classes)

    def _choose_candidates(self, anchor_class: int | str, rng: random.Random) -> list[_HardNegativeCandidate]:
        if self.hard_negatives_per_anchor_class <= 0:
            return []
        candidates = list(self.hard_candidates.get(anchor_class, []))
        rng.shuffle(candidates)
        chosen: list[_HardNegativeCandidate] = []
        seen_classes: set[int | str] = set()
        for candidate in candidates:
            if candidate.negative_class in seen_classes:
                continue
            chosen.append(candidate)
            seen_classes.add(candidate.negative_class)
            if len(chosen) >= self.hard_negatives_per_anchor_class:
                break
        return chosen

    def _sample_class(self, class_id: int | str, forced: set[int], rng: random.Random) -> list[int]:
        indices = self.class_to_indices[class_id]
        forced_valid = sorted(index for index in forced if index in indices)
        if len(forced_valid) >= self.images_per_class:
            return rng.sample(forced_valid, self.images_per_class)
        sample = list(forced_valid)
        remaining = [index for index in indices if index not in forced_valid]
        needed = self.images_per_class - len(sample)
        if len(remaining) >= needed:
            sample.extend(rng.sample(remaining, needed))
            return sample
        sample.extend(remaining)
        while len(sample) < self.images_per_class:
            sample.append(rng.choice(indices))
        return sample


_STATIC_HISTORIC = (
    "historic_archival",
    "historic_print",
    "historic_cropped_record",
    "historic_framed_photo",
)
_MODERN_ROLES = ("modern_original", "modern_generated")


def _quota_bucket(role: str | None, quotas: dict[str, int]) -> list[str]:
    """Return the quota keys a role can fill."""
    if role is None:
        return []
    name = str(role)
    matched: list[str] = []
    if name in quotas:
        matched.append(name)
    if name in _STATIC_HISTORIC and "historic" in quotas:
        matched.append("historic")
    if name in _MODERN_ROLES and "modern" in quotas:
        matched.append("modern")
    return matched


class RoleQuotaPKBatchSampler(Sampler[list[int]]):
    """Yield P-class batches with a fixed count from each named role.

    ``historic`` means any of the four static historic profiles, and ``modern``
    means either modern role. ``historic_online`` stays its own quota, so an
    online emulsion view does not consume a static historic slot.
    """

    def __init__(
        self,
        labels: Sequence[int | str],
        roles: Sequence[str | None],
        classes_per_batch: int,
        quotas: dict[str, int],
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        if classes_per_batch <= 0:
            raise ValueError("classes_per_batch must be positive")
        if len(labels) != len(roles):
            raise ValueError("roles must have the same length as labels")
        if not isinstance(quotas, dict) or not quotas:
            raise ValueError("role_quota_pk requires sampler.quotas")
        normalized: dict[str, int] = {}
        for key, count in quotas.items():
            name = str(key).strip()
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"sampler.quotas.{name} must be a positive integer")
            normalized[name] = count
        grouped: dict[int | str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, (label, role) in enumerate(zip(labels, roles)):
            for bucket in _quota_bucket(role, normalized):
                grouped[label][bucket].append(index)
        eligible = {
            label: buckets
            for label, buckets in grouped.items()
            if all(len(buckets.get(key, ())) >= count for key, count in normalized.items())
        }
        if len(eligible) < classes_per_batch:
            raise ValueError(
                "Role-quota sampling needs at least "
                f"{classes_per_batch} paintings with {normalized}; found {len(eligible)}"
            )
        self.class_to_buckets = eligible
        self.classes = sorted(eligible, key=str)
        self.classes_per_batch = int(classes_per_batch)
        self.quotas = dict(normalized)
        self.images_per_class = sum(normalized.values())
        self.seed = int(seed)
        default_batches = math.ceil(len(labels) / float(self.classes_per_batch * self.images_per_class))
        self.batches_per_epoch = int(batches_per_epoch or max(1, default_batches))
        self.epoch = 0

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            selected = rng.sample(self.classes, self.classes_per_batch)
            for class_id in selected:
                buckets = self.class_to_buckets[class_id]
                for key, count in self.quotas.items():
                    batch.extend(rng.sample(buckets[key], count))
            rng.shuffle(batch)
            yield batch


class CrossEraGraphPKBatchSampler(Sampler[list[int]]):
    """Yield batches of a painting and the historic paintings it confuses.

    Until ``set_neighbors`` is called the sampler falls back to ordinary PK
    sampling, so the first epoch can still run if the probe fails to find
    pairs. Image draws prefer an even modern/historic split so a confuser's
    historic view is actually in the batch.
    """

    def __init__(
        self,
        labels: Sequence[int | str],
        roles: Sequence[str | None],
        classes_per_batch: int,
        images_per_class: int,
        *,
        hard_fraction: float = 0.5,
        neighbors_pool: int = 128,
        refresh_every_epochs: int = 1,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        if classes_per_batch < 2 or images_per_class < 2:
            raise ValueError("cross_era_graph_pk requires at least 2 classes and 2 images per class")
        if not 0.0 <= float(hard_fraction) <= 1.0:
            raise ValueError("hard_fraction must be between 0 and 1")
        if neighbors_pool <= 0:
            raise ValueError("neighbors_pool must be positive")
        if len(labels) != len(roles):
            raise ValueError("roles must have the same length as labels")
        grouped: dict[int | str, list[int]] = defaultdict(list)
        by_origin: dict[int | str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, (label, role) in enumerate(zip(labels, roles)):
            grouped[label].append(index)
            origin = _origin_name(role)
            if origin is not None:
                by_origin[label][origin].append(index)
        if len(grouped) < classes_per_batch:
            raise ValueError(
                f"cross_era_graph_pk needs at least {classes_per_batch} paintings; found {len(grouped)}"
            )
        self.class_to_indices = {label: indices for label, indices in grouped.items()}
        self.class_to_origin = {label: dict(origins) for label, origins in by_origin.items()}
        self.classes = sorted(self.class_to_indices, key=str)
        self.classes_per_batch = int(classes_per_batch)
        self.images_per_class = int(images_per_class)
        self.hard_fraction = float(hard_fraction)
        self.neighbors_pool = int(neighbors_pool)
        self.refresh_every_epochs = int(refresh_every_epochs)
        self.neighbors: dict[int | str, list[int | str]] = {}
        self.seed = int(seed)
        default_batches = math.ceil(len(labels) / float(self.classes_per_batch * self.images_per_class))
        self.batches_per_epoch = int(batches_per_epoch or max(1, default_batches))
        self.epoch = 0
        self.last_selected_classes: list[list[int | str]] = []

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_neighbors(self, neighbors: dict[int | str, list[int | str]]) -> None:
        known = set(self.classes)
        cleaned: dict[int | str, list[int | str]] = {}
        for anchor, ranked in neighbors.items():
            if anchor not in known:
                continue
            cleaned[anchor] = [class_id for class_id in ranked if class_id in known and class_id != anchor]
        self.neighbors = cleaned

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.last_selected_classes = []
        for _ in range(self.batches_per_epoch):
            selected = self._select_classes(rng)
            self.last_selected_classes.append(list(selected))
            batch: list[int] = []
            for class_id in selected:
                batch.extend(self._sample_class(class_id, rng))
            rng.shuffle(batch)
            yield batch

    def _select_classes(self, rng: random.Random) -> list[int | str]:
        anchor = rng.choice(self.classes)
        selected = [anchor]
        ranked = list(self.neighbors.get(anchor, []))[: self.neighbors_pool]
        slots = self.classes_per_batch - 1
        n_hard = min(len(ranked), int(round(slots * self.hard_fraction)))
        selected.extend(ranked[:n_hard])
        pool = [class_id for class_id in ranked[n_hard:] if class_id not in selected]
        n_random = slots - n_hard
        if pool and n_random > 0:
            selected.extend(rng.sample(pool, min(n_random, len(pool))))
        while len(selected) < self.classes_per_batch:
            remaining = [class_id for class_id in self.classes if class_id not in selected]
            if not remaining:
                break
            selected.append(rng.choice(remaining))
        return selected

    def _sample_class(self, class_id: int | str, rng: random.Random) -> list[int]:
        by_origin = self.class_to_origin.get(class_id, {})
        historic = list(by_origin.get("historic", ()))
        modern = list(by_origin.get("modern", ()))
        half = self.images_per_class // 2
        if historic and modern and half > 0:
            taken_h = rng.sample(historic, min(half, len(historic)))
            taken_m = rng.sample(modern, min(half, len(modern)))
            sample = taken_h + taken_m
            rest = [index for index in self.class_to_indices[class_id] if index not in sample]
            while len(sample) < self.images_per_class and rest:
                pick = rng.choice(rest)
                rest.remove(pick)
                sample.append(pick)
            while len(sample) < self.images_per_class:
                sample.append(rng.choice(self.class_to_indices[class_id]))
            return sample
        indices = self.class_to_indices[class_id]
        if len(indices) >= self.images_per_class:
            return rng.sample(indices, self.images_per_class)
        return [rng.choice(indices) for _ in range(self.images_per_class)]
