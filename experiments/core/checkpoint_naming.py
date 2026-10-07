"""Canonical, machine-readable names for finetuning checkpoint runs.

Checkpoint files retain the stable ``best_checkpoint.pt`` and
``last_checkpoint.pt`` names.  The containing run directory carries the
semantic configuration name so post-evaluation model IDs can preserve it.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

ARCHITECTURES = ("clsmlp", "loralin", "loramlp", "siglip2loramlp", "cliploramlp", "resnetloramlp")
_LORA_ARCHITECTURES = frozenset({"loralin", "loramlp", "siglip2loramlp", "cliploramlp", "resnetloramlp"})
_MLP_ARCHITECTURES = frozenset({"clsmlp", "loramlp", "siglip2loramlp", "cliploramlp", "resnetloramlp"})
_TRANSFER_ARCHITECTURES = {
    "clip_lora_qv_projection": "cliploramlp",
    "resnet_hf_lora_conv_projection": "resnetloramlp",
}
_FIELD_ORDER = ("p", "k", "r", "la", "lb", "mh", "llr", "hd", "hlr", "hdo", "sam", "ep")
_SPAN_TOKEN = re.compile(r"^(?:all|first[1-9][0-9]*|last[1-9][0-9]*)$")
_SAMPLER_NAMES = {
    "pk": "pk",
    "pk_batch": "pk",
    "pk_batch_sampler": "pk",
    "role_stratified_pk": "rspk",
    "role_stratified_pk_batch": "rspk",
    "role_stratified_pk_batch_sampler": "rspk",
    "origin_balanced_pk": "opk",
    "origin_balanced_pk_batch": "opk",
    "role_quota_pk": "rqpk",
    "role_quota_pk_batch": "rqpk",
    "cross_era_graph_pk": "gpk",
    "cross_era_graph_pk_batch": "gpk",
}


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _integer(value: object, field: str) -> str:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"checkpoint name field {field!r} must be numeric") from exc
    if number != number.to_integral_value():
        raise ValueError(f"checkpoint name field {field!r} must be an integer")
    return str(int(number))


def _decimal(value: object, field: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"checkpoint name field {field!r} must be numeric") from exc
    if not number.is_finite():
        raise ValueError(f"checkpoint name field {field!r} must be finite")
    return number


def _learning_rate(value: object, field: str) -> str:
    number = _decimal(value, field)
    if number <= 0:
        raise ValueError(f"checkpoint name field {field!r} must be positive")
    normalized = number.normalize()
    exponent = normalized.adjusted()
    coefficient = normalized.scaleb(-exponent)
    coefficient_text = format(coefficient, "f").rstrip("0").rstrip(".")
    return f"{coefficient_text}e{exponent}"


def _slug_decimal(value: object, field: str) -> str:
    number = _decimal(value, field)
    text = format(number, "f").rstrip("0").rstrip(".") if number else "0"
    return text.replace("-", "m").replace(".", "p")


def _architecture(model: dict[str, Any]) -> str | None:
    model_type = str(model.get("type") or model.get("model_type") or "").lower()
    head = _mapping(model.get("head"))
    head_type = str(head.get("type") or "").lower()
    if model_type in {"dinov3_lora_qv_projection", "dinov3_lora_projection"}:
        if head_type == "linear_projection":
            return "loralin"
        if head_type == "mlp_projection" or head.get("hidden_dim") is not None:
            return "loramlp"
    if model_type == "siglip2_lora_qv_projection":
        return "siglip2loramlp"
    if model_type in _TRANSFER_ARCHITECTURES:
        return _TRANSFER_ARCHITECTURES[model_type]
    if model_type in {"dinov3_projection", "dinov3_gem_projection"} and (
        model.get("projection_hidden_dim") is not None or head_type == "mlp_projection"
    ):
        return "clsmlp"
    return None


def checkpoint_name_from_finetuning(config: Mapping[str, Any]) -> str | None:
    """Build the approved semantic checkpoint name from a finetuning mapping.

    Unsupported architectures return ``None`` so general-purpose runners can
    retain their existing fallback run IDs.
    """
    cfg = _mapping(config.get("finetuning")) or dict(config)
    model = _mapping(cfg.get("model"))
    architecture = _architecture(model)
    if architecture is None:
        return None

    sampler = _mapping(cfg.get("sampler"))
    optimizer = _mapping(cfg.get("optimizer"))
    train = _mapping(cfg.get("train"))
    head = _mapping(model.get("head"))
    lora = _mapping(model.get("lora"))
    images_per_class = sampler.get("images_per_class", sampler.get("k"))
    if images_per_class is None:
        per_origin = _mapping(sampler.get("per_origin"))
        if per_origin:
            images_per_class = sum(int(count) for count in per_origin.values())
        quotas = _mapping(sampler.get("quotas"))
        if images_per_class is None and quotas:
            images_per_class = sum(int(count) for count in quotas.values())

    fields: dict[str, str] = {
        "p": _integer(sampler.get("classes_per_batch", sampler.get("p")), "p"),
        "k": _integer(images_per_class, "k"),
    }
    if architecture in _LORA_ARCHITECTURES:
        fields.update(
            r=_integer(lora.get("rank"), "r"),
            la=_integer(lora.get("alpha"), "la"),
            llr=_learning_rate(optimizer.get("lr_lora"), "llr"),
        )
        span = lora.get("span")
        if span is not None:
            token = str(span).strip().lower()
            if _SPAN_TOKEN.fullmatch(token) is None:
                raise ValueError("lora.span must be 'all', 'first<N>', or 'last<N>'")
            fields["lb"] = token
    if architecture == "siglip2loramlp":
        pooling = _mapping(model.get("pooling"))
        fields["mh"] = "train" if bool(pooling.get("trainable", False)) else "frozen"
    if architecture in _MLP_ARCHITECTURES:
        fields["hd"] = _integer(
            head.get("hidden_dim", model.get("projection_hidden_dim")), "hd"
        )
    fields["hlr"] = _learning_rate(
        optimizer.get("lr_head", optimizer.get("head_lr")), "hlr"
    )
    if architecture in _MLP_ARCHITECTURES:
        fields["hdo"] = _slug_decimal(
            head.get("dropout", model.get("dropout", 0.0)), "hdo"
        )
    sampler_type = str(sampler.get("type", sampler.get("name", "pk"))).lower()
    if sampler_type not in _SAMPLER_NAMES:
        raise ValueError(f"Unsupported sampler.type for checkpoint name: {sampler_type!r}")
    fields["sam"] = _SAMPLER_NAMES[sampler_type]
    fields["ep"] = _integer(train.get("epochs"), "ep")

    return "__".join(
        [architecture, *(f"{key}-{fields[key]}" for key in _FIELD_ORDER if key in fields)]
    )


def checkpoint_run_name(config: Mapping[str, Any], run_token: str | None = None) -> str | None:
    """Return a semantic run-directory name, optionally with a unique run token."""
    name = checkpoint_name_from_finetuning(config)
    if name is None or not run_token:
        return name
    safe_token = re.sub(r"[^A-Za-z0-9.-]+", "-", str(run_token)).strip("-.")
    return f"{name}__run-{safe_token}" if safe_token else name


def parse_checkpoint_name(identifier: str | Path) -> dict[str, str] | None:
    """Mine a canonical checkpoint name from a path or model identifier."""
    for component in reversed(re.split(r"[/\\]", str(identifier))):
        stem = component[:-3] if component.endswith(".pt") else component
        parts = stem.split("__")
        if not parts or parts[0] not in ARCHITECTURES:
            continue
        fields: dict[str, str] = {"architecture": parts[0]}
        valid = True
        for token in parts[1:]:
            if "-" not in token:
                valid = False
                break
            key, value = token.split("-", 1)
            if not key or not value or key in fields:
                valid = False
                break
            fields[key] = value
        if valid and all(key in fields for key in ("p", "k")):
            semantic = [parts[0]]
            semantic.extend(
                f"{key}-{fields[key]}" for key in _FIELD_ORDER if key in fields
            )
            fields["title"] = "__".join(semantic)
            return fields
    return None
