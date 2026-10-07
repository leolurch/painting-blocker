"""Trainable ResNet retrieval wrappers.

ResNet-18, ResNet-50, and ResNet-152 share one wrapper. The Multi-Similarity
head is an MLP whose hidden width is twice the trunk and whose output matches
the trunk.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import TrainableRetrievalModel, set_module_trainable
from .pooling import GeMPooling2d
from .projection import ProjectionHead, normalize_embedding

TRUNK_DIM = {"resnet18": 512, "resnet50": 2048, "resnet152": 2048}
BACKBONE_ALIASES = {"microsoft/resnet-50": "resnet50"}
MLP_TYPES = {"resnet50_projection", "resnet50_gem_projection", "resnet_gem_mlp"}


def canonical_backbone(name: str) -> str:
    cleaned = BACKBONE_ALIASES.get(str(name).strip(), str(name).strip())
    if cleaned not in TRUNK_DIM:
        raise ValueError("ResNet backbone must be resnet18, resnet50, or resnet152")
    return cleaned


def trunk_dim(backbone: str) -> int:
    return TRUNK_DIM[canonical_backbone(backbone)]


class ResNetRetrievalModel(TrainableRetrievalModel):
    """ResNet trunk with GeM or average pooling and a whitening map or MLP."""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pooling: str = "avgpool",
        projection_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        unfreeze_layer4: bool = False,
        pretrained: bool = True,
        gem_p: float = 3.0,
        gem_learn_p: bool = True,
        *,
        weights: str | None = None,
        head: str = "mlp",
        freeze_bn_running_stats: bool = False,
        model_type: str | None = None,
    ) -> None:
        super().__init__()
        self.backbone_name = canonical_backbone(backbone_name)
        self.pooling_type = str(pooling)
        if str(head).lower() == "whitening":
            raise ValueError("ResNet head must be mlp")
        self.head_type = "mlp"
        self.freeze_bn_running_stats = bool(freeze_bn_running_stats)
        feature_dim = TRUNK_DIM[self.backbone_name]
        self.projection_dim = int(projection_dim)
        if self.backbone_name == "resnet50" and (model_type or "").startswith("resnet50_"):
            self.model_type = model_type
        else:
            self.model_type = model_type or "resnet_gem_mlp"
        resnet = _load_backbone(self.backbone_name, _resolve_weights(self.backbone_name, weights, pretrained))
        self.layer4 = resnet.layer4
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            self.layer4,
        )
        if pooling == "avgpool":
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
        elif pooling == "gem":
            self.pool = GeMPooling2d(p=gem_p, learn_p=gem_learn_p)
        else:
            raise ValueError("ResNet pooling must be 'avgpool' or 'gem'")
        self.projection = ProjectionHead(feature_dim, self.projection_dim, hidden_dim, dropout)
        if freeze_backbone:
            set_module_trainable(self.features, False)
        if unfreeze_layer4:
            set_module_trainable(self.layer4, True)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_bn_running_stats:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward_pooled(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """L2-normalized GeM or average-pool descriptor, before the head."""
        return normalize_embedding(self.pool(self.features(pixel_values)))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.forward_pooled(pixel_values)
        return normalize_embedding(self.projection(pooled))

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            **super().checkpoint_metadata(),
            "pooling_type": self.pooling_type,
            "backbone_name": self.backbone_name,
            "head_type": self.head_type,
        }


ResNet50RetrievalModel = ResNetRetrievalModel


def _resolve_weights(backbone: str, weights: str | None, pretrained: bool) -> str:
    if weights is not None and str(weights).strip():
        name = str(weights).strip().lower().replace("-", "_")
    elif not pretrained:
        name = "none"
    elif backbone == "resnet18":
        name = "imagenet_v1"
    else:
        name = "imagenet_v2"
    allowed = {"none", "swsl", "imagenet_v1", "imagenet_v2"}
    if name not in allowed:
        raise ValueError(f"ResNet weights must be one of {sorted(allowed)}")
    if name == "swsl" and backbone == "resnet152":
        raise ValueError("ResNet-152 has no semi-weakly supervised checkpoint")
    if name == "imagenet_v2" and backbone == "resnet18":
        raise ValueError("ResNet-18 torchvision weights are ImageNet-1K V1 only")
    return name


def _load_backbone(backbone: str, weights: str) -> nn.Module:
    if weights == "swsl":
        hub_name = {"resnet18": "resnet18_swsl", "resnet50": "resnet50_swsl"}[backbone]
        try:
            return torch.hub.load(
                "facebookresearch/semi-supervised-ImageNet1K-models",
                hub_name,
                trust_repo=True,
            )
        except TypeError:
            return torch.hub.load("facebookresearch/semi-supervised-ImageNet1K-models", hub_name)
    from torchvision.models import resnet18, resnet50, resnet152
    from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet152_Weights

    constructors = {"resnet18": resnet18, "resnet50": resnet50, "resnet152": resnet152}
    enums = {
        ("resnet18", "imagenet_v1"): ResNet18_Weights.IMAGENET1K_V1,
        ("resnet50", "imagenet_v1"): ResNet50_Weights.IMAGENET1K_V1,
        ("resnet50", "imagenet_v2"): ResNet50_Weights.IMAGENET1K_V2,
        ("resnet152", "imagenet_v1"): ResNet152_Weights.IMAGENET1K_V1,
        ("resnet152", "imagenet_v2"): ResNet152_Weights.IMAGENET1K_V2,
    }
    weight = None if weights == "none" else enums[(backbone, weights)]
    return constructors[backbone](weights=weight)


def build_resnet_model(config: dict[str, object]) -> ResNetRetrievalModel:
    pooling_cfg = dict(config.get("pooling") or {})
    model_type = str(config.get("type") or config.get("model_type") or "")
    if model_type == "resnet_gem_whitening" or str(config.get("head", "")).lower() == "whitening":
        raise ValueError("ResNet head must be mlp")
    head = "mlp"
    backbone = canonical_backbone(str(config.get("backbone", "resnet50")))
    projection_dim = int(config.get("projection_dim", 512))
    hidden_dim = int(config.get("projection_hidden_dim", 1024))
    stored_type = model_type or "resnet_gem_mlp"
    return ResNetRetrievalModel(
        backbone_name=backbone,
        pooling=str(pooling_cfg.get("type", config.get("pooling_type", "avgpool"))),
        projection_dim=projection_dim,
        hidden_dim=hidden_dim,
        dropout=float(config.get("dropout", 0.1)),
        freeze_backbone=bool(config.get("freeze_backbone", True)),
        unfreeze_layer4=bool(config.get("unfreeze_layer4", False)),
        pretrained=bool(config.get("pretrained", True)),
        gem_p=float(pooling_cfg.get("p_init", 3.0)),
        gem_learn_p=bool(pooling_cfg.get("learn_p", True)),
        weights=None if config.get("weights") is None else str(config.get("weights")),
        head=head,
        freeze_bn_running_stats=bool(config.get("freeze_bn_running_stats", False)),
        model_type=stored_type,
    )
