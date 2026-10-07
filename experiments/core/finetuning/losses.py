"""Metric-learning losses shared by finetuning recipes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MultiSimilarityStats:
    valid_anchors: int
    positive_pairs: int
    negative_pairs: int
    mined_positive_pairs: int
    mined_negative_pairs: int


class MultiSimilarityLoss(nn.Module):
    """Multi-Similarity loss with per-anchor informative-pair mining."""

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 50.0,
        base: float = 0.5,
        mining_epsilon: float = 0.1,
    ) -> None:
        super().__init__()
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.base = float(base)
        self.mining_epsilon = float(mining_epsilon)
        self.cross_origin_only = False
        # 1 and 1 is the original pair rule. A smaller same-origin weight
        # keeps those pairs in the batch and reduces their share of the loss.
        self.same_origin_weight = 1.0
        self.cross_origin_weight = 1.0
        # None counts every image as an anchor. 0 is historic, 1 is modern.
        self.anchor_origin: int | None = None
        self.last_stats = MultiSimilarityStats(0, 0, 0, 0, 0)

    @property
    def needs_origins(self) -> bool:
        return (
            self.cross_origin_only
            or self.same_origin_weight != 1.0
            or self.cross_origin_weight != 1.0
            or self.anchor_origin is not None
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        origins: torch.Tensor | None = None,
        memory_negative_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be [B, D], got {tuple(embeddings.shape)}")
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels must be a 1D tensor with one item per embedding")
        batch_size = int(embeddings.shape[0])
        if batch_size <= 1:
            self.last_stats = MultiSimilarityStats(0, 0, 0, 0, 0)
            return embeddings.sum() * 0.0

        embeddings = embeddings.float()
        similarities = embeddings @ embeddings.T
        identity = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        labels_equal = labels[:, None] == labels[None, :]
        # Default pair rule, unchanged: every distinct pair in the batch counts.
        positive_mask = labels_equal & ~identity
        negative_mask = ~labels_equal & ~identity
        pair_weights: torch.Tensor | None = None
        if self.cross_origin_only:
            positive_mask, negative_mask = _keep_cross_origin_pairs(
                positive_mask, negative_mask, origins, batch_size
            )
        elif self.same_origin_weight != 1.0 or self.cross_origin_weight != 1.0:
            pair_weights = _origin_pair_weights(
                origins, batch_size, self.same_origin_weight, self.cross_origin_weight, similarities
            )
            positive_mask = positive_mask & (pair_weights > 0)
            negative_mask = negative_mask & (pair_weights > 0)
        if self.anchor_origin is not None and (
            origins is None or origins.ndim != 1 or int(origins.shape[0]) != batch_size
        ):
            raise ValueError("anchor_origin requires an origin tag for every embedding")

        losses: list[torch.Tensor] = []
        mined_pos_total, mined_neg_total = 0, 0
        for anchor in range(batch_size):
            if self.anchor_origin is not None and int(origins[anchor]) != self.anchor_origin:
                continue
            pos_scores = similarities[anchor][positive_mask[anchor]]
            neg_scores = similarities[anchor][negative_mask[anchor]]
            pos_weights = None if pair_weights is None else pair_weights[anchor][positive_mask[anchor]]
            neg_weights = None if pair_weights is None else pair_weights[anchor][negative_mask[anchor]]
            if pos_scores.numel() == 0:
                continue
            if memory_negative_scores is not None:
                remembered = memory_negative_scores[anchor]
                remembered = remembered[torch.isfinite(remembered)]
                if remembered.numel() > 0:
                    remembered = remembered[remembered <= pos_scores.min()]
                if remembered.numel() > 0:
                    neg_scores = torch.cat((neg_scores, remembered))
                    if neg_weights is not None:
                        neg_weights = torch.cat((neg_weights, neg_weights.new_ones(remembered.shape)))
            if neg_scores.numel() == 0:
                continue
            max_negative = neg_scores.max()
            min_positive = pos_scores.min()
            pos_keep = pos_scores < max_negative + self.mining_epsilon
            neg_keep = neg_scores > min_positive - self.mining_epsilon
            mined_pos = pos_scores[pos_keep]
            mined_neg = neg_scores[neg_keep]
            mined_pos_total += int(mined_pos.numel())
            mined_neg_total += int(mined_neg.numel())
            terms: list[torch.Tensor] = []
            if mined_pos.numel() > 0:
                pos_logits = -self.alpha * (mined_pos - self.base)
                mined_pos_weights = None if pos_weights is None else pos_weights[pos_keep]
                terms.append(_weighted_logsumexp(pos_logits, mined_pos_weights, self.alpha))
            if mined_neg.numel() > 0:
                neg_logits = self.beta * (mined_neg - self.base)
                mined_neg_weights = None if neg_weights is None else neg_weights[neg_keep]
                terms.append(_weighted_logsumexp(neg_logits, mined_neg_weights, self.beta))
            if terms:
                losses.append(sum(terms))

        self.last_stats = MultiSimilarityStats(
            valid_anchors=len(losses),
            positive_pairs=int(positive_mask.sum().item()),
            negative_pairs=int(negative_mask.sum().item()),
            mined_positive_pairs=mined_pos_total,
            mined_negative_pairs=mined_neg_total,
        )
        if not losses:
            return embeddings.sum() * 0.0
        return torch.stack(losses).mean()



def _keep_cross_origin_pairs(
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    origins: torch.Tensor | None,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop pairs whose images share an origin tag.

    Historic and modern are the two tags. A generated profile such as
    ``historic_print`` shares the historic tag with the other historic views.
    """
    if origins is None:
        raise ValueError("cross-origin loss requires an origin tag for every embedding")
    if origins.ndim != 1 or origins.shape[0] != batch_size:
        raise ValueError("origins must be a 1D tensor with one item per embedding")
    different_origin = origins[:, None] != origins[None, :]
    return positive_mask & different_origin, negative_mask & different_origin


def _origin_pair_weights(
    origins: torch.Tensor | None,
    batch_size: int,
    same_weight: float,
    cross_weight: float,
    similarities: torch.Tensor,
) -> torch.Tensor:
    """Weight each pair by whether its two images share an origin."""
    if origins is None:
        raise ValueError("origin weights require an origin tag for every embedding")
    if origins.ndim != 1 or int(origins.shape[0]) != batch_size:
        raise ValueError("origins must be a 1D tensor with one item per embedding")
    different = origins[:, None] != origins[None, :]
    same = similarities.new_tensor(float(same_weight))
    cross = similarities.new_tensor(float(cross_weight))
    return torch.where(different, cross, same)


def _weighted_logsumexp(
    logits: torch.Tensor,
    weights: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """``log(1 + sum_i w_i exp(logit_i)) / scale``. Unit weights match the original term."""
    if weights is None:
        packed = torch.cat([logits.new_zeros(1), logits])
    else:
        packed = torch.cat([logits.new_zeros(1), logits + torch.log(weights)])
    return torch.logsumexp(packed, dim=0) / scale


class SupConLoss(nn.Module):
    """Supervised contrastive loss for normalized retrieval embeddings."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be [B, D], got {tuple(embeddings.shape)}")
        embeddings = embeddings.float()
        labels = labels.view(-1)
        batch_size = int(embeddings.shape[0])
        if batch_size <= 1:
            return embeddings.sum() * 0.0
        logits = (embeddings @ embeddings.T) / self.temperature
        identity = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        logits = logits.masked_fill(identity, torch.finfo(logits.dtype).min)
        positive_mask = (labels[:, None] == labels[None, :]) & ~identity
        valid = positive_mask.any(dim=1)
        if not bool(valid.any()):
            return embeddings.sum() * 0.0
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        per_anchor = -(log_prob * positive_mask.float()).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
        return per_anchor[valid].mean()


def _pair_masks(
    labels: torch.Tensor,
    origins: torch.Tensor | None,
    *,
    cross_origin_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    identity = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    same_label = labels[:, None] == labels[None, :]
    positive = same_label & ~identity
    negative = ~same_label & ~identity
    if not cross_origin_only:
        return positive, negative
    if origins is None:
        raise ValueError("cross-origin pairs require an origin tag for every embedding")
    different = origins[:, None] != origins[None, :]
    return positive & different, negative & different


def threshold_consistent_margin(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    origins: torch.Tensor | None,
    *,
    m_pos: float,
    m_neg: float,
    weight_pos: float,
    weight_neg: float,
    cross_origin_only: bool,
) -> torch.Tensor:
    """Mean hinge on positives below ``m_pos`` and negatives above ``m_neg``.

    This is the threshold-consistent margin of Zhang et al. (ICLR 2024). The
    two margins are absolute cosine values, so every painting is pulled toward
    the same match threshold.
    """
    positive, negative = _pair_masks(labels, origins, cross_origin_only=cross_origin_only)
    scores = embeddings.float() @ embeddings.float().T
    loss = embeddings.sum() * 0.0
    hard_pos = scores[positive]
    hard_pos = hard_pos[hard_pos <= m_pos]
    if weight_pos > 0 and hard_pos.numel() > 0:
        loss = loss + float(weight_pos) * (float(m_pos) - hard_pos).mean()
    hard_neg = scores[negative]
    hard_neg = hard_neg[hard_neg >= m_neg]
    if weight_neg > 0 and hard_neg.numel() > 0:
        loss = loss + float(weight_neg) * (hard_neg - float(m_neg)).mean()
    return loss


def koleo_historic(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    origins: torch.Tensor | None,
    *,
    weight: float,
) -> torch.Tensor:
    """Negative log distance to the nearest other historic painting.

    The Kozachenko-Leononenko estimator used by SSCD (Pizzi et al., CVPR 2022).
    Only historic embeddings enter, because the blocker ranks one historic
    catalog.
    """
    if weight <= 0 or origins is None:
        return embeddings.sum() * 0.0
    historic = origins == 0
    points = embeddings.float()[historic]
    point_labels = labels[historic]
    if points.shape[0] < 2:
        return embeddings.sum() * 0.0
    distances = torch.cdist(points, points)
    distances.fill_diagonal_(float("inf"))
    same = point_labels[:, None] == point_labels[None, :]
    distances = distances.masked_fill(same, float("inf"))
    nearest = distances.min(dim=1).values
    valid = torch.isfinite(nearest)
    if not bool(valid.any()):
        return embeddings.sum() * 0.0
    return -float(weight) * torch.log(nearest[valid].clamp_min(1e-6)).mean()


def cross_era_mixup(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    origins: torch.Tensor | None,
    *,
    positives: bool,
    synthetic_negatives: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Interpolate modern and historic views, and optionally hard negatives.

    An interpolated positive keeps the painting's label. A synthetic hard
    negative, the point between an anchor and its hardest other painting, gets
    a fresh label so it is a negative for every real painting.
    """
    if origins is None:
        raise ValueError("cross-era mixup requires an origin tag for every embedding")
    if not positives and not synthetic_negatives:
        return embeddings, labels, origins
    pieces_z = [embeddings]
    pieces_y = [labels]
    pieces_o = [origins]
    next_label = int(labels.max().item()) + 1
    for label in labels.unique():
        historic = torch.nonzero((labels == label) & (origins == 0), as_tuple=False).flatten()
        modern = torch.nonzero((labels == label) & (origins == 1), as_tuple=False).flatten()
        if historic.numel() == 0 or modern.numel() == 0:
            continue
        historic_index = historic[int(torch.randint(historic.numel(), (1,), device=embeddings.device))]
        modern_index = modern[int(torch.randint(modern.numel(), (1,), device=embeddings.device))]
        if positives:
            alpha = embeddings.new_empty(1).uniform_(0.2, 0.8)
            mixed = torch.nn.functional.normalize(
                alpha * embeddings[modern_index] + (1.0 - alpha) * embeddings[historic_index],
                dim=0,
            )
            pieces_z.append(mixed.unsqueeze(0))
            pieces_y.append(labels[modern_index].view(1))
            pieces_o.append(origins[modern_index].view(1))
        if synthetic_negatives:
            scores = embeddings.float()[modern_index] @ embeddings.float().T
            scores = scores.clone()
            scores[labels == label] = float("-inf")
            if not bool(torch.isfinite(scores).any()):
                continue
            negative_index = int(torch.argmax(scores))
            beta = embeddings.new_empty(1).uniform_(0.3, 0.7)
            synthetic = torch.nn.functional.normalize(
                beta * embeddings[modern_index] + (1.0 - beta) * embeddings[negative_index],
                dim=0,
            )
            pieces_z.append(synthetic.unsqueeze(0))
            pieces_y.append(labels.new_tensor([next_label]))
            pieces_o.append(origins.new_tensor([0]))
            next_label += 1
    if len(pieces_z) == 1:
        return embeddings, labels, origins
    return torch.cat(pieces_z, dim=0), torch.cat(pieces_y, dim=0), torch.cat(pieces_o, dim=0)


class HistoricCatalogMemory:
    """Detached historic embeddings from earlier steps, keyed by dataset index."""

    def __init__(self, warmup_steps: int) -> None:
        if warmup_steps < 0:
            raise ValueError("memory.warmup_steps must be non-negative")
        self.warmup_steps = int(warmup_steps)
        self._embeddings: dict[int, torch.Tensor] = {}
        self._labels: dict[int, int] = {}
        self.updates = 0

    def observe(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        origins: torch.Tensor | None,
        indices: torch.Tensor | None,
    ) -> None:
        if origins is None or indices is None:
            return
        historic = torch.nonzero(origins == 0, as_tuple=False).flatten()
        stored = torch.nn.functional.normalize(embeddings.detach().float(), dim=1)
        for row in historic.tolist():
            key = int(indices[row])
            self._embeddings[key] = stored[row].cpu()
            self._labels[key] = int(labels[row])
        self.updates += 1

    def negative_scores(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        origins: torch.Tensor | None,
        indices: torch.Tensor | None,
        *,
        cross_origin_only: bool,
    ) -> torch.Tensor | None:
        if self.updates < self.warmup_steps or not self._embeddings:
            return None
        excluded = set()
        if indices is not None:
            excluded = {int(value) for value in indices.detach().cpu().tolist()}
        keys = [key for key in self._embeddings if key not in excluded]
        if not keys:
            return None
        remembered = torch.stack([self._embeddings[key] for key in keys]).to(embeddings.device)
        remembered_labels = torch.tensor(
            [self._labels[key] for key in keys], dtype=labels.dtype, device=labels.device
        )
        scores = embeddings.float() @ remembered.float().T
        scores = scores.masked_fill(labels[:, None] == remembered_labels[None, :], float("-inf"))
        if cross_origin_only and origins is not None:
            scores = scores.masked_fill(origins[:, None] != 1, float("-inf"))
        return scores


class CompositeMetricLoss(nn.Module):
    """Multi-Similarity plus the optional domain terms.

    With no regularizer, no memory, and no mixup, ``build_loss`` returns the
    plain multi-similarity module instead, so the current recipe is untouched.
    """

    accepts_batch_context = True

    def __init__(
        self,
        base: MultiSimilarityLoss,
        *,
        margins: dict[str, float] | None = None,
        koleo_weight: float = 0.0,
        memory: HistoricCatalogMemory | None = None,
        mixup_positives: bool = False,
        mixup_negatives: bool = False,
        prototype_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.margins = dict(margins or {})
        self.koleo_weight = float(koleo_weight)
        self.memory = memory
        self.mixup_positives = bool(mixup_positives)
        self.mixup_negatives = bool(mixup_negatives)
        self.prototype_weight = float(prototype_weight)

    @property
    def cross_origin_only(self) -> bool:
        return bool(self.base.cross_origin_only)

    @property
    def last_stats(self) -> MultiSimilarityStats:
        return self.base.last_stats

    def observe_batch(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        origins: torch.Tensor | None,
        indices: torch.Tensor | None,
    ) -> None:
        if self.memory is not None:
            self.memory.observe(embeddings, labels, origins, indices)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        origins: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mixup_positives or self.mixup_negatives:
            embeddings, labels, origins = cross_era_mixup(
                embeddings,
                labels,
                origins,
                positives=self.mixup_positives,
                synthetic_negatives=self.mixup_negatives,
            )
        remembered = None
        if self.memory is not None:
            remembered = self.memory.negative_scores(
                embeddings,
                labels,
                origins,
                indices,
                cross_origin_only=self.cross_origin_only,
            )
        loss = self.base(embeddings, labels, origins, memory_negative_scores=remembered)
        if self.margins:
            loss = loss + threshold_consistent_margin(
                embeddings,
                labels,
                origins,
                m_pos=float(self.margins["m_pos"]),
                m_neg=float(self.margins["m_neg"]),
                weight_pos=float(self.margins["weight_pos"]),
                weight_neg=float(self.margins["weight_neg"]),
                cross_origin_only=bool(self.margins["cross_origin_only"]),
            )
        if self.koleo_weight > 0:
            loss = loss + koleo_historic(embeddings, labels, origins, weight=self.koleo_weight)
        if self.prototype_weight > 0:
            loss = loss + origin_prototype_pull(
                embeddings, labels, origins, weight=self.prototype_weight
            )
        return loss


def origin_prototype_pull(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    origins: torch.Tensor | None,
    *,
    weight: float,
) -> torch.Tensor:
    """Pull a painting's mean modern embedding toward its mean historic embedding.

    Classes that lack one of the two origins in this batch contribute nothing.
    The means are L2-normalized, so the term is one minus their cosine.
    """
    if origins is None:
        raise ValueError("origin prototypes require an origin tag for every embedding")
    if origins.ndim != 1 or int(origins.shape[0]) != int(embeddings.shape[0]):
        raise ValueError("origins must be a 1D tensor with one item per embedding")
    total = embeddings.sum() * 0.0
    count = 0
    for label in labels.unique():
        members = labels == label
        modern = embeddings[members & (origins == 1)]
        historic = embeddings[members & (origins == 0)]
        if modern.shape[0] == 0 or historic.shape[0] == 0:
            continue
        modern_mean = torch.nn.functional.normalize(modern.float().mean(dim=0), dim=0)
        historic_mean = torch.nn.functional.normalize(historic.float().mean(dim=0), dim=0)
        total = total + (1.0 - torch.dot(modern_mean, historic_mean))
        count += 1
    if count == 0:
        return embeddings.sum() * 0.0
    return float(weight) * total / count


def _margin_config(entry: dict[str, object]) -> dict[str, float | bool]:
    m_pos = float(entry.get("m_pos", 0.8))
    m_neg = float(entry.get("m_neg", 0.5))
    if m_pos <= m_neg:
        raise ValueError("loss regularizer tcm requires m_pos above m_neg")
    pairs = str(entry.get("pairs", "cross_origin"))
    if pairs not in {"cross_origin", "all"}:
        raise ValueError("loss regularizer tcm pairs must be 'cross_origin' or 'all'")
    return {
        "m_pos": m_pos,
        "m_neg": m_neg,
        "weight_pos": float(entry.get("weight_pos", entry.get("weight", 1.0))),
        "weight_neg": float(entry.get("weight_neg", entry.get("weight", 1.0))),
        "cross_origin_only": pairs == "cross_origin",
    }


def build_loss(config: dict[str, object] | None = None) -> nn.Module:
    """Instantiate the configured finetuning loss."""
    cfg = dict(config or {})
    name = str(cfg.get("name", "multi_similarity")).lower()
    if name in {"supcon", "supervised_contrastive"}:
        if any(cfg.get(key) for key in ("regularizers", "memory", "embedding_mixup", "origin_weights")):
            raise ValueError("supcon does not take regularizers, memory, embedding_mixup, or origin_weights")
        return SupConLoss(temperature=float(cfg.get("temperature", 0.07)))
    if name not in {"multi_similarity", "multisimilarity", "ms"}:
        raise ValueError("loss.name must be 'multi_similarity' or 'supcon'")
    loss = MultiSimilarityLoss(
        alpha=float(cfg.get("alpha", 2.0)),
        beta=float(cfg.get("beta", 50.0)),
        base=float(cfg.get("base", 0.5)),
        mining_epsilon=float(cfg.get("mining_epsilon", cfg.get("miner_epsilon", 0.1))),
    )
    loss.cross_origin_only = bool(cfg.get("cross_origin_only", False))
    same_weight, cross_weight, anchor_origin = _origin_weight_config(cfg)
    if loss.cross_origin_only and (same_weight != 1.0 or cross_weight != 1.0 or anchor_origin is not None):
        raise ValueError(
            "loss.cross_origin_only already drops same-origin pairs; "
            "set origin_weights or anchor_origin on their own"
        )
    loss.same_origin_weight = same_weight
    loss.cross_origin_weight = cross_weight
    loss.anchor_origin = anchor_origin
    margins: dict[str, float | bool] | None = None
    koleo_weight = 0.0
    prototype_weight = 0.0
    regularizers = cfg.get("regularizers") or []
    if regularizers and not isinstance(regularizers, list):
        raise ValueError("loss.regularizers must be a list")
    for entry in regularizers:
        if not isinstance(entry, dict):
            raise ValueError("each loss regularizer must be a mapping")
        regularizer = str(entry.get("name", "")).lower()
        if regularizer in {"tcm", "threshold_consistent_margin"}:
            if margins is not None:
                raise ValueError("loss.regularizers lists tcm more than once")
            margins = _margin_config(entry)
        elif regularizer == "koleo":
            koleo_weight += float(entry.get("weight", 0.1))
        elif regularizer in {"origin_prototypes", "origin_prototype"}:
            if prototype_weight > 0:
                raise ValueError("loss.regularizers lists origin_prototypes more than once")
            prototype_weight = float(entry.get("weight", 0.1))
            if prototype_weight < 0:
                raise ValueError("origin_prototypes weight must be non-negative")
        else:
            raise ValueError(f"unsupported loss regularizer {regularizer!r}")
    memory_cfg = cfg.get("memory")
    memory = None
    if memory_cfg:
        if not isinstance(memory_cfg, dict):
            raise ValueError("loss.memory must be a mapping")
        if bool(memory_cfg.get("enabled", True)):
            memory = HistoricCatalogMemory(warmup_steps=int(memory_cfg.get("warmup_steps", 30)))
    mixup_cfg = dict(cfg.get("embedding_mixup") or {})
    mixup_positives = bool(mixup_cfg.get("cross_era_positives", False))
    mixup_negatives = bool(mixup_cfg.get("synthetic_hard_negatives", False))
    if (
        margins is None
        and koleo_weight <= 0
        and memory is None
        and not mixup_positives
        and not mixup_negatives
        and prototype_weight <= 0
    ):
        return loss
    return CompositeMetricLoss(
        loss,
        margins=margins,
        koleo_weight=koleo_weight,
        memory=memory,
        mixup_positives=mixup_positives,
        mixup_negatives=mixup_negatives,
        prototype_weight=prototype_weight,
    )


def _origin_weight_config(config: dict[str, object]) -> tuple[float, float, int | None]:
    raw = config.get("origin_weights")
    same, cross = 1.0, 1.0
    if raw:
        if not isinstance(raw, dict):
            raise ValueError("loss.origin_weights must be a mapping")
        same = float(raw.get("same", 1.0))
        cross = float(raw.get("cross", 1.0))
        if same < 0 or cross < 0:
            raise ValueError("loss.origin_weights.same and loss.origin_weights.cross must be non-negative")
    anchor = config.get("anchor_origin")
    if anchor in (None, "", "both"):
        return same, cross, None
    token = str(anchor).strip().lower()
    if token == "modern":
        return same, cross, 1
    if token == "historic":
        return same, cross, 0
    raise ValueError("loss.anchor_origin must be 'modern', 'historic', or 'both'")
