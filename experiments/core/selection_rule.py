"""Epoch and configuration selection on the Met splits.

The Met checkpoint split chooses the epoch. A new checkpoint replaces the
stored one when its mean pairs completeness at k = 5, 10, 20, 40, 60, and 80
is higher by more than the configured margin. Within that margin it replaces
the stored one only when its reduction ratio at 95% pairs completeness is
higher by more than 0.01. The sweep uses margin 0.005. The final runs use
margin 10^{-5}. A tie keeps the earlier epoch.

The Met configuration split keeps the configuration with the highest mean
pairs completeness over the same budgets. Pass margin 0 for that comparison.
"""

from __future__ import annotations

from typing import Any

KS = [5, 10, 20, 40, 60, 80]
# Sweep checkpoint margin, 0.005 on a 0–1 score.
PC_TIE = 0.005
# Final-run checkpoint margin, 0.00001 on a 0–1 score.
PC_TIE_TIGHT = 0.00001
RR_TIE = 0.01
# v2: checkpoints are chosen with this same five-step funnel. v1 scores used
# checkpoints chosen by the older precision rule.
RULE_ID = "mean_pc_k60_rr95_v2"


def _steps(pc_tie: float) -> tuple[tuple[str, str, float, str], ...]:
    return (
        ("mean_pc", "higher", float(pc_tie), "Mean pairs completeness"),
        ("rr95", "higher", RR_TIE, "Reduction ratio at 95% pairs completeness"),
        ("lora_lr", "lower", 0.0, "LoRA learning rate"),
        ("head_lr", "lower", 0.0, "Head learning rate"),
        ("epoch", "lower", 0.0, "Epoch"),
    )


def rule_spec(pc_tie: float | None = None) -> dict[str, Any]:
    """The steps the dashboard and the training loop both apply."""
    margin = PC_TIE if pc_tie is None else float(pc_tie)
    return {
        "id": RULE_ID,
        "k": list(KS),
        "pc_tie": margin,
        "steps": [
            {"key": key, "direction": direction, "margin": step_margin, "title": title}
            for key, direction, step_margin, title in _steps(margin)
        ],
    }


def _within(leader: float, value: float, direction: str, margin: float) -> bool:
    if direction == "higher":
        return leader - value <= margin + 1e-12
    return value - leader <= margin + 1e-12


def replaces(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    pc_tie: float | None = None,
) -> bool:
    """Return whether candidate is the sole survivor of the two-checkpoint funnel."""
    ranking = funnel(
        [
            {**candidate, "id": "candidate"},
            {**incumbent, "id": "incumbent"},
        ],
        pc_tie=pc_tie,
    )
    return ranking["winner"] == "candidate" and not ranking["tied"]


def funnel(models: list[dict[str, Any]], pc_tie: float | None = None) -> dict[str, Any]:
    """Apply the epoch rule and return the surviving ids at each step.

    ``pc_tie`` is the checkpoint margin. The sweep passes 0.005 and the final
    runs pass 10^{-5}. Configuration selection passes 0, so the highest mean
    pairs completeness remains.
    """
    steps = _steps(PC_TIE if pc_tie is None else float(pc_tie))
    if not models:
        return {"winner": None, "tied": False, "stages": []}
    remaining = list(models)
    stages: list[dict[str, Any]] = []
    for key, direction, margin, title in steps:
        present = [model for model in remaining if model.get(key) is not None]
        if not present:
            stages.append(
                {
                    "key": key,
                    "title": title,
                    "direction": direction,
                    "margin": margin,
                    "leader": None,
                    "alive_ids": [str(model["id"]) for model in remaining],
                }
            )
            break
        values = [float(model[key]) for model in present]
        leader = max(values) if direction == "higher" else min(values)
        alive = [
            model
            for model in present
            if _within(leader, float(model[key]), direction, margin)
        ]
        stages.append(
            {
                "key": key,
                "title": title,
                "direction": direction,
                "margin": margin,
                "leader": leader,
                "alive_ids": [str(model["id"]) for model in alive],
            }
        )
        remaining = alive
    tied = len(remaining) > 1
    if not remaining:
        winner = None
    elif tied:
        remaining.sort(
            key=lambda model: (
                -float(model["mean_pc"]),
                -float(model["rr95"]),
                float(model.get("lora_lr") if model.get("lora_lr") is not None else 1e9),
                float(model.get("head_lr") if model.get("head_lr") is not None else 1e9),
                float(model.get("epoch") if model.get("epoch") is not None else 1e9),
                str(model["id"]),
            )
        )
        winner = str(remaining[0]["id"])
    else:
        winner = str(remaining[0]["id"])
    return {"winner": winner, "tied": tied, "stages": stages}
