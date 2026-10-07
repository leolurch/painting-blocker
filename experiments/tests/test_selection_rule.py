"""Order-independent fine-tuned selection funnel."""

from __future__ import annotations

import unittest

from experiments.core.selection_rule import PC_TIE, PC_TIE_TIGHT, RULE_ID, funnel, replaces, rule_spec


def _model(
    model_id: str,
    mean_pc: float,
    rr95: float,
    epoch: int = 1,
    head_lr: float = 3e-4,
    lora_lr: float = 3e-5,
) -> dict:
    return {
        "id": model_id,
        "mean_pc": mean_pc,
        "rr95": rr95,
        "epoch": epoch,
        "head_lr": head_lr,
        "lora_lr": lora_lr,
    }


class SelectionRuleTest(unittest.TestCase):
    def test_mean_pc_outside_the_margin_eliminates_the_rest(self) -> None:
        result = funnel(
            [
                _model("low", 0.90, 0.80),
                _model("high", 0.91, 0.10),
            ]
        )
        self.assertEqual(result["winner"], "high")
        self.assertEqual(result["stages"][0]["alive_ids"], ["high"])

    def test_within_mean_pc_margin_uses_rr_at_95(self) -> None:
        result = funnel(
            [
                _model("precise", 0.910000, 0.20),
                _model("efficient", 0.909995, 0.40),
            ]
        )
        self.assertEqual(result["winner"], "efficient")
        self.assertCountEqual(result["stages"][0]["alive_ids"], ["precise", "efficient"])
        self.assertEqual(result["stages"][1]["alive_ids"], ["efficient"])

    def test_rr_within_one_point_keeps_the_lower_epoch(self) -> None:
        result = funnel(
            [
                _model("later", 0.90, 0.50, epoch=3),
                _model("earlier", 0.90, 0.505, epoch=1),
            ]
        )
        self.assertEqual(result["winner"], "earlier")
        self.assertEqual(result["stages"][4]["key"], "epoch")
        self.assertEqual(result["stages"][4]["alive_ids"], ["earlier"])

    def test_lower_lora_learning_rate_comes_before_head_and_epoch(self) -> None:
        result = funnel(
            [
                _model("high-lora", 0.90, 0.50, epoch=1, head_lr=1e-4, lora_lr=3e-5),
                _model("low-lora", 0.90, 0.50, epoch=4, head_lr=3e-4, lora_lr=1e-5),
            ]
        )
        self.assertEqual(result["winner"], "low-lora")
        self.assertEqual(result["stages"][2]["key"], "lora_lr")
        self.assertEqual(result["stages"][2]["alive_ids"], ["low-lora"])

    def test_same_lora_rate_keeps_the_lower_head_learning_rate(self) -> None:
        result = funnel(
            [
                _model("fast-head", 0.90, 0.50, epoch=1, head_lr=3e-4),
                _model("slow-head", 0.90, 0.50, epoch=1, head_lr=1e-4),
            ]
        )
        self.assertEqual(result["winner"], "slow-head")
        self.assertEqual(result["stages"][3]["key"], "head_lr")
        self.assertEqual(result["stages"][3]["alive_ids"], ["slow-head"])

    def test_funnel_is_independent_of_input_order(self) -> None:
        models = [
            _model("a", 0.900000, 0.20, epoch=1, head_lr=3e-4),
            _model("b", 0.899995, 0.40, epoch=2, head_lr=1e-4),
            _model("c", 0.899000, 0.60, epoch=1, head_lr=1e-5),
        ]
        forward = funnel(models)
        backward = funnel(list(reversed(models)))
        self.assertEqual(forward["winner"], backward["winner"])
        self.assertEqual(forward["winner"], "c")
        tight = funnel(models, pc_tie=PC_TIE_TIGHT)
        self.assertEqual(tight["winner"], "b")

    def test_later_epoch_does_not_replace_a_tied_checkpoint(self) -> None:
        incumbent = {"mean_pc": 0.90, "rr95": 0.50}
        tied = {"mean_pc": 0.900008, "rr95": 0.509}
        better_pc = {"mean_pc": 0.90002, "rr95": 0.10}
        better_rr = {"mean_pc": 0.899995, "rr95": 0.52}
        self.assertFalse(replaces(tied, incumbent))
        self.assertFalse(replaces(better_pc, incumbent))
        self.assertTrue(replaces(better_pc, incumbent, pc_tie=PC_TIE_TIGHT))
        self.assertTrue(replaces(better_rr, incumbent))
        self.assertFalse(replaces(incumbent, better_rr))

    def test_checkpoint_tie_uses_lora_then_head_then_epoch(self) -> None:
        base = {"mean_pc": 0.90, "rr95": 0.50, "lora_lr": 3e-5, "head_lr": 3e-4, "epoch": 1}
        later = {**base, "epoch": 3}
        lower_head = {**base, "head_lr": 1e-4, "epoch": 4}
        lower_lora = {**base, "lora_lr": 1e-5, "head_lr": 1e-3, "epoch": 5}
        self.assertFalse(replaces(later, base))
        self.assertTrue(replaces(base, later))
        self.assertTrue(replaces(lower_head, base))
        self.assertTrue(replaces(lower_lora, lower_head))

    def test_rule_spec_lists_the_five_steps_in_order(self) -> None:
        spec = rule_spec()
        self.assertEqual(spec["id"], RULE_ID)
        self.assertEqual(spec["k"], [5, 10, 20, 40, 60, 80])
        self.assertEqual(
            [step["key"] for step in spec["steps"]],
            ["mean_pc", "rr95", "lora_lr", "head_lr", "epoch"],
        )
        self.assertEqual(spec["steps"][0]["margin"], PC_TIE)
        self.assertEqual(spec["steps"][0]["margin"], 0.005)
        self.assertEqual(rule_spec(PC_TIE_TIGHT)["steps"][0]["margin"], PC_TIE_TIGHT)
        self.assertEqual(spec["steps"][1]["margin"], 0.01)


if __name__ == "__main__":
    unittest.main()
