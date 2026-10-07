import unittest

from experiments.core.checkpoint_naming import (
    checkpoint_name_from_finetuning,
    checkpoint_run_name,
    parse_checkpoint_name,
)


def _config(*, architecture="loramlp", sampler="pk"):
    if architecture == "clsmlp":
        model = {
            "type": "dinov3_projection",
            "projection_hidden_dim": 1024,
            "dropout": 0.1,
        }
        optimizer = {"head_lr": 1e-3}
    else:
        head = (
            {"type": "linear_projection", "dropout": 0.1}
            if architecture == "loralin"
            else {"type": "mlp_projection", "hidden_dim": 1024, "dropout": 0.1}
        )
        model = {
            "type": "dinov3_lora_qv_projection",
            "lora": {"rank": 4, "alpha": 8},
            "head": head,
        }
        optimizer = {"lr_lora": 3e-5, "lr_head": 3e-4}
    return {
        "model": model,
        "sampler": {"type": sampler, "classes_per_batch": 32, "images_per_class": 4},
        "optimizer": optimizer,
        "train": {"epochs": 15},
    }


class CheckpointNamingTest(unittest.TestCase):
    def test_lora_mlp_name(self):
        self.assertEqual(
            checkpoint_name_from_finetuning(_config()),
            "loramlp__p-32__k-4__r-4__la-8__llr-3e-5__hd-1024__hlr-3e-4__hdo-0p1__sam-pk__ep-15",
        )

    def test_linear_and_cls_names_only_include_relevant_fields(self):
        self.assertEqual(
            checkpoint_name_from_finetuning(_config(architecture="loralin")),
            "loralin__p-32__k-4__r-4__la-8__llr-3e-5__hlr-3e-4__sam-pk__ep-15",
        )
        self.assertEqual(
            checkpoint_name_from_finetuning(_config(architecture="clsmlp")),
            "clsmlp__p-32__k-4__hd-1024__hlr-1e-3__hdo-0p1__sam-pk__ep-15",
        )

    def test_role_stratified_sampler_and_unique_run_suffix(self):
        self.assertEqual(
            checkpoint_run_name(_config(sampler="role_stratified_pk"), "2369999-2"),
            "loramlp__p-32__k-4__r-4__la-8__llr-3e-5__hd-1024__hlr-3e-4__hdo-0p1__sam-rspk__ep-15__run-2369999-2",
        )

    def test_parser_mines_semantic_title_from_model_id(self):
        name = checkpoint_run_name(_config(), "2369999-2")
        parsed = parse_checkpoint_name(f"local/finetuned/experiment/{name}/best")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["architecture"], "loramlp")
        self.assertEqual(parsed["llr"], "3e-5")
        self.assertEqual(parsed["run"], "2369999-2")
        self.assertEqual(parsed["title"], checkpoint_name_from_finetuning(_config()))

    def test_lora_span_is_part_of_the_run_name(self):
        config = _config()
        config["model"]["lora"]["span"] = "first16"
        self.assertIn(
            "__lb-first16__",
            checkpoint_name_from_finetuning(config),
        )
        config["model"]["lora"]["span"] = "all"
        self.assertIn("__lb-all__", checkpoint_name_from_finetuning(config))
        config["model"]["lora"]["span"] = "last24"
        self.assertIn("__lb-last24__", checkpoint_name_from_finetuning(config))

    def test_unsupported_architecture_is_not_named(self):
        config = _config()
        config["model"] = {"type": "resnet50_gem_projection"}
        self.assertIsNone(checkpoint_name_from_finetuning(config))


if __name__ == "__main__":
    unittest.main()
