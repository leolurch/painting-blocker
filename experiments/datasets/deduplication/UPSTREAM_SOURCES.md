# Upstream sources and provenance

This standalone pipeline treats `smARTmatch` as a behavioural reference, not a package to
transplant. The verifier classifier is not in this archive. No smARTmatch Python module is copied; the
DINO embedding adapter is reimplemented as a local copy of *this repository's* adapter, and the
SuperPoint/LightGlue verification reuses this repository's own `lightglue-verified` code.

## Classifier

The verifier classifier is `image_matching/matching/classifier.pkl` in https://github.com/HPI-Information-Systems/smARTmatch (commit `723cfaf1feb29ae478332a59f65788a5a33b775e`).
Its SHA-256 is `f9b313a44d86b7352583cd365d2af9181ec018708768aeaeaa4c762ebe7ff24d`.
Save that file as `deduplication/classifier.pkl` before running verification.
It was produced under `scikit-learn==1.8.0` / `joblib==1.5.3`.

## Reference files consulted / locally vendored adapter

| Reference | Behaviour preserved | Local implementation |
|---|---|---|
| `experiments/core/validate/embedding_adapters/dino_adapter.py` + `experiments/core/validate/embedding_geometry.py` | DINOv3 `AutoImageProcessor`/`AutoModel`, bicubic letterbox resize-and-pad to 512 using ImageNet-mean padding, disabled processor geometry, CLS-token pooling, L2 normalization, revision pinning | `deduplication/embedding.py` is a trimmed, self-contained local adapter derived from these repository files, so later experiment-adapter changes cannot alter deduplication outcomes. No full upstream module was copied. |
| `smARTmatch: blocking/search.py` | normalized matmul + top-k descending ordering | `deduplication/retrieval.py` |
| [https://github.com/HPI-Information-Systems/smARTmatch](https://github.com/HPI-Information-Systems/smARTmatch) `image_matching/matching/classifier.pkl` | 17 score-summary features + classifier predict/predict_proba | download into `deduplication/classifier.pkl` |

## Reused (imported, not copied) from this repository

| Module | Reused symbols |
|---|---|
| `experiments/datasets/lightglue-verified/image_matching/keypoint/lightglue_score_pseudo_pairs.py` | `classify_scores`, `_match_feature_tensors`, `_load_feature_file`, and tensor-tree CPU/device conversion helpers |
| `experiments/core/db.py`, `experiments/core/config_schema.py` | export-time dataset validation (`ensure_schema`, `dataset_counts`, `validate_dataset_config`) |

The standalone environment pins LightGlue to commit
`eb42fee2d71449efb0aa5c10549752b5d75384d8` (resolved 2026-07-10). GPU parity against the
existing `lightglue-verified` environment remains an operational preflight before the corpus run.

## Enforcement

`tests/test_provenance.py` verifies that the classifier is referenced at that repository and is not
included in this archive, and that no stage before `export-dataset` creates a SQLite database.
