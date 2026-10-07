# Conference paper artifacts

Code, labels, and reviews for *Synthetically Adapted Visual Embeddings for Blocking Painting Duplicates*. Image files are not included. Run every command from this directory.

## Setup

- GPU training: `pip install -r experiments/requirements.txt` (Python 3.11+). The content check needs PyYAML, Pillow, NumPy, and pytest.
- `python scripts/check_artifact.py` checks the packaged labels and configs. It does not download or train.
- Each experiment YAML sets `dataset.path` and `split.file`. Point those at the images you downloaded or generated.

## Download

- Art Institute, Cleveland, National Gallery, Rijksmuseum. One open-access photograph per painting is the SynView source:

```bash
python experiments/datasets/data_artic/download_artic_paintings.py --ids-from data/syn/class_split.csv --out-dir data/images/artic
python experiments/datasets/data_cleveland/download_cleveland_paintings.py --ids-from data/syn/class_split.csv --out-dir data/images/cleveland
python experiments/datasets/data_nationalgallery/download_nga_paintings.py --ids-from data/syn/class_split.csv --out-dir data/images/nga
python experiments/datasets/data_rijksmuseum/download_rijks_paintings.py --ids-from data/syn/class_split.csv --out-dir data/images/rijks
```

- Wikidata 1.5. `data/wikidata/labels.csv` is the full labeled list: 1089 images of 211 paintings, with the Commons download URL, role, class id, and the ten partition halves. `python scripts/download_wikidata_labels.py` saves each file into `data/datasets/wikidata-1.5/images` and writes `dataset.db` beside it. That directory is the dataset the Wiki evaluations use.
- Met selection photographs. `python experiments/datasets/met_benchmark/download_met_benchmark.py --root data/datasets/met_benchmark --execute`. The checkpoint and configuration halves are `experiments/configs/splits/met_benchmark/`. The selection eval is `experiments/configs/experiments/eval_met_painting_model_selection_from_finetuning_v1.yml`.
- Lost Art photographs are not redistributable. `data/lostart/distractor_files.csv` lists 27564 files from 19771 reports. `report_url` is the report page. Place those files, together with the Wikidata images, in `data/datasets/wikidata1.5-lostart-distractors/images`, then rebuild the test splits:

```bash
python experiments/datasets/wikidata_lostart_distractors/build_splits.py --names data/lostart/distractor_files.csv
```

## SynView

- 11094 paintings from the Art Institute of Chicago, the Cleveland Museum of Art, the National Gallery of Art, and the Rijksmuseum. Each keeps its museum photograph and gets three modern views and four historic views (archival, print, cropped record, framed photo). `data/syn/class_split.csv` says which painting is train, validation, or test.
- Recipe: `experiments/configs/synthetic/hard_synth_medium_o1m3h4_v1.yml`.
- Stage the downloaded photographs under the class ids, generate the views into the training dataset directory, then write the per-image split:

```bash
python experiments/datasets/synview_sources.py
python experiments/datasets/image-distorter/generate_synth_dataset.py data/images/sources \
  --output-root data/datasets/museum_deduplication_hard_synth_v1_met_deduped_medium_o1m3h4_v1 \
  --source-kind image-dir \
  --recipe hard-synth-v1 \
  --recipe-config experiments/configs/synthetic/hard_synth_medium_o1m3h4_v1.yml \
  --modern-variants 3 --historic-variants 4 \
  --seed 42 --max-output-dimension 1024
python experiments/datasets/write_synview_split.py \
  --dataset-root data/datasets/museum_deduplication_hard_synth_v1_met_deduped_medium_o1m3h4_v1
```

- Frames, reflection overlays, and wall textures are not in this archive. Pass `--frames-horizontal`, `--overlays`, and `--wall-textures`. Without them the views that use a frame or a wall are not the views in the paper.
- Omitting the variant counts uses the same three modern views and four historic views. The modern photograph crops at most 6% and uses JPEG quality 75 to 100; an archival view is aged in 90% of cases and sepia-toned in 45%; a print is placed on a paper document in 75% of cases; a cropped record removes at most 25%. A historic view loses resolution with probability 0.40 to 0.75 and uses JPEG quality 40 to 95.
- `write_synview_split.py` leaves an existing split file in place.

## Train and evaluate

- Selected model, seeds 42, 43, and 44:

```bash
python -m experiments.cli train --experiment experiments/configs/experiments/finetune_dinov3_vithplus_met_paint_winner_followup_v1/none.yml
```

- Same directory: `historic_archival.yml`, `historic_print.yml`, `historic_cropped_record.yml`, `historic_framed_photo.yml`, and `modern_generated.yml` each drop one generated role.
- Controls: `finetune_dinov3_vithplus_met_paint_head_only_v1`, `finetune_dinov3_vithplus_met_paint_modern_only_v1`, `finetune_dinov3_vithplus_met_paint_originals_aug_v1`.
- The 64-setting sweep is `finetune_dinov3_vithplus_met_paint_wave1_v1`. SigLIP2 is `finetune_siglip2_giant_met_paint_wave1_v1` and `wave2_v1`. CLIP, OpenCLIP, and ResNet-50 each have four settings under `finetune_transfer_*`.
- Frozen and adapted Wikidata runs: `python -m experiments.cli run --experiment <eval.yml>`. An eval that loads a trained checkpoint also needs `--checkpoint`.
- The Met checkpoint split chooses the epoch. A new checkpoint replaces the stored one when its mean pairs completeness at k = 5, 10, 20, 40, 60, and 80 is higher by more than the margin in that config: 0.005 in the sweep, and 10^-5 in the final runs. Within the margin it replaces the stored one only when its reduction ratio at 95% pairs completeness is higher by more than 0.01. The Met configuration split keeps the configuration with the highest mean pairs completeness. Wikidata does not choose either. The rule is `experiments/core/selection_rule.py`.
- `writeups/tooling/build_paper_numbers.py` rebuilds the reported numbers from the WiLo evaluation outputs built from the Wikidata 1.5 catalog. `writeups/tooling/risk_control.py` fits those operating points by conformal risk control.

## Labels and reviews

- Wikidata role labels, the self-deduplication edits (`data/wikidata/identity_edits.json`), and the crops removed by the frozen-ranker cap (`removed_crops.json`).
- `data/dedup/` holds the manual decisions. The museum file is one row per reviewed pair. Wikidata and Lost Art keep the decision log and the pair table.
- To propose pairs again: `experiments/datasets/deduplication/README.md`. The blocker is frozen DINOv3-7B and the verifier is LightGlue with the classifier `image_matching/matching/classifier.pkl` from https://github.com/HPI-Information-Systems/smARTmatch. Save that file as `deduplication/classifier.pkl`. The recorded reviews are already in `data/dedup/`.

## Not in this archive

- Photograph files. Use the downloaders and the Commons URLs.
- Lost Art image files. The report identifiers are in `data/lostart/distractor_files.csv`.
- Generator frame, overlay, and wall-texture assets.
- The per-image SynView split and the Wikidata-plus-Lost-Art splits. Both are written by the commands above. The class assignment is `data/syn/class_split.csv`.
- Checkpoints and evaluation outputs.
