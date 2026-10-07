# Conference paper artifacts

Code, labels, and reviews for *Synthetically Adapted Visual Embeddings for Blocking Painting Duplicates*. Image files are not included. Run every command from this directory.

## Setup

- GPU training: `pip install -r experiments/requirements.txt` (Python 3.11+). The content check needs PyYAML, Pillow, NumPy, and pytest.
- `python scripts/check_artifact.py` checks the packaged labels and configs. It does not download or train.
- Each experiment YAML sets `dataset.path` and `split.file`. Point those at the images you downloaded or generated.

## Download

- Art Institute, Cleveland, National Gallery, Rijksmuseum. One open-access photograph per painting is the SynView source:

```bash
python experiments/datasets/data_artic/download_artic_paintings.py --out-dir data/images/artic
python experiments/datasets/data_cleveland/download_cleveland_paintings.py --out-dir data/images/cleveland
python experiments/datasets/data_nationalgallery/download_nga_paintings.py --out-dir data/images/nga
python experiments/datasets/data_rijksmuseum/download_rijks_paintings.py --out-dir data/images/rijks
```

- Wikidata 1.5. `data/wikidata/labels.csv` is the full labeled list: 1089 images of 211 paintings, with the Commons download URL, role, class id, and the ten partition halves. `python scripts/download_wikidata_labels.py` saves each file into `data/datasets/wikidata-1.5/images` and writes `dataset.db` beside it. That directory is the dataset the evaluation configs use.
- Met selection photographs. `python experiments/datasets/met_benchmark/download_met_benchmark.py --help`, then the same command with `--execute`. The checkpoint and configuration halves are `experiments/configs/splits/met_benchmark/`.
- Lost Art photographs are not redistributable. `data/lostart/distractor_files.csv` lists 27564 files from 19771 reports. `report_url` is the report page.

## SynView

- 11094 paintings. Each keeps its museum photograph and gets three modern views and four historic views (archival, print, cropped record, framed photo), all at the medium preset. `data/syn/class_split.csv` says which painting is train, validation, or test.
- Recipe: `experiments/configs/synthetic/hard_synth_medium_o1m3h4_v1.yml`.
- Put one source image per painting in a directory, then:

```bash
python experiments/datasets/image-distorter/generate_synth_dataset.py data/images/sources \
  --output-root data/synview \
  --source-kind image-dir \
  --recipe hard-synth-v1 \
  --recipe-config experiments/configs/synthetic/hard_synth_medium_o1m3h4_v1.yml \
  --modern-variants 3 --historic-variants 4 \
  --seed 42 --max-output-dimension 1024
```

- Frames, reflection overlays, and wall textures are not in this archive. Pass `--frames-horizontal`, `--overlays`, and `--wall-textures`. Without them the views that use a frame or a wall are not the views in the paper.
- Omitting the variant counts uses the same three modern views and four historic views. The modern photograph crops at most 6% and uses JPEG quality 75 to 100; an archival view is aged in 90% of cases and sepia-toned in 45%; a print is placed on a paper document in 75% of cases; a cropped record removes at most 25%.
- `experiments/datasets/build_met_deduped_medium7.py` staged the deduplicated sources, wrote the generation plan, and recorded the class split. Point its dataset paths at the local image directories.

## Train and evaluate

- Selected model, seeds 42, 43, and 44:

```bash
python -m experiments.cli train --experiment experiments/configs/experiments/finetune_dinov3_vithplus_met_paint_winner_followup_v1/none.yml
```

- Same directory: `historic_archival.yml`, `historic_print.yml`, `historic_cropped_record.yml`, `historic_framed_photo.yml`, and `modern_generated.yml` each drop one generated role.
- Controls: `finetune_dinov3_vithplus_met_paint_head_only_v1`, `finetune_dinov3_vithplus_met_paint_modern_only_v1`, `finetune_dinov3_vithplus_met_paint_originals_aug_v1`.
- The 64-setting sweep is `finetune_dinov3_vithplus_met_paint_wave1_v1`. SigLIP2 is `finetune_siglip2_giant_met_paint_wave1_v1` and `wave2_v1`. CLIP, OpenCLIP, and ResNet-50 each have four settings under `finetune_transfer_*`.
- Frozen and adapted Wikidata runs: `python -m experiments.cli run --experiment <eval.yml>`. An eval that loads a trained checkpoint also needs `--checkpoint`.
- The epoch and the configuration are chosen on the Met halves by `experiments/core/selection_rule.py`. Wikidata does not choose either.
- `writeups/tooling/build_paper_numbers.py` rebuilds the reported numbers from the Wikidata 1.5 evaluation outputs. `writeups/tooling/risk_control.py` is the certified operating-point rule.

## Labels and reviews

- Wikidata role labels, the self-deduplication edits (`data/wikidata/identity_edits.json`), and the crops removed by the frozen-ranker cap (`removed_crops.json`).
- `data/dedup/` holds the manual decisions. The museum file is one row per reviewed pair. Wikidata and Lost Art keep the decision log and the pair table.
- To propose pairs again: `experiments/datasets/deduplication/README.md`. The blocker is frozen DINOv3-7B and the verifier is LightGlue with the classifier `image_matching/matching/classifier.pkl` from https://github.com/HPI-Information-Systems/smARTmatch. Save that file as `deduplication/classifier.pkl`. The recorded reviews are already in `data/dedup/`.

## Not in this archive

- Photograph files. Use the downloaders and the Commons URLs.
- Lost Art image files.
- Generator frame, overlay, and wall-texture assets.
- The per-image SynView split (the class assignment is `data/syn/class_split.csv`) and the combined Wikidata-plus-Lost-Art split.
- Checkpoints and evaluation outputs.
