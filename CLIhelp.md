# CLI help

Every command of the `id_classifier` package: what it does, its options, and what it writes.
**Rule:** every new or changed command is documented here in the same step.

## 0. Before any command: open the environment

PyTorch (needed by YOLO) is blocked on Windows by Smart App Control, so everything runs in WSL Ubuntu.

```bash
wsl -d Ubuntu                                    # from PowerShell / Windows Terminal
source ~/venvs/id_classifier/bin/activate        # the project's Python environment
cd /mnt/c/Users/LENOVO/Downloads/Valify_Project1_OCR
```

- The NFS share is at `/mnt/nfs` inside WSL.
- Output under `data/` is gitignored and never committed. Windows sees it at
  `C:\Users\LENOVO\Downloads\Valify_Project1_OCR\data\`.
- Every command prints help with `-h`, e.g. `python -m id_classifier sweep-detector -h`.
- One-time setup (already done): `python3 -m venv ~/venvs/id_classifier`, then
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` and `pip install -e ".[yolo,dev]"`.

## Overview

| Command | What it does | Reads | Writes |
| --- | --- | --- | --- |
| `pytest` | Runs the automated tests (no real images, no real model) | `tests/` | nothing |
| `build-ground-truth` | Builds the labeled answer key from the NFS folders | NFS zips | `data/ground_truth.csv` |
| `evaluate` | Runs and compares configurations against the answer key | experiment YAML + ground truth | `data/runs/<name>_<time>/` |
| `classify` | Runs the pipeline and stores results in the database | YAML with a `classify` section | SQLite/Postgres DB |
| `preview-detector` | Draws a detector's boxes on a few images for a visual check | YAML with a `preview` section | `data/detect_preview/` (**real ID images**) |
| `sweep-detector` | Compares detector prompts x thresholds: detection counts per folder | YAML with a `sweep` section | `data/sweeps/sweep_<time>/` (numbers only) |
| `find-duplicates` | Finds exact and near (re-saved) copies, within and across folders | NFS zips | `data/duplicates/` (numbers; real images only with `--review-images`) |
| `train-cls` | Trains the YOLO26 classification models for configuration B (5 folds + full) | YAML with a `train_cls` section + ground truth | `models/yolo26_cls/`, crops in `data/cls_dataset/` (**real ID images**) |

## Tests

```bash
python -m pytest -v                         # all tests
python -m pytest -v tests/test_detectors.py # one file
```

Expected: all pass. The tests use tiny generated images and fake models (`tests/fakes.py`). They check our
own code (crops, routing, storage, reports), not the real models.

## build-ground-truth

Writes the answer key for `evaluate`. Every labeled NFS folder holds one document type of one country
(e.g. `tun_nid_ocr` = Tunisian national ID), and the file name gives the side (front/back).
Only **distinct pictures** are kept: exact copies and the same card re-saved or re-shot (perceptual
distance <= 40) appear once. Optional generated **negatives** (blank, noise, gradient images, label
`none`) check that images without a document go to review.

```bash
python -m id_classifier build-ground-truth --root /mnt/nfs --per-folder 150 --negatives 20
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--root` | `/mnt/nfs` | NFS root as this machine sees it |
| `--out` | `data/ground_truth.csv` | output CSV (negatives go to `negatives/` next to it) |
| `--per-folder` | all | at most N distinct pictures per folder (random, same seed = same pictures) |
| `--seed` | 42 | same seed = same sample |
| `--negatives` | 0 | also generate N no-document images |

The labeled folders are listed in `FOLDER_LABELS` in `id_classifier/ground_truth.py`. `epassport` is not
one of them: its zips hold only the chip's face photo (`dg2_face.jp2`), not an image of the passport.

## evaluate

Runs every configuration (detector + classifiers) listed in an experiment YAML on the ground truth,
scores it, and picks the winner with the selection rule written in the YAML.

```bash
python -m id_classifier evaluate --config configs/nfs_eval.yaml
```

| Option | Meaning |
| --- | --- |
| `--config` | experiment YAML (required) |

Writes `data/runs/<experiment>_<time>/`: `comparison.md` (read this first), `comparison.csv`,
`per_country.csv`, `confusion_<config>.csv/.md/.png`, `predictions_<config>.csv`, `experiment.yaml`.
Note: it stops with "no configurations to run" until a real classifier exists.

## classify

Runs the pipeline on a source and stores images, regions, predictions and reviews in the database.
Rerunning with the same `--run-id` adds nothing (safe to repeat).

```bash
python -m id_classifier classify --config <yaml with a classify section> [--run-id <id>]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--config` | (required) | YAML with a `classify` section: `source`, `detector`, `classifiers`, `database_url` |
| `--run-id` | `cli-<timestamp>` | reuse an id to rerun safely |

## preview-detector

Runs a detector on a few random images per NFS folder and saves, for each image, a copy with the boxes
drawn (`<txn>_<side>_boxes.jpg`, labels `<index>: <confidence>`), every crop (`<txn>_<side>_crop<i>.jpg`),
and `summary.csv` (boxes, confidences, ms per image).

```bash
python -m id_classifier preview-detector --config configs/yolo_preview.yaml
python -m id_classifier preview-detector --config configs/yolo_preview.yaml --folders dza_nid_ocr --per-folder 10
```

| Option | Meaning |
| --- | --- |
| `--config` | YAML with a `preview` section (required), e.g. `configs/yolo_preview.yaml` |
| `--root` | NFS root, overrides the YAML |
| `--folders` | folders to sample, overrides the YAML |
| `--per-folder` | images per folder, overrides the YAML |

Images are sampled exactly like `sweep-detector`: the same `seed` and `per_folder` give the same images,
and duplicate uploads are skipped (`unique_only: true` in the YAML). So a preview can be compared one to
one with a sweep's `per_image.csv`. `summary.csv` also lists each image's hash.

**Privacy:** this writes copies of real ID images under `data/detect_preview/` (the exact folder is
`output_dir` in the YAML). Delete it after looking. Only the folders listed in the YAML are read.

## sweep-detector

Answers "which prompts and thresholds make YOLOE find the documents?" without writing any image. Every
prompt set becomes one YOLOE detector. Each runs once per image at the lowest threshold, and the higher
thresholds are counted from the same confidences. Duplicate uploads are skipped by hash
(`unique_only: true`).

```bash
python -m id_classifier sweep-detector --config configs/detector_sweep.yaml
```

| Option | Meaning |
| --- | --- |
| `--config` | YAML with a `sweep` section (required), e.g. `configs/detector_sweep.yaml` |
| `--root` | NFS root, overrides the YAML |
| `--folders` | folders to sample, overrides the YAML |
| `--per-folder` | distinct images per folder, overrides the YAML |

Edit `prompt_sets` and `thresholds` in the YAML to try other prompts. Writes `data/sweeps/sweep_<time>/`:
- `sweep.md`: the table to read. Cells are `detected/images`, with `(n)` = images with 2+ boxes.
- `sweep.csv`: the same numbers.
- `per_image.csv`: every confidence per image, to look up a single case.

## find-duplicates

Finds images uploaded more than once in the labeled folders, and also across folders (an image in two
different folders means one of its labels is wrong). There are two kinds of copies:
- **exact**: the same bytes (same sha256 of `original.jpeg`)
- **near**: the same picture re-saved or re-compressed. The bytes differ, but the *perceptual hash* is
  almost the same. The perceptual hash is a 256-bit fingerprint of how the image looks: the image is
  shrunk to 17x16 grey pixels and each bit says "is this pixel brighter than its right neighbour?". The
  distance is the number of differing bits: 0 = looks identical, about 128 = unrelated pictures.

```bash
python -m id_classifier find-duplicates
python -m id_classifier find-duplicates --folders dza_nid_ocr mar_nid_ocr --review-images
python -m id_classifier find-duplicates --max-distance 6 --review-distance 30
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--root` | `/mnt/nfs` | NFS root |
| `--folders` | the labeled ID folders | folders to check |
| `--out` | `data/duplicates` | output folder |
| `--max-distance` | 40 | distance (of 256 bits) at or below which two images count as copies (chosen by eye: above 40 it is about 50/50 same card or not) |
| `--review-distance` | 40 | every pair up to this distance is listed, to check the threshold |
| `--review-images` | off | also draw every close pair side by side in `<out>/review/` (**real ID images**) |

Writes to `--out`:
- `summary.csv`: per folder, zips, `distinct_exact` (bytes) and `distinct_near` (look), unreadable.
- `groups.csv`: one row per picture uploaded more than once. `kind` is exact/near, `max_distance` is the
  largest distance inside the group, plus every upload.
- `distances.csv`: how many image pairs sit at each distance (the gap shows where to put the threshold).
- `close_pairs.csv`: every pair up to `--review-distance`, with `duplicate` True/False.
- `review/` (with `--review-images`): `<distance>_<dup|diff>_<a>__<b>.jpg`, both images side by side.
  Check that every `dup` really is the same picture and every `diff` really is different.

The same distance is used by `sweep-detector` and `preview-detector` (`unique_only: true`) to skip copies
when sampling. Slow on the full share: it reads and decodes every zip over NFS (about 15 minutes).

## train-cls

Trains configuration B's classifier: a small YOLO26 classification model that learns the label triple
(e.g. `national_id__TN__front`) from our own crops. Every ground-truth image is cropped by the detector in
the YAML, then 5-fold cross-validation trains 5 models, each without one fifth of the crops. At
`evaluate`, each image is classified by the model that never saw it (found by a hash of the crop's
pixels), so the score is honest. A 6th model ("full", trained on everything) is used for new images.

```bash
python -m id_classifier train-cls --config configs/train_cls.yaml
```

| Option | Meaning |
| --- | --- |
| `--config` | YAML with a `train_cls` section (required), e.g. `configs/train_cls.yaml` |

YAML settings: `ground_truth`, `detector` (must match the configuration's detector), `folds` (5),
`base_weights` (`yolo26n-cls.pt`), `epochs` (15), `image_size` (224), `dataset_dir`, `output_dir`.
Writes the crops to `data/cls_dataset/` (**real ID images**, delete when done) and the models plus
`manifest.json` to `models/yolo26_cls/` (gitignored). CPU only: expect 10-20 minutes.

## Config files

| File | Used by |
| --- | --- |
| `configs/nfs_eval.yaml` | `evaluate` |
| `configs/yolo_preview.yaml` | `preview-detector` |
| `configs/detector_sweep.yaml` | `sweep-detector` |
| `configs/train_cls.yaml` | `train-cls` |
