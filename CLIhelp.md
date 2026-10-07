# CLI help

Every command of the `id_classifier` package: what it does, its options, and what it writes.
**Rule:** every new or changed command is documented here in the same step.

## 0. Before any command: open the environment

PyTorch (needed by YOLO and DINOv2) is blocked on Windows by Smart App Control, so the model commands run
in WSL Ubuntu.

```bash
wsl -d Ubuntu                                    # from PowerShell / Windows Terminal
source ~/venvs/id_classifier/bin/activate        # the project's Python environment
cd /mnt/c/Users/LENOVO/Downloads/Valify_Project1_OCR
```

- The NFS share is at `/mnt/nfs` inside WSL.
- Output under `data/` is gitignored and never committed. Windows sees it at
  `C:\Users\LENOVO\Downloads\Valify_Project1_OCR\data\`.
- Every command prints help with `-h`, e.g. `python -m id_classifier evaluate -h`.
- One-time setup (already done): `python3 -m venv ~/venvs/id_classifier`, then
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` and `pip install -e ".[models,dev]"`.

## Overview

| Command | What it does | Reads | Writes |
| --- | --- | --- | --- |
| `pytest` | Runs the automated tests (no real images, no real model) | `tests/` | nothing |
| `build-ground-truth` | Builds the labeled answer key from the NFS folders (distinct pictures only) | NFS zips | `data/ground_truth.csv` |
| `evaluate` | Detects + classifies + routes every ground-truth image and scores it | experiment YAML + ground truth | `data/runs/<name>_<time>/` |

## Tests

```bash
python -m pytest -v                                      # all tests
python -m pytest -v tests/test_storage_and_pipeline.py   # one file
```

The tests need no PyTorch, so they also run on Windows in the repo's `.venv`:

```powershell
.venv\Scripts\python.exe -m pytest -v
```

Expected: all pass. The tests use tiny generated images and fake models (`tests/fakes.py`). They check our
own code (storage, pipeline), not the real models.

## build-ground-truth

Writes the answer key for `evaluate`. Every labeled NFS folder holds one document type of one country
(e.g. `tun_nid_ocr` = Tunisian national ID), and the file name gives the side (front/back). Copies are
dropped: same bytes (sha256) or the same card re-saved / re-shot (perceptual distance <= 40 of 256 bits).

```bash
python -m id_classifier build-ground-truth --root /mnt/nfs --per-folder 150
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--root` | `/mnt/nfs` | NFS root as this machine sees it |
| `--out` | `data/ground_truth.csv` | output CSV |
| `--per-folder` | all | at most N distinct pictures per folder (random) |
| `--seed` | 42 | same seed = same sample |

The labeled folders are listed in `FOLDER_LABELS` in `id_classifier/ground_truth.py`.

## evaluate

Runs every configuration (detector + classifiers) listed in an experiment YAML on the ground truth,
scores it, and picks the winner with the selection rule written in the YAML.

```bash
python -m id_classifier evaluate --config configs/nfs_eval.yaml
```

| Option | Meaning |
| --- | --- |
| `--config` | experiment YAML (required) |

Writes `data/runs/<experiment>_<time>/`: `comparison.md` (read this first), `confusion_<config>.png/.md`,
`predictions_<config>.csv`. A progress bar shows how many images are done.

## Config files

| File | Used by |
| --- | --- |
| `configs/nfs_eval.yaml` | `evaluate` |
