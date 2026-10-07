# Valify OCR 4201: ID document classifier (`nidClassifier`)

Core transactions that fail OCR with error **4201** ("incorrect document type") are sorted by their
images: **document type, issuing country and side**. A label is only accepted automatically when the
system is confident; everything else goes to **human review**. Nothing is ever forced.

This branch holds only what is needed to reproduce the results from the raw NFS folders.

## Pipeline

```
Core DB ──(ocr_error_extract.py, Airflow, daily)──> ocr_error_4201 table          step 1: which transactions failed

NFS folders (one zip per uploaded image)
  │
  ├─ build-ground-truth ─────────────────────────────────────────────────────────────── ground_truth.py, sources.py
  │    read each zip in memory -> drop copies: same bytes (sha256) or same card re-saved /
  │    re-shot (perceptual hash, distance <= 40 of 256 bits) -> label from folder + file name
  │    -> data/ground_truth.csv
  │
  └─ evaluate ─────────────────────────────────────────────────────────────────────── evaluate.py, pipeline.py
       for every image:
         1. YOLOE-26 finds the document and crops it                          detectors.py (YoloDetector)
            (no box, e.g. a close-up -> the whole image is the document)
         2. DINOv2 turns the crop into an embedding; the 5 most similar        classifiers/knn.py
            labeled cards vote on (type, country, side); the image's own
            picture is skipped (leave-one-out). Nearest card too different
            (similarity < 0.72) -> "unknown"
         3. Routing: auto-accept, or human review with the reason             routing.py
            (low confidence, unknown country, no document, several documents)
       -> data/runs/nfs_eval_<time>/  (report, confusion matrix, every prediction)
```

## One-time setup

PyTorch is blocked on Windows by Smart App Control here, so everything runs in WSL Ubuntu, where the NFS
share is mounted at `/mnt/nfs`.

```bash
wsl -d Ubuntu
python3 -m venv ~/venvs/id_classifier
source ~/venvs/id_classifier/bin/activate
cd /mnt/c/Users/LENOVO/Downloads/Valify_Project1_OCR      # this repo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch
pip install -e ".[models]"
```

The first run downloads the model weights: `yoloe-26s-seg.pt` (31 MB) plus its text encoder (254 MB)
from Ultralytics, and `facebook/dinov2-small` (90 MB) from Hugging Face.

## Reproduce the results (two commands)

Start from nothing but the NFS folders:

```bash
source ~/venvs/id_classifier/bin/activate
cd /mnt/c/Users/LENOVO/Downloads/Valify_Project1_OCR

# 1. Answer key: distinct pictures only (about 5 minutes; reads the NFS zips)
python -m id_classifier build-ground-truth --root /mnt/nfs --per-folder 150

# 2. Detect + classify + route every image and score it (about 5 minutes on CPU)
python -m id_classifier evaluate --config configs/nfs_eval.yaml
```

| Command | Option | Default | Meaning |
| --- | --- | --- | --- |
| `build-ground-truth` | `--root` | `/mnt/nfs` | NFS root as this machine sees it |
| | `--out` | `data/ground_truth.csv` | output CSV |
| | `--per-folder` | all | at most N distinct pictures per folder (random; same seed = same pictures) |
| | `--seed` | 42 | sampling seed |
| `evaluate` | `--config` | (required) | experiment YAML: `configs/nfs_eval.yaml` |

Labeled folders used (`FOLDER_LABELS` in `ground_truth.py`): `tun_nid_ocr`, `mar_nid_ocr`, `dza_nid_ocr`,
`mar_driver_license_ocr`, `UAE_RESIDENT_ID_OCR`. No other NFS folder is ever read.

## Reading the results

`evaluate` prints the folder it wrote, `data/runs/nfs_eval_<time>/`:

| File | What it shows |
| --- | --- |
| `comparison.md` | Read first: accuracy, accepted accuracy, review rate, per-country table, metric definitions |
| `confusion_<config>.png` / `.md` | True country (rows) vs predicted country (columns) |
| `predictions_<config>.csv` | One row per image: true vs predicted labels, review flag and reason, kNN neighbours |

The two numbers that matter most: **accepted_accuracy** (how often an auto-accepted label is fully
correct) and **review_rate** (how much is left for humans).

## Current results

Run on 161 distinct ID images (2026-10-06):

| Metric | Value |
| --- | --- |
| Accepted accuracy (auto-accepted labels fully correct) | **95.2%** |
| Wrong country auto-accepted | **0** |
| Tunisia (150 images): type / country / side | 96.0% / 96.0% / 91.3% |
| Coverage (documents auto-accepted) | 78.3% |
| Review rate | 21.7% |
| Time per image (CPU) | ~0.18 s |

Morocco, Algeria and UAE have only 1-7 distinct pictures in the NFS folders. With leave-one-out they have
no other card of their country to match, so they correctly come out as `unknown` and go to review. Their
real accuracy needs more distinct images per country.

## Files

| File | Role |
| --- | --- |
| `ocr_error_extract.py` | Step 1: Airflow DAG copying the 4201 transactions from Core into `ocr_error_4201` |
| `id_classifier/__main__.py` | The two commands |
| `id_classifier/ground_truth.py` | Builds the answer key from the NFS folders |
| `id_classifier/sources.py` | Gets the image bytes, then reads them in memory (NFS zip, base64 text or plain image); exact + perceptual de-duplication |
| `id_classifier/references.py` | Turns an image reference from Core (`media/...`, `/...`, `./...`) into one storage key |
| `id_classifier/detectors.py` | YOLOE-26 document detector and cropping |
| `id_classifier/classifiers/knn.py` | DINOv2 embedding + k-nearest-neighbour classifier |
| `id_classifier/routing.py` | Auto-accept vs human review |
| `id_classifier/pipeline.py` | One image: detect -> classify -> route; `run_classification()` loops over a source and stores the results |
| `id_classifier/storage.py` | The pipeline's own tables (images, regions, predictions, reviews) on SQLite or Postgres |
| `id_classifier/evaluate.py` | Runs every image, scores it, writes the report |
| `id_classifier/config.py`, `types.py` | YAML -> components; shared labels and records |
| `configs/nfs_eval.yaml` | The configuration (detector + classifier settings) |
| `tests/` | Automated tests with fake images and models (`python -m pytest`) |
| `CLIhelp.md`, `progress.md`, `OPEN_QUESTIONS.md` | Every command; journal of every trial; questions for the supervisors |

## Privacy

Images are read from the zips in memory and never written to disk; `data/` (ground truth, results,
cached embeddings) is gitignored and never committed.
