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
own code (storage, pipeline, image payloads, references, cloud fetchers), not the real models.

The cloud tests need the `dev` extra (`pip install -e ".[dev]"`: boto3 + moto). moto fakes S3 in memory, so
no AWS account, network or real image is used. Without the extra, those tests are skipped, not failed.

## Local cloud emulators (docker compose)

Fake cloud storages on this machine for testing the fetchers: **Azurite** (Azure Blob Storage, for
`AzureBlobFetcher`) and **MinIO** (S3-compatible, for `S3Fetcher`). Defined in `docker-compose.yml`. Run in WSL
(Docker Desktop with WSL integration for Ubuntu turned on):

```bash
docker compose up -d       # start both (first time downloads the images)
docker compose ps          # both should say "Up"
docker compose logs minio  # a service's log
docker compose down        # stop (seeded data is kept)
docker compose down -v     # stop AND delete the seeded data
```

| Emulator | Address | Credentials (dev defaults, not secrets) |
| --- | --- | --- |
| Azurite blob | `http://127.0.0.1:10000/devstoreaccount1` | Azurite's fixed dev connection string, in env `AZURE_STORAGE_CONNECTION_STRING` |
| MinIO S3 API | `http://127.0.0.1:9222` (`endpoint_url`) | `AWS_ACCESS_KEY_ID=rootuser`, `AWS_SECRET_ACCESS_KEY=rootpass123` |
| MinIO web console | `http://127.0.0.1:9223` in a browser | same user / password |

- Ports listen on `127.0.0.1` only: the seeded data may be real ID images, which never leave this machine.
- MinIO uses the `cgr.dev/chainguard/minio` image: `minio/minio` was deleted from Docker Hub (2026-09-11).
- Quick check: `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:9222/minio/health/live` gives `200`.
- Azurite runs with `--skipApiVersionCheck`: the installed Azure SDK sends a newer API version than Azurite knows.
- `permission denied while trying to connect to the docker` in WSL: the shell was opened before Docker's WSL
  integration was turned on. `exit` and open WSL again.

### Seed the emulators

Uploads every zip listed in the ground truth (161 today) to Azurite container `ids` and MinIO bucket `ids`, under
its storage key (`/mnt/nfs/tun_nid_ocr/x.zip` -> `tun_nid_ocr/x.zip`, as `references.storage_key()` makes it).
The zips are real ID images; they only go to the local emulators. Needs the `[s3,azure]` extras.

```bash
docker compose up -d
python scripts/seed_emulators.py                 # both emulators
python scripts/seed_emulators.py --only azure    # or --only s3
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--ground-truth` | `data/ground_truth.csv` | which zips to upload (`image_path` column) |
| `--strip-prefix` | `/mnt/nfs` | NFS root removed to make the key |
| `--container` / `--bucket` | `ids` / `ids` | Azurite container / MinIO bucket |
| `--only` | both | `azure` or `s3` |
| `--manifest` | `data/emulator_seed.csv` | written list: key, NFS path, size, sha256 of each zip |

Safe to re-run: objects already there are skipped (second run prints `0 uploaded, 161 already there`).
Check in the MinIO console (`http://127.0.0.1:9223`, bucket `ids`).

### Emulator tests

The real `AzureBlobFetcher` (-> Azurite) and `S3Fetcher` (-> MinIO) read every seeded object: bytes identical
to the NFS zip (sha256 from the manifest), the decoded image identical to the NFS one (compared when `/mnt/nfs`
is mounted, i.e. in WSL), and the real servers' answers to a missing object, a missing container / bucket and a
wrong key. Skipped automatically when the emulators are not running or not seeded.

```bash
docker compose up -d && python scripts/seed_emulators.py
python -m pytest -v tests/test_emulators.py      # 5 tests, ~1 min (161 images x 2 emulators)
```

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

**Where the images come from:** by default the files named in the ground truth (`image_path`) are read from
disk. An optional `source:` in the YAML reads them from a storage instead, through the same fetchers as
production. Each `image_path` is turned into a storage key by removing `strip_prefixes`, then `prefix` is added in
the bucket. Credentials never go in the YAML (refused with an error); they come from the environment.

```yaml
source:
  type: s3                       # nfs (default) | s3 | azure | gcs
  bucket: ids                    # azure: container: ids
  prefix: ""                     # folder inside the bucket where Core's files start
  endpoint_url: http://127.0.0.1:9222   # s3 only, for S3-compatible stores such as MinIO
  strip_prefixes: ["/mnt/nfs"]   # removed from image_path to make the key
```

Same experiment from the local emulators (the results must equal `nfs_eval`):

```bash
export AWS_ACCESS_KEY_ID=rootuser AWS_SECRET_ACCESS_KEY=rootpass123
python -m id_classifier evaluate --config configs/minio_eval.yaml
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
python -m id_classifier evaluate --config configs/azurite_eval.yaml
```

`comparison.md` says where the images were read from (`Images read from: s3:ids`).

Writes `data/runs/<experiment>_<time>/`: `comparison.md` (read this first), `confusion_<config>.png/.md`,
`predictions_<config>.csv`. A progress bar shows how many images are done.

## Config files

| File | Used by |
| --- | --- |
| `configs/nfs_eval.yaml` | `evaluate` (images from the NFS share) |
| `configs/minio_eval.yaml` | `evaluate`, images from MinIO (needs the emulators) |
| `configs/azurite_eval.yaml` | `evaluate`, images from Azurite (needs the emulators) |
