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
| `backfill-report` | Adds the "Beyond <biggest country>" section to an older run's report (no re-run) | a run folder | its `comparison.md`, `beyond_majority.csv` |
| `tune-routing` | Grid over the kNN decision / routing settings on embeddings made once | grid YAML (+ its experiment) | `data/sweeps/routing_tune_<time>/` |

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

- After a restart of this Windows machine the emulators are down (emulator tests then show 1 skipped): start
  Docker Desktop (`C:\Users\LENOVO\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe`; `docker` is not on
  PATH, the CLI is `...\DockerDesktop\resources\bin\docker.exe`), then `docker compose up -d`. The seeded objects
  are kept in the named volumes, no re-seeding needed (2026-10-14: 172 objects each after a restart).
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

The hand-labeled DocumentData drop (needed by the sweep configs, `data/ground_truth_plus.csv`) lives outside the
share, so it is seeded on its own, with its own root and its own manifest (the tests read `data/emulator_seed.csv`):

```bash
python scripts/seed_emulators.py --ground-truth data/document_data.csv \
    --strip-prefix C:/Users/LENOVO/Downloads --manifest data/emulator_seed_document_data.csv
# -> 14 uploaded per emulator, keys DocumentData/<file>.jpg; both emulators then hold 175 objects
```
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
| `--per-folder` | 150 | at most N distinct pictures per folder (random); 0 = all |
| `--seed` | 42 | same seed = same sample |

The labeled folders are listed in `FOLDER_LABELS` in `id_classifier/ground_truth.py`.

**Excluded images:** `configs/excluded_images.csv` (`key,reason,reported`, key = `<folder>/<file>`) lists pictures
whose folder label is wrong (e.g. `mar_nid_ocr/1575124_..._back_img.zip`, an Egyptian ID in the Moroccan folder).
They are left out of every rebuild; the NFS share itself is never changed. When adding a line, also drop the row from
the existing ground-truth CSVs and delete the object from the emulators (progress.md row 30 shows how).

**Relabelled images:** `configs/relabelled_images.csv` (`key,document_type,issuing_country,document_side,reason,reported`)
replaces the folder's label for pictures that are something else, e.g. the selfie
`dza_nid_ocr/1572908_..._front_img.zip` -> `none,unknown,n/a` (kept as a no-document example). Applied on every
rebuild; also update the existing ground-truth CSVs when adding a line (progress.md row 32).

## ingest-labeled-folder

Turns a folder of hand-labeled images into ground-truth rows. The label is in the file name:
`<country>_<docType>_<side>_img[_<n>].<ext>`, e.g. `egy_nid_front_img_2.jpg`.

```bash
python -m id_classifier ingest-labeled-folder --folder /mnt/c/Users/<you>/Downloads/DocumentData \
    --merge-with data/ground_truth.csv          # -> data/document_data.csv + data/ground_truth_plus.csv
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--folder` | (required) | folder of labeled images (in WSL, Windows' Downloads is under `/mnt/c/Users/<you>/`) |
| `--out` | `data/document_data.csv` | the new rows only |
| `--merge-with` | none | existing ground truth(s) to merge with, e.g. `data/ground_truth.csv` |
| `--merged-out` | `data/ground_truth_plus.csv` | merged CSV (what `configs/knn_sweep.yaml` reads) |

Name tokens (`id_classifier/labeled_files.py`): countries `egy uae ind irq nig syr tun mar dza ...` (nig = Nigeria,
NG); docTypes `nid` = national_id, `mrz`/`passport` = passport, `rp` = residence_permit, `dl` = driving_license,
`other` = no document (blank photo / no card: labeled none / unknown / n/a). A UAE `nid` is filed as
residence_permit/AE, the class of the NFS folder `UAE_RESIDENT_ID_OCR`. Front and back with the same country,
docType and number are one card (one transaction id). Any name it cannot read stops the command with the full list.

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

Live results: `comparison.md` (and the CSVs) are rewritten after **every** finished configuration, so a long sweep can
be read while it runs. The top says `Status: RUNNING. X of N configurations done, Y left (~Z min to go ...)` and the
winner is shown as "Leading so far (provisional)" until the last configuration is done. With a `source:` the kNN
reference images are fetched from that storage too (nothing is read from disk); an unreadable reference stops the run.

`comparison.md` also has a **"Beyond <biggest country>"** section (TN today, 150 of 173 documents): coverage, accepted
accuracy and wrong-country accepts over the OTHER countries only, plus per country `correct / auto-accepted / documents`.
The overall coverage / accepted_accuracy mostly measure the biggest country; read this section for the rest.

## backfill-report

Adds the "Beyond <biggest country>" section to a run made before it existed, from that run's own
`predictions_<config>.csv` files. Nothing is re-run; the rest of `comparison.md` is left as it was. Safe to repeat.

```bash
python -m id_classifier backfill-report --run data/runs/azurite_sweep_20261014-080858
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--run` | required | run folder (holds `comparison.csv`, `comparison.md`, `predictions_<config>.csv`) |

## tune-routing

Tunes only the LAST step (kNN floor, 1-NN similarity / margin / trigger share, routing threshold) without re-running
the pipeline per setting. The images are fetched from the experiment's `source:`, cropped and embedded **once**
(~75 s for 175 images; cached in `data/cache/routing_queries_<key>.*`, so the next grid skips it); then every setting
is decided and routed by the real code on the stored vectors (milliseconds each).

```bash
# Azurite: export AZURE_STORAGE_CONNECTION_STRING first (see "Emulator tests")
python -m id_classifier tune-routing --grid configs/routing_grid.yaml
python -m id_classifier tune-routing --grid configs/routing_grid.yaml --check-run data/runs/azurite_sweep_20261014-080858
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--grid` | required | grid YAML: `experiment:` (ground truth, source, detector, kNN), `fixed:` and `grid:` (lists, every combination) |
| `--out` | `data/sweeps/routing_tune_<time>/` | output folder |
| `--check-run` | none | a sweep run of the same experiment: replays its configurations and compares image for image (`check.md`) |

The embedding pass stores every region in 4 turns (0/90/180/270), so `query_rotations` can be tuned too:
`none` (as before), `portrait` (a crop taller than wide is also compared turned 90 and 270; the best match per
reference counts) or `all` (every crop in all 4 turns). The same setting exists on the kNN in experiment YAMLs.

`nn_max_class_refs` (kNN setting, tunable): the 1-NN fallback may only decide when the nearest reference's label has
at most this many references; a bigger class (TN) keeps the k-vote, unless `nn_big_class_min_similarity` is set:
then a big class may use 1-NN too, but only from that (stricter) similarity on. `nn_side_check: true` adds a
front/back check to that big-class 1-NN: the nearest reference's side must agree with the side the k neighbours of
the same type + country vote, else the k-vote stands. A grid file may add
`constraint: {min_majority_accepted_accuracy: 0.95}`: results.md then gets a "Constrained best" section (biggest
country's accepted accuracy at least that, 0 wrong-country, ranked by correct non-TN accepts) and the
`majority_accepted_accuracy` column.

Writes `results.md` / `results.csv` **live** (one row per setting as soon as it is done), then "Best settings"
(ranked: wrong-country accepts, then multi-sample documents in review, then coverage). Goal column = no document of a
class with 2+ images left in review AND no document auto-accepted with the wrong country.

## scripts/inspect_crops.py (crop review)

Shows what the kNN actually embeds for every document outside the biggest country: the photo with the detector box,
the crop, and the nearest other reference's crop, plus the numbers (box share of the photo, aspect ratio, detector
confidence, "no box found" fallback, nearest label + similarity, rank of the true class). Needs a `tune-routing` run
first (it reuses its regions + embeddings) and re-fetches only the shown images from the experiment's `source:`.

```bash
python scripts/inspect_crops.py --experiment configs/azurite_sweep.yaml --configuration nn_s0.85_m0.02
# --all: every document; --configuration: whose routing (predicted / routed / reasons) is shown
# -> data/crop_review/<time>/index.html (pictures: real IDs, local only) + crops.csv (numbers only)
```

The sheet lists the classes with 2+ images first (these should be recognised), then the single-sample classes.

### Evaluation rule: single-sample classes go to review

`review_single_sample_classes: true` in an experiment YAML (on in `azurite_sweep.yaml` / `minio_sweep.yaml`): a
document whose class (type/country/side) has ONE image in the ground truth has no reference of its own class under
leave-one-out, so it always goes to review (reason `single_sample_class`) instead of being matched to another class.
Applied by `evaluate`, `tune-routing` and the crop sheet. It uses the ground truth, so it is an evaluation rule, not
something the live pipeline can do.

## Config files

| File | Used by |
| --- | --- |
| `configs/nfs_eval.yaml` | `evaluate` (images from the NFS share) |
| `configs/minio_eval.yaml` | `evaluate`, images from MinIO (needs the emulators) |
| `configs/azurite_eval.yaml` | `evaluate`, images from Azurite (needs the emulators) |
| `configs/azurite_sweep.yaml` / `minio_sweep.yaml` | `evaluate`: the 12-configuration kNN sweep, images and references from Azurite / MinIO |
| `configs/routing_grid.yaml` | `tune-routing`: floor x s x m x share x query_rotations (864 settings) on the Azurite sweep experiment |
| `configs/routing_baseline.yaml` | `tune-routing`: the baseline (k-vote, no 1-NN) on the same images, to compare with the grid |
| `configs/routing_combined.yaml` | `tune-routing`: 1-NN gated by `nn_max_class_refs` (big classes keep the k-vote), with the constraint TN accepted accuracy >= 0.95 |
| `configs/routing_big_strict.yaml` | `tune-routing`: big classes may use 1-NN above `nn_big_class_min_similarity` (stricter bar), same constraint |
| `configs/routing_side_check.yaml` | `tune-routing`: big-class 1-NN with the front/back check `nn_side_check` on / off, same constraint |

