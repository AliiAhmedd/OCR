# Valify OCR 4201: ID document classifier (`nidClassifier`)

Core transactions that fail OCR with error **4201** ("incorrect document type") are sorted by their images:
**document type, issuing country and side**. A label is only accepted automatically when the system is confident;
everything else goes to **human review**. Nothing is ever forced.

This branch holds only what is needed to reproduce the results, starting from the raw NFS folders and the
credentials of the two local cloud emulators. It shows three things:

1. **Better non-Tunisian classification**: the chosen configuration recognises the non-TN cards that can be
   recognised, keeps Tunisian accuracy at 95.5% and never auto-accepts a wrong country.
2. **Single-sample documents go to human review automatically**: a card type with only one image has nothing to be
   compared with, so it is never guessed.
3. **Cloud storage**: the images are put into an Azure Blob bucket (Azurite) and an S3 bucket (MinIO), and the
   whole pipeline reads every image from there, as it will from the real storage after deployment.

## Pipeline

```
Core DB ──(ocr_error_extract.py, Airflow, daily)──> ocr_error_4201 table              step 1: which transactions failed

NFS folders (one zip per uploaded image)  +  DocumentData/ (hand-labeled images)
  │
  ├─ build-ground-truth ────────────────────────────────────────── ground_truth.py, sources.py
  │    read each zip in memory -> drop copies: same bytes (sha256) or the same card re-saved / re-shot
  │    (perceptual hash, distance <= 40 of 256 bits) -> label from folder + file name
  │    - configs/excluded_images.csv (wrong labels found by hand) + configs/relabelled_images.csv
  │    -> data/ground_truth.csv
  ├─ ingest-labeled-folder: + DocumentData ─────────────────────── labeled_files.py
  │    -> data/ground_truth_plus.csv
  ├─ find-duplicates (evidence): byte-identical re-uploads per folder ── duplicates.py
  │
  ├─ scripts/seed_emulators.py ─────────────────────────────────── references.py
  │    create bucket "ids" in Azurite and MinIO, upload every ground-truth image under <folder>/<file>
  │
  └─ evaluate --config configs/demo_azurite.yaml | demo_minio.yaml ─ evaluate.py, pipeline.py
       for every image:
         0. fetch the image from the bucket (Azure / S3 SDK, read-only)   fetchers.py, sources.py
         1. YOLOE-26 finds the document and crops it                     detectors.py
            (no box, e.g. a close-up -> the whole image is the document)
         2. DINOv2 embeds the crop (a portrait crop also turned 90/270)    classifiers/knn.py
            and compares it with every labeled reference, ALSO fetched
            from the bucket; the image's own picture is skipped
            (leave-one-out):
              - the 5 nearest references vote on (type, country, side)
              - a small class (<= 5 references) that loses the vote but is
                clearly the nearest is decided by its nearest reference (1-NN)
              - Tunisia (74+ references per side) may use 1-NN too, but only
                when the side of the nearest reference agrees with the side
                its Tunisian neighbours vote (front/back check)
         3. Routing: auto-accept, or human review with the reason         routing.py
            (low confidence, unrecognised, no document, single-sample class)
       -> data/runs/demo_<storage>_<time>/  (report, confusion matrix, every prediction)
```

## One-time setup (Windows, PowerShell)

```powershell
cd C:\Users\LENOVO\Downloads\Valify_Project1_OCR
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e ".[models,azure,s3]"
```

The first run downloads the model weights: `yoloe-26s-seg.pt` (31 MB) plus its text encoder from Ultralytics,
and `facebook/dinov2-small` (90 MB) from Hugging Face. Docker Desktop is needed for the emulators
(`docker` may not be on PATH: `C:\Users\LENOVO\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe`).

## Reproduce the results

From nothing but the NFS folders, the DocumentData folder and the emulators' credentials. Run in PowerShell from
the repo folder. `$py = ".venv\Scripts\python.exe"`.

```powershell
# 0. Mount the NFS share in WSL (it is gone after a restart); Windows then sees it at \\wsl.localhost\Ubuntu\mnt\nfs
wsl -d Ubuntu -u root -e sh -c "<NFS MOUNT COMMAND - TO FILL IN>"

# 1. Start the two cloud emulators (Azurite = Azure Blob, MinIO = S3); the data survives restarts
docker compose up -d

# 2. Answer key from the NFS: distinct pictures only, at most 150 per folder (~5 min, reads every zip)
& $py -m id_classifier build-ground-truth --root //wsl.localhost/Ubuntu/mnt/nfs

# 3. Add the hand-labeled DocumentData images
& $py -m id_classifier ingest-labeled-folder --folder C:/Users/LENOVO/Downloads/DocumentData --merge-with data/ground_truth.csv

# 4. (evidence) How many uploads are byte-identical copies of another upload
& $py -m id_classifier find-duplicates --root //wsl.localhost/Ubuntu/mnt/nfs

# 5. Create bucket "ids" in Azurite and MinIO and upload every ground-truth image
& $py scripts/seed_emulators.py --ground-truth data/ground_truth_plus.csv --strip-prefix //wsl.localhost/Ubuntu/mnt/nfs C:/Users/LENOVO/Downloads

# 6. Run the pipeline on Azurite: every image and every reference read from the Azure bucket (~2 min)
$env:AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
& $py -m id_classifier evaluate --config configs/demo_azurite.yaml

# 7. The same on MinIO: every image and every reference read from the S3 bucket (~3 min)
$env:AWS_ACCESS_KEY_ID = "rootuser"; $env:AWS_SECRET_ACCESS_KEY = "rootpass123"
& $py -m id_classifier evaluate --config configs/demo_minio.yaml
```

The credentials above are the emulators' public development defaults (`docker-compose.yml`), not secrets. Real
credentials never go into a config file: the fetchers read them from the environment only.

| Command | Option | Default | Meaning |
| --- | --- | --- | --- |
| `build-ground-truth` | `--root` | `/mnt/nfs` | NFS root as this machine sees it |
| | `--out` | `data/ground_truth.csv` | output CSV |
| | `--per-folder` | 150 | at most N distinct pictures per folder (random; same seed = same pictures); 0 = all |
| | `--seed` | 42 | sampling seed |
| `ingest-labeled-folder` | `--folder` | (required) | images named `<country>_<docType>_<side>_img[_n].<ext>` |
| | `--merge-with` / `--merged-out` | - / `data/ground_truth_plus.csv` | ground truth to merge into / merged CSV |
| `find-duplicates` | `--root` / `--out` | `/mnt/nfs` / `data/duplicates` | writes `summary.csv` (per folder) and `groups.csv` |
| `seed_emulators.py` | `--ground-truth` | `data/ground_truth.csv` | which files to upload |
| | `--strip-prefix` | `/mnt/nfs` | root(s) removed to make the key `<folder>/<file>` |
| | `--only` | both | `azure` or `s3` |
| `evaluate` | `--config` | (required) | `configs/demo_azurite.yaml` or `configs/demo_minio.yaml` |

Labeled NFS folders (`FOLDER_LABELS` in `ground_truth.py`): `tun_nid_ocr`, `mar_nid_ocr`, `dza_nid_ocr`,
`mar_driver_license_ocr`, `UAE_RESIDENT_ID_OCR`. No other NFS folder is ever read.

## Reading the results

`evaluate` prints the folder it wrote, `data/runs/demo_azurite_<time>/` (or `demo_minio_<time>/`):

| File | What it shows |
| --- | --- |
| `comparison.md` | Read first: accepted accuracy, coverage, review rate, **"Beyond TN"** (what is auto-accepted outside Tunisia, per country), small classes, metric definitions |
| `confusion_<config>.png` / `.md` | True country (rows) vs predicted country (columns) |
| `predictions_<config>.csv` | One row per image: true vs predicted labels, review flag and reasons (e.g. `single_sample_class`), the kNN's neighbours, side vote and rotation |

## Results (2026-10-14, 172 images, all read from Azurite)

169 documents + 3 photos without a document. Tunisia = 150 of the 169, so the "beyond TN" columns show what
happens to the other countries.

| | baseline (k-vote only) | **chosen configuration** |
| --- | --- | --- |
| Accepted accuracy (auto-accepted labels fully correct) | 95.2% | **95.6%** |
| Tunisian accepted accuracy | 95.2% | **95.5%** (128 / 134) |
| Auto-accepted with the wrong country | 0 | **0** |
| Non-TN documents auto-accepted / correct | 0 / 0 | **2 / 2** |
| Photos without a document rejected | 1 / 3 | **3 / 3** |
| Coverage (documents auto-accepted) | 74.0% | **80.5%** |
| Single-sample documents sent to review | (guessed) | **12 / 12** (`single_sample_class`) |

- The non-TN classes have 1 to 3 distinct images each. Under leave-one-out a class with n images has n-1
  references, so a single-sample class can never be recognised: those cards always go to human review.
- The 3 Moroccan-folder images that are really Egyptian IDs and 1 selfie were found by hand and are listed in
  `configs/excluded_images.csv` / `configs/relabelled_images.csv`.
- MinIO gives the same predictions as Azurite, image for image: only the storage differs.

## Files

| File | Role |
| --- | --- |
| `ocr_error_extract.py` | Step 1: Airflow DAG copying the 4201 transactions from Core into `ocr_error_4201` |
| `id_classifier/__main__.py` | The commands |
| `id_classifier/ground_truth.py` | Builds the answer key from the NFS folders (with the exclusion / relabel lists) |
| `id_classifier/sources.py` | Reads zips / images in memory; exact + perceptual de-duplication |
| `id_classifier/duplicates.py` | Report of byte-identical re-uploads (evidence for the de-duplication) |
| `id_classifier/labeled_files.py` | Ground truth from the hand-labeled DocumentData file names |
| `id_classifier/references.py` | Image reference -> storage key (`<folder>/<file>`) |
| `id_classifier/fetchers.py` | Read-only Azure Blob / S3 / GCS fetchers and the bucket image source |
| `id_classifier/detectors.py` | YOLOE-26 document detector and cropping |
| `id_classifier/classifiers/knn.py` | DINOv2 embedding + nearest-neighbour classifier (1-NN fallback, front/back check, rotation) |
| `id_classifier/routing.py` | Auto-accept vs human review, with the reasons |
| `id_classifier/pipeline.py`, `storage.py` | One image: fetch -> detect -> classify -> route; results database |
| `id_classifier/evaluate.py` | Runs every image, scores it, writes the report |
| `id_classifier/config.py`, `types.py` | YAML -> components; shared labels and records |
| `scripts/seed_emulators.py` | Creates the buckets in Azurite and MinIO and uploads the images |
| `configs/demo_azurite.yaml`, `configs/demo_minio.yaml` | The chosen configuration, reading from Azurite / MinIO |
| `configs/excluded_images.csv`, `configs/relabelled_images.csv` | Wrong labels found by hand |
| `docker-compose.yml` | The two emulators (ports on 127.0.0.1 only) |

## Privacy

Images are read in memory and never written to disk by the pipeline; they go only to the local emulators, never to
a real cloud. `data/` (ground truth, results, cached embeddings) and every `.jpg` are gitignored and never committed.
