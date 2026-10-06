# Progress journal

Evidence of every trial: what we tried, what it measured, and what we decided. One row per milestone,
oldest first. Numbers are copied from the run outputs named in the row (under `data/`, not committed).
How to run each command: `CLIhelp.md`.

## Milestones

| # | Date | Milestone | What we tried | Result (measured) | Decision |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026-10-05 | Research and plan | Read recent ID-document work (2026 ID forgery competition, Persona layout embeddings, 2026 ID-attack survey, IDCard-YOLO) and the YOLO-OCR vs VLM car-plate paper | Best recent ID systems: YOLO crops the document, then a DINO-style embedding classifies it. Zero-shot VLMs are unreliable on IDs and slow on CPU | Plan: YOLO first, then embedding kNN (A) / YOLO26-cls (B); a small VLM only as a fallback for unknown countries (E) |
| 2 | 2026-10-05 | Environment | Installed PyTorch + Ultralytics on Windows | Windows Smart App Control blocked PyTorch's `shm.dll` | Run everything in WSL Ubuntu (`~/venvs/id_classifier`); NFS is native at `/mnt/nfs` |
| 3 | 2026-10-05 | Step 1: YOLO detector | `YoloDetector` with YOLOE-26s (open vocabulary), prompts `["identity card", "passport", "driving licence"]`, confidence >= 0.25, 5 random images per folder (`preview-detector`) | **4 / 22 images** with a box, all Algerian ID backs, confidence 0.29-0.32. Boxes tight where found. 85-350 ms/image on CPU. 44 tests pass | Prompts and/or threshold are wrong; measure before changing anything |
| 4 | 2026-10-05 | Duplicates found (bytes) | `find-duplicates`: sha256 of every image in the 5 labeled folders | Distinct / zips: TN 596 / 914, MA ID 5 / 242, DZ 7 / 62, MA licence 4 / 62, UAE 1 / 2. One image in both the TN and DZ folders (same image as front and back) | Only Tunisia has real volume; other folders look like QA/test uploads. Sample only distinct images from now on |
| 5 | 2026-10-05 | Step 2: prompt x threshold sweep | `sweep-detector`, 4 prompt sets x thresholds 0.05-0.25 on 37 byte-distinct images (`sweep_20261005_053615`) | Images with a box at >= 0.15: `["identity card", "passport", "driving licence"]` **3/37**, `["identity card"]` **0/37**, `["id card", "card"]` **22/37**, `["document", "plastic card"]` **12/37**. At >= 0.05: 8, 1, 29, 25 /37 (with 3, 0, 4, 2 images getting 2+ boxes). At >= 0.25: 3, 0, 18, 7 /37. ~100 ms/image | The prompt wording matters far more than the threshold. Use `["id card", "card"]` at 0.15 (no image with 2+ boxes there) |
| 6 | 2026-10-05 | Step 3: preview with the best prompts | `preview-detector` with `["id card", "card"]`, confidence >= 0.15, same 37 images as the sweep | **22 / 37** with a box (TN 12/20, MA ID 2/5, DZ 6/7, MA licence 2/4, UAE 0/1), matching the sweep exactly. 0 images with 2+ boxes. Confidence up to 0.70 (DZ 0.47-0.62, was 0.29-0.32) | User check: duplicates still visible in the samples; some close-ups get a box, others don't (detailed report pending) |
| 7 | 2026-10-06 | Re-saved copies (perceptual hash) | Compared the sampled DZ / MA licence images by look (16x16 dHash, 256 bits) | "Distinct" images that look identical have byte-different files at perceptual distance **0-1**; different pictures are > 40 apart. Byte dedup misses them: DZ 7 -> 3 real pictures, MA licence 4 -> 2. The 37 images of rows 5-6 therefore hold fewer than 37 distinct pictures | Add near-duplicate detection (perceptual distance <= threshold). Calibrate the threshold on the real data and check close pairs by eye before trusting it |

| 8 | 2026-10-06 | Near-duplicate threshold calibration | `find-duplicates --review-images` over all 1,282 zips: perceptual distance of every pair of byte-distinct images; 46 pairs at distance <= 40 drawn side by side | Distances of close pairs: 27 pairs at **0-3**, then one pair at **8**, then nothing until **15**, 21, 21, 29+. No close pair crosses folders. Distinct pictures with threshold 10: TN 596 -> **582**, MA ID 5 -> **5**, DZ 7 -> **3**, MA licence 4 -> **2**, UAE **1**. Pairs at 15-39 are mostly consecutive transaction ids (likely a new photo of the same card, not a re-saved file) | Threshold 10 sits in the gap (8 = copy, 15 = different). Pending: user checks the review images at 8, 15 and 21 before it is final |

## Open questions

- Not enough distinct images outside Tunisia to evaluate MA / DZ / UAE. The real 4201 transactions need
  mapping to their NFS files in the full QA/prod database.
- Close-ups (the card fills the photo): some get a box, others don't. Step 4 (whole-image fallback)
  targets them; the detailed check comes after the duplicate fix.
