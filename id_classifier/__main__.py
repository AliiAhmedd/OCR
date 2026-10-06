"""Command line: python -m id_classifier <command> ...

    build-ground-truth   write ground_truth.csv from the labeled NFS folders (tun_nid_ocr, mar_nid_ocr, ...)
    evaluate             run every configuration of an experiment YAML and write the comparison table
    classify             run the pipeline on a source and store the results in the database
    preview-detector     run a detector on a few NFS images and save the boxes + crops for a visual check
    sweep-detector       compare detector settings (prompts x thresholds): detection counts per folder, no images
    find-duplicates      find NFS images uploaded more than once (same bytes), within and across folders

Every command, its options and its output files are described in CLIhelp.md.
"""

from __future__ import annotations

import argparse
import logging
import sys

from id_classifier.config import build, build_classifiers, build_detector, build_source, load_yaml
from id_classifier.duplicates import find_duplicates
from id_classifier.evaluate import run_experiment
from id_classifier.ground_truth import FOLDER_LABELS, build_ground_truth
from id_classifier.pipeline import new_run_id, run_classification
from id_classifier.preview import preview_detector
from id_classifier.routing import RoutingConfig
from id_classifier.sources import NEAR_DUPLICATE_DISTANCE
from id_classifier.storage import Storage
from id_classifier.sweep import sweep_detectors


def cmd_build_ground_truth(args: argparse.Namespace) -> int:
    csv_path = build_ground_truth(args.root, args.out, per_folder=args.per_folder, seed=args.seed)
    print(f"Wrote {csv_path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    result = run_experiment(args.config)
    print(f"Results: {result.output_dir / 'comparison.md'}")
    print(f"Winner: {result.winner or 'none (no configuration meets the selection rule)'}")
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    if "classify" not in config:
        raise SystemExit(f"{args.config} has no 'classify' section")
    section = config["classify"]
    counts = run_classification(
        source=build_source(section["source"]),
        detector=build_detector(section.get("detector")),
        classifiers=build_classifiers(section["classifiers"]),
        routing=RoutingConfig.from_dict(config.get("routing")),
        storage=Storage(section["database_url"]),
        run_id=args.run_id or new_run_id(),
    )
    print(f"Run {counts['run_id']}: {counts['images']} images")
    print(f"Per country: {dict(sorted(counts['per_country'].items()))}")
    print(f"Per routing status: {dict(sorted(counts['per_status'].items()))}")
    return 0


def cmd_preview_detector(args: argparse.Namespace) -> int:
    section = load_yaml(args.config).get("preview")
    if section is None:
        raise SystemExit(f"{args.config} has no 'preview' section")
    summary = preview_detector(
        detector=build_detector(section.get("detector")),
        root=args.root or section.get("root", "/mnt/nfs"),
        folders=args.folders or section.get("folders") or list(FOLDER_LABELS),   # only these folders are read
        output_dir=section.get("output_dir", "data/detect_preview"),
        per_folder=args.per_folder or section.get("per_folder", 5),
        seed=section.get("seed", 42),
        unique_only=section.get("unique_only", True),
    )
    print(f"Wrote {summary} (boxes + crops next to it)")
    print("Reminder: these are copies of real ID images. Delete the folder after the check.")
    return 0


def cmd_sweep_detector(args: argparse.Namespace) -> int:
    section = load_yaml(args.config).get("sweep")
    if section is None:
        raise SystemExit(f"{args.config} has no 'sweep' section")
    thresholds = section.get("thresholds", [0.05, 0.10, 0.15, 0.25])
    detectors = {   # one YOLO detector per prompt set, all at the lowest threshold (see sweep.py)
        name: build("detector", {**section["detector"], "name": name, "prompts": prompts,
                                 "confidence": min(thresholds)})
        for name, prompts in section["prompt_sets"].items()
    }
    run_dir = sweep_detectors(
        detectors=detectors,
        thresholds=thresholds,
        root=args.root or section.get("root", "/mnt/nfs"),
        folders=args.folders or section.get("folders") or list(FOLDER_LABELS),
        output_dir=section.get("output_dir", "data/sweeps"),
        per_folder=args.per_folder or section.get("per_folder", 20),
        seed=section.get("seed", 42),
        unique_only=section.get("unique_only", True),
    )
    print(f"Results: {run_dir / 'sweep.md'}")
    return 0


def cmd_find_duplicates(args: argparse.Namespace) -> int:
    out = find_duplicates(args.root, args.folders or list(FOLDER_LABELS), args.out, max_distance=args.max_distance,
                          review_distance=args.review_distance, review_images=args.review_images)
    print(f"Results in {out}: summary.csv, groups.csv, distances.csv, close_pairs.csv")
    if args.review_images:
        print(f"Review images in {out / 'review'}: copies of real ID images. Delete the folder after the check.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m id_classifier", description="Classify non-Egyptian ID images.")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("build-ground-truth", help="write ground_truth.csv from the labeled NFS folders")
    p.add_argument("--root", default="/mnt/nfs", help="NFS root as this machine sees it (default: /mnt/nfs)")
    p.add_argument("--out", default="data/ground_truth.csv", help="output CSV (default: data/ground_truth.csv)")
    p.add_argument("--per-folder", type=int, help="at most this many images per folder (default: all)")
    p.add_argument("--seed", type=int, default=42, help="same seed = same sample (default: 42)")
    p.set_defaults(func=cmd_build_ground_truth)

    p = commands.add_parser("evaluate", help="compare the configurations of an experiment YAML")
    p.add_argument("--config", required=True, help="experiment YAML, e.g. configs/nfs_eval.yaml")
    p.set_defaults(func=cmd_evaluate)

    p = commands.add_parser("classify", help="run the pipeline and store results in the database")
    p.add_argument("--config", required=True, help="YAML with a 'classify' section")
    p.add_argument("--run-id", help="reuse a run id to rerun safely (default: cli-<timestamp>)")
    p.set_defaults(func=cmd_classify)

    p = commands.add_parser("preview-detector", help="save a detector's boxes and crops on a few NFS images")
    p.add_argument("--config", required=True, help="YAML with a 'preview' section, e.g. configs/yolo_preview.yaml")
    p.add_argument("--root", help="NFS root, overrides the YAML (e.g. /mnt/nfs)")
    p.add_argument("--folders", nargs="+", help="NFS folders to sample, overrides the YAML")
    p.add_argument("--per-folder", type=int, help="images per folder, overrides the YAML")
    p.set_defaults(func=cmd_preview_detector)

    p = commands.add_parser("sweep-detector", help="detection counts per folder for several prompt sets x thresholds")
    p.add_argument("--config", required=True, help="YAML with a 'sweep' section, e.g. configs/detector_sweep.yaml")
    p.add_argument("--root", help="NFS root, overrides the YAML (e.g. /mnt/nfs)")
    p.add_argument("--folders", nargs="+", help="NFS folders to sample, overrides the YAML")
    p.add_argument("--per-folder", type=int, help="distinct images per folder, overrides the YAML")
    p.set_defaults(func=cmd_sweep_detector)

    p = commands.add_parser("find-duplicates", help="find NFS images uploaded more than once")
    p.add_argument("--root", default="/mnt/nfs", help="NFS root as this machine sees it (default: /mnt/nfs)")
    p.add_argument("--folders", nargs="+", help="NFS folders to check (default: the labeled ID folders)")
    p.add_argument("--out", default="data/duplicates", help="output folder (default: data/duplicates)")
    p.add_argument("--max-distance", type=int, default=NEAR_DUPLICATE_DISTANCE,
                   help=f"perceptual distance (of 256 bits) at or below which two images are copies "
                        f"(default: {NEAR_DUPLICATE_DISTANCE})")
    p.add_argument("--review-distance", type=int, default=40,
                   help="list (and with --review-images draw) every pair up to this distance (default: 40)")
    p.add_argument("--review-images", action="store_true",
                   help="write each close pair side by side to <out>/review (real ID images)")
    p.set_defaults(func=cmd_find_duplicates)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
