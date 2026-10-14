"""Command line: python -m id_classifier <command> ...

    build-ground-truth   write ground_truth.csv from the labeled NFS folders (distinct pictures only)
    ingest-labeled-folder  write a ground-truth CSV from hand-labeled files (<country>_<docType>_<side>_img[_n])
                           and optionally merge it with another ground truth
    evaluate             detect + classify + route every ground-truth image and write the results report

See README.md for the full walk-through.
"""

from __future__ import annotations

import argparse
import logging
import sys

from id_classifier.evaluate import run_experiment
from id_classifier.ground_truth import build_ground_truth
from id_classifier.labeled_files import ground_truth_rows, merge_ground_truths, read_labeled_folder, summarize, write_rows


def cmd_build_ground_truth(args: argparse.Namespace) -> int:
    csv_path = build_ground_truth(args.root, args.out, per_folder=args.per_folder, seed=args.seed)
    print(f"Wrote {csv_path}")
    return 0


def cmd_ingest_labeled_folder(args: argparse.Namespace) -> int:
    files = read_labeled_folder(args.folder)
    print(summarize(files))
    csv_path = write_rows(args.out, ground_truth_rows(files))
    print(f"Wrote {csv_path}")
    if args.merge_with:
        merged = merge_ground_truths([*args.merge_with, csv_path], args.merged_out)
        print(f"Wrote {merged} (= {' + '.join(map(str, [*args.merge_with, csv_path]))})")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    result = run_experiment(args.config)
    print(f"Results: {result.output_dir / 'comparison.md'}")
    print(f"Winner: {result.winner or 'none (no configuration meets the selection rule)'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m id_classifier", description="Classify non-Egyptian ID images.")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("build-ground-truth", help="write ground_truth.csv from the labeled NFS folders")
    p.add_argument("--root", default="/mnt/nfs", help="NFS root as this machine sees it (default: /mnt/nfs)")
    p.add_argument("--out", default="data/ground_truth.csv", help="output CSV (default: data/ground_truth.csv)")
    # 150 by default, not "all": the share holds 565 distinct Tunisian pictures against 1-5 for every other
    # country, and that imbalance lets the k-vote be won by Tunisia on looks alone. 150 is also the cap the
    # 161-image baseline used (progress.md rows 9 and 20), so runs stay comparable with it.
    # Pass --per-folder 0 for no cap.
    p.add_argument("--per-folder", type=int, default=150,
                   help="at most this many DISTINCT pictures per folder (default: 150; 0 = all)")
    p.add_argument("--seed", type=int, default=42, help="same seed = same sample (default: 42)")
    p.set_defaults(func=cmd_build_ground_truth)

    p = commands.add_parser("ingest-labeled-folder", help="ground truth from hand-labeled image file names")
    p.add_argument("--folder", required=True, help="folder of <country>_<docType>_<side>_img[_n].<ext> images")
    p.add_argument("--out", default="data/document_data.csv", help="output CSV (default: data/document_data.csv)")
    p.add_argument("--merge-with", nargs="+", help="ground-truth CSV(s) to merge the new rows into, e.g. data/ground_truth.csv")
    p.add_argument("--merged-out", default="data/ground_truth_plus.csv",
                   help="merged CSV (default: data/ground_truth_plus.csv)")
    p.set_defaults(func=cmd_ingest_labeled_folder)

    p = commands.add_parser("evaluate", help="run the configuration(s) of an experiment YAML and write the report")
    p.add_argument("--config", required=True, help="experiment YAML, e.g. configs/nfs_eval.yaml")
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
