"""One table across the embedding experiments (configs/embeddings/, progress.md row 44).

For every variant: its latest evaluate run (data/runs/embed_<variant>_<time>/) with the DINOv2-small thresholds, and
the best tuned setting from data/sweeps/embed_<variant>/results.csv under the same constraint as configuration 15
(TN accepted accuracy >= 0.95, 0 wrong-country accepts; most correct non-TN accepts, then coverage).

    python scripts/compare_embeddings.py          # -> data/runs/embedding_comparison.md
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

RUNS, SWEEPS, CONFIGS = Path("data/runs"), Path("data/sweeps"), Path("configs/embeddings")
CONFIGS_SHOWN = ("A_baseline", "configuration_14", "configuration_15")


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_run(variant: str) -> Path | None:
    runs = sorted(RUNS.glob(f"embed_{variant}_*/comparison.csv"))
    return runs[-1].parent if runs else None


def run_cells(run: Path) -> dict[str, str]:
    comparison = {r["config"]: r for r in _rows(run / "comparison.csv")}
    beyond = {r["config"]: r for r in _rows(run / "beyond_majority.csv")} if (run / "beyond_majority.csv").exists() else {}
    cells = {}
    for name in CONFIGS_SHOWN:
        if name not in comparison:
            cells[name] = "-"
            continue
        c, b = comparison[name], beyond.get(name, {})
        cells[name] = (f"{c['coverage']} / {c['accepted_accuracy']} / wrong {c['accepted_wrong_country_rate']} / "
                       f"non-TN {b.get('auto_accepted', '?')}:{b.get('accepted_correct', '?')}")
    return cells


def tuned_best(variant: str) -> str:
    path = SWEEPS / f"embed_{variant}" / "results.csv"
    if not path.exists():
        return "not tuned yet"
    ok = [r for r in _rows(path) if r["accepted_wrong_country"] == "0"
          and r["majority_accepted_accuracy"] not in ("", "None") and float(r["majority_accepted_accuracy"]) >= 0.95]
    if not ok:
        return "no setting keeps TN >= 0.95 with 0 wrong-country"
    best = max(ok, key=lambda r: (int(r["beyond_correct"]), float(r["coverage"])))
    return (f"coverage {float(best['coverage']):.3f}, accepted {float(best['accepted_accuracy']):.3f}, "
            f"TN {float(best['majority_accepted_accuracy']):.3f}, non-TN {best['beyond_accepted']}:{best['beyond_correct']}, "
            f"multi in review {best['reviewed_multi_sample']} ({best['setting']})")


def main() -> int:
    variants = sorted(p.stem for p in CONFIGS.glob("*.yaml") if not p.stem.endswith("_grid"))
    lines = ["# Embedding comparison", "",
             "Each cell: coverage / accepted_accuracy / wrong-country rate / non-TN auto-accepted:correct, with the "
             "DINOv2-small thresholds. Reference (DINOv2-small, progress.md rows 39 and 42): configuration 14 = "
             "0.805 / 0.956 / 0 / 2:2, configuration 15 = 0.864 / 0.952 / 0 / 2:2.", "",
             "| variant | run | " + " | ".join(CONFIGS_SHOWN) + " |", "|" + "---|" * (len(CONFIGS_SHOWN) + 2)]
    tuned = []
    for v in variants:
        run = latest_run(v)
        cells = run_cells(run) if run else {name: "not run" for name in CONFIGS_SHOWN}
        lines.append(f"| {v} | {run.name if run else '-'} | " + " | ".join(cells[n] for n in CONFIGS_SHOWN) + " |")
        tuned.append(f"- **{v}**: {tuned_best(v)}")
    lines += ["", "## Best retuned setting per embedding (TN >= 0.95, 0 wrong-country)", "", *tuned, ""]
    out = RUNS / "embedding_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
