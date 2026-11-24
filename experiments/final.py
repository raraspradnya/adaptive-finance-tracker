"""The combined best setup vs the frozen model alone. (EXPERIMENTS.md, "At the end")

The combined best is read from results/exp3/chosen.json, which carries every choice made on tune:
Experiment 2's reader (store everything, trust, freshness) plus Experiment 3's memory limit,
eviction rule and merging. Nothing is chosen here.

Run:  .venv/bin/python experiments/final.py                  tune (for reference: same numbers as Exp 3)
      .venv/bin/python experiments/final.py --val            val, once, after Experiments 1-3
      .venv/bin/python experiments/final.py --test FILE.csv  the blind test set, once, at the very end

The test file needs the columns: txn_id, date, description, amount, note, true_category.
Writes results/final/<split>/ : both logs, summary.csv, monthly.csv, two charts.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from common import ROOT, TUNE, VAL, report, run_conditions
from memory import Config

CHOSEN = ROOT / "results" / "exp3" / "chosen.json"
COLUMNS = {"txn_id", "date", "description", "amount", "note", "true_category"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--val", action="store_true", help="run on the val year (once, after Experiments 1-3)")
    which.add_argument("--test", type=Path, help="run on this blind test file (once, at the very end)")
    args = ap.parse_args()
    pd.set_option("display.width", 220)

    if args.test:
        split, data = "test", args.test
        missing = COLUMNS - set(pd.read_csv(data, nrows=0).columns)
        if missing:
            sys.exit(f"{data} is missing columns: {', '.join(sorted(missing))}")
    elif args.val:
        split, data = "val", VAL
        if not VAL.exists():
            sys.exit("No val year yet. Generate it with: .venv/bin/python src/data_generator.py --val")
    else:
        split, data = "tune", TUNE

    if not CHOSEN.exists():
        sys.exit("Run Experiments 2 and 3 on tune first: they choose the combined best setup.")
    best = Config(**json.loads(CHOSEN.read_text()))
    configs = [Config(name="Frozen only", storage="none"), best]

    out = ROOT / "results" / "final" / split
    logs, summary_t, monthly_t = run_conditions(data, configs, out)
    report(split, data, out, logs, summary_t, monthly_t)

    frozen, final = summary_t.loc["Frozen only"], summary_t.loc[best.name]
    print(f"\nHeadline ({split}): accuracy {frozen['accuracy']}% -> {final['accuracy']}%, "
          f"manual checks {int(frozen['manual_checks'])} -> {int(final['manual_checks'])} a year, "
          f"memory {int(final['memory_size'])} rows.")


if __name__ == "__main__":
    main()
