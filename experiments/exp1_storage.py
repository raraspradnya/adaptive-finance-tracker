"""Experiment 1: what to store. (EXPERIMENTS.md)

Same reader in every condition (similarity-only kNN). Only the storage policy changes.
Nothing is tuned here: the four conditions are fixed, so val simply reruns them.

Run:  .venv/bin/python experiments/exp1_storage.py         tune (all choices are made here)
      .venv/bin/python experiments/exp1_storage.py --val   val, once, after Experiments 1-3

Writes results/exp1/<split>/ : one log per condition, summary.csv, monthly.csv, two charts
(plus cases.csv for tune).
"""

from common import ROOT, pick_split, report, run_conditions
from memory import Config

CONDITIONS = [
    Config(name="Frozen only", storage="none"),
    Config(name="Corrections only", storage="corrections"),
    Config(name="Corrections + answered", storage="corrections_asked"),
    Config(name="Everything", storage="everything"),
]


def main():
    split, data = pick_split(__doc__.splitlines()[0])
    out = ROOT / "results" / "exp1" / split
    logs, summary_t, monthly_t = run_conditions(data, CONDITIONS, out)
    report(split, data, out, logs, summary_t, monthly_t)


if __name__ == "__main__":
    main()
