"""Experiment 2: how to read. (EXPERIMENTS.md)

Builds on Experiment 1's winner: memory stores corrected and answered rows only. Every stored row
is then a user's own answer, so trust is the same for all of them and drops out. What is left to
test is freshness:

    vote = similarity x freshness        freshness: a vote halves every N months (1 / 3 / 6)

Choice (on tune, saved to results/exp2/chosen.json): fewest manual checks, then highest accuracy;
a tie goes to the simpler setup (no freshness first, then the longest half-life).

One extra comparison row, not a candidate: "Everything + trust + freshness" (store every row,
left-alone rows count 0.25, votes halve every month). It asks whether a smarter reader would
rescue storing everything. Its setting came from an earlier exploratory grid on tune
(results/exp2/exploratory/), so on tune it is optimistic; val gives the fair comparison.

Run:  .venv/bin/python experiments/exp2_read.py         tune: run, choose, save
      .venv/bin/python experiments/exp2_read.py --val   val, once, after Experiments 1-3

Writes results/exp2/<split>/ : logs, summary.csv, monthly.csv, two charts (+ cases.csv on tune).
"""

import json
from dataclasses import asdict, replace

from common import ROOT, pick_split, report, run_conditions
from memory import Config

HALF_LIFE = [6, 3, 1]     # months for a vote to halve; listed from mildest to strongest (tie order)

FROZEN = Config(name="Frozen only", storage="none")
EXP1 = Config(name="Exp 1 winner", storage="corrections_asked")   # corrections + answered
COMPARISON = Config(name="Everything + trust + freshness", storage="everything",
                    trust_accepted=0.25, half_life_months=1)

CHOSEN = ROOT / "results" / "exp2" / "chosen.json"


def candidates() -> list[Config]:
    """Simplest first, so min() breaks ties toward the simpler setup."""
    return [EXP1] + [replace(EXP1, name=f"Exp 1 winner + Freshness {h} mo", half_life_months=h)
                     for h in HALF_LIFE]


def conditions() -> list[Config]:
    return [FROZEN] + candidates() + [COMPARISON]


def main():
    split, data = pick_split(__doc__.splitlines()[0])
    if split == "val" and not CHOSEN.exists():
        raise SystemExit("Run Experiment 2 on tune first: it chooses the setup Experiment 3 builds on.")

    out = ROOT / "results" / "exp2" / split
    configs = conditions()
    logs, summary_t, monthly_t = run_conditions(data, configs, out)
    report(split, data, out, logs, summary_t, monthly_t)

    if split == "tune":
        best = min(candidates(), key=lambda c: (summary_t.at[c.name, "manual_checks"],
                                                -summary_t.at[c.name, "accuracy"]))
        CHOSEN.write_text(json.dumps(asdict(best), indent=2) + "\n")
        print(f"\nChosen on tune -> {CHOSEN.relative_to(ROOT)}: {best.name}")


if __name__ == "__main__":
    main()
