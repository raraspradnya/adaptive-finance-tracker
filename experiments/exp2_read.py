"""Experiment 2: how to read. (EXPERIMENTS.md)

Memory stores everything. What changes is how much each neighbour's vote counts:

    vote = similarity x trust x freshness
    trust       rows the user changed or answered count 1.0; rows left alone count less
    freshness   a vote halves every N months

On tune, every trust / freshness setting is tried, the best one per family is chosen (fewest
manual checks, then highest accuracy) and saved to results/exp2/chosen.json. Val only reads that
file, so it cannot re-tune anything.

Run:  .venv/bin/python experiments/exp2_read.py         tune: try the grid, choose, save
      .venv/bin/python experiments/exp2_read.py --val   val, once, after Experiments 1-3

Writes results/exp2/<split>/ : logs of the finalists, summary.csv, monthly.csv, two charts,
(tune only) grid.csv with every setting tried, cases.csv and heatmaps of the trust x freshness grid.
"""

import json
from dataclasses import asdict, replace

import pandas as pd

from common import ROOT, TUNE, pick_split, report, run_conditions, slug
from memory import Config
from metrics import heatmap

TRUST = [0.1, 0.25, 0.5, 0.75]       # trust of rows the user left alone
HALF_LIFE = [1, 3, 6]                # months for a vote to halve

FROZEN = Config(name="Frozen only", storage="none")
SIMILARITY = Config(name="Similarity", storage="everything")
REFERENCE = Config(name="Exp 1 winner", storage="corrections_asked")   # corrections + answered

CHOSEN = ROOT / "results" / "exp2" / "chosen.json"


def families() -> dict[str, list[Config]]:
    return {
        "+ Trust": [replace(SIMILARITY, name=f"+ Trust {t}", trust_accepted=t) for t in TRUST],
        "+ Freshness": [replace(SIMILARITY, name=f"+ Freshness {h} mo", half_life_months=h)
                        for h in HALF_LIFE],
        "+ Both": [replace(SIMILARITY, name=f"+ Both {t}, {h} mo", trust_accepted=t, half_life_months=h)
                   for t in TRUST for h in HALF_LIFE],
        # Experiment 1's storage with freshness (trust does nothing here: no left-alone rows stored)
        "Exp 1 winner + Freshness": [replace(REFERENCE, name=f"Exp 1 winner + Freshness {h} mo",
                                             half_life_months=h) for h in HALF_LIFE],
    }


def best(configs: list[Config], summary_t: pd.DataFrame) -> Config:
    """Fewest manual checks; ties broken by higher accuracy."""
    return min(configs, key=lambda c: (summary_t.at[c.name, "manual_checks"], -summary_t.at[c.name, "accuracy"]))


def tune():
    out = ROOT / "results" / "exp2" / "tune"
    fams = families()
    everything = [FROZEN, SIMILARITY, REFERENCE] + [c for cs in fams.values() for c in cs]
    logs, summary_t, monthly_t = run_conditions(TUNE, everything)

    # every setting tried, for the record
    grid = summary_t.copy()
    grid.insert(0, "trust", [c.trust_accepted for c in everything])
    grid.insert(1, "half_life_months", [c.half_life_months for c in everything])
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "grid.csv")
    print("\nEvery setting tried on tune\n")
    print(grid[["trust", "half_life_months", "accuracy", "manual_checks", "adaptation_errors",
                "spread_errors", "flips"]].to_string())

    # choose one per family, on tune only, and save the choice for val
    finalists = [FROZEN, SIMILARITY] + [best(cs, summary_t) for cs in fams.values()] + [REFERENCE]
    CHOSEN.write_text(json.dumps([asdict(c) for c in finalists], indent=2) + "\n")
    print(f"\nChosen on tune -> {CHOSEN.relative_to(ROOT)}: " + ", ".join(c.name for c in finalists[2:-1]))

    names = [c.name for c in finalists]
    for c in finalists:
        logs[c.name].to_csv(out / f"log_{slug(c.name)}.csv", index=False)
    fin_logs = {n: logs[n] for n in names}
    fin_monthly = monthly_t[monthly_t.index.get_level_values("condition").isin(names)]
    report("tune", TUNE, out, fin_logs, summary_t.loc[names], fin_monthly)

    both = grid.loc[[c.name for c in fams["+ Both"]]]
    for metric, label in [("manual_checks", "Manual checks per year"), ("accuracy", "Accuracy (%)")]:
        table = both.pivot(index="trust", columns="half_life_months", values=metric)
        table.columns = [f"{int(h)} mo" for h in table.columns]
        heatmap(table, f"{label}, trust × freshness (tune)", out / f"heatmap_{metric}.png",
                xlabel="freshness: a vote halves every …", ylabel="trust of rows left alone",
                fmt="{:.0f}" if metric == "manual_checks" else "{:.1f}")


def val(data):
    if not CHOSEN.exists():
        raise SystemExit("Run Experiment 2 on tune first: it chooses the settings val uses.")
    configs = [Config(**c) for c in json.loads(CHOSEN.read_text())]
    out = ROOT / "results" / "exp2" / "val"
    logs, summary_t, monthly_t = run_conditions(data, configs, out)
    report("val", data, out, logs, summary_t, monthly_t)


def main():
    split, data = pick_split(__doc__.splitlines()[0])
    tune() if split == "tune" else val(data)


if __name__ == "__main__":
    main()
