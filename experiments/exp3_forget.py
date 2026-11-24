"""Experiment 3: what to forget. (EXPERIMENTS.md)

Builds on Experiment 2's choice (read from results/exp2/chosen.json): memory stores corrected and
answered rows, votes count similarity. Now memory can hold at most K rows. When it is full, an
eviction rule decides which row goes:

    oldest first          the row stored longest ago
    random                reservoir sampling: every row seen so far has the same chance to stay
                          (5 seeds, averaged)
    least useful          the row whose vote counts least right now (weight x freshness). With no
                          freshness and every stored row a user's answer, all rows count the same,
                          so this falls back to oldest first
    least useful + merge  as above, and a new row first merges into a near-identical stored row
                          with the same label (similarity >= 0.99); the merged row's weight is the
                          number of rows in it, so rows the user confirmed again and again stay

The year stores up to ~48 rows, so K = 10 / 20 / 30 are real limits.

Choice for the combined best (made on tune, saved to results/exp3/chosen.json): the rule with the
fewest manual checks summed over the three K, at the smallest K whose manual checks stay within
10% of unlimited memory.

Run:  .venv/bin/python experiments/exp3_forget.py         tune
      .venv/bin/python experiments/exp3_forget.py --val   val, once, after Experiments 1-3
"""

import json
from dataclasses import asdict, replace

import pandas as pd

from common import ROOT, SHOWN, pick_split, report, run_conditions, slug
from memory import Config
from metrics import heatmap

SIZES = [10, 20, 30]
SEEDS = range(5)
MERGE = 0.99
RULES = ["Oldest first", "Random", "Least useful", "Least useful + merge"]
CHOSEN = ROOT / "results" / "exp3" / "chosen.json"


def base_reader() -> Config:
    path = ROOT / "results" / "exp2" / "chosen.json"
    if not path.exists():
        raise SystemExit("Run Experiment 2 on tune first: Experiment 3 builds on its choice.")
    return Config(**json.loads(path.read_text()))


def conditions(base: Config):
    """All configs, plus a map from each config name to its (rule, K) group."""
    frozen = Config(name="Frozen only", storage="none")
    configs = [frozen, replace(base, name="Unlimited"),
               replace(base, name="Unlimited + merge", merge_above=MERGE)]
    group = {"Frozen only": ("Frozen only", None), "Unlimited": ("Unlimited", None),
             "Unlimited + merge": ("Unlimited + merge", None)}
    for k in SIZES:
        variants = {
            "Oldest first": [replace(base, capacity=k, eviction="oldest")],
            "Random": [replace(base, capacity=k, eviction="random", seed=s) for s in SEEDS],
            "Least useful": [replace(base, capacity=k, eviction="least_useful")],
            "Least useful + merge": [replace(base, capacity=k, eviction="least_useful", merge_above=MERGE)],
        }
        for rule, cs in variants.items():
            for c in cs:
                name = f"{rule}, K={k}" + (f" (seed {c.seed})" if rule == "Random" else "")
                configs.append(replace(c, name=name))
                group[name] = (rule, k)
    return configs, group


def averaged(group, summary_t, monthly_t):
    """Average the random seeds: one summary row and one monthly curve per (rule, K)."""
    key = {n: (r if k is None else f"{r}, K={k}") for n, (r, k) in group.items()}
    summ = summary_t.groupby(summary_t.index.map(key), sort=False).mean(numeric_only=True)
    spread = summary_t.groupby(summary_t.index.map(key), sort=False)[["accuracy", "manual_checks"]].agg(["min", "max"])
    m = monthly_t.reset_index()
    m["condition"] = m["condition"].map(key)
    mon = m.groupby(["condition", "month"], sort=False).mean(numeric_only=True)
    return summ.round(1), spread, mon


def choose(group, summ) -> tuple[str, int | None]:
    rows = {(r, k): summ.loc[f"{r}, K={k}"] for r in RULES for k in SIZES}
    rule = min(RULES, key=lambda r: sum(rows[(r, k)]["manual_checks"] for k in SIZES))
    limit = 1.10 * summ.loc["Unlimited", "manual_checks"]
    k = next((k for k in SIZES if rows[(rule, k)]["manual_checks"] <= limit), None)
    return rule, k


def run(split, data):
    out = ROOT / "results" / "exp3" / split
    base = base_reader()
    configs, group = conditions(base)
    logs, summary_t, monthly_t = run_conditions(data, configs)
    summ, spread, mon = averaged(group, summary_t, monthly_t)
    out.mkdir(parents=True, exist_ok=True)
    summary_t.to_csv(out / "all_runs.csv")
    summ.to_csv(out / "grid.csv")
    spread.to_csv(out / "random_seed_range.csv")

    print(f"\nBase reader (Experiment 2): {base.name}\n")
    print(summ[["accuracy", "manual_checks", "adaptation_errors", "spread_errors", "long_gap_rows",
                "long_gap_accuracy", "memory_size"]].to_string())
    print("\nRandom eviction across 5 seeds (min / max)\n")
    print(spread.to_string())

    # rule x K tables (+ heatmaps on tune)
    for metric, label, fmt in [("manual_checks", "Manual checks per year", "{:.0f}"),
                               ("accuracy", "Accuracy (%)", "{:.1f}"),
                               ("long_gap_accuracy", "Long-gap accuracy (%)", "{:.0f}")]:
        table = pd.DataFrame({f"K={k}": [summ.loc[f"{r}, K={k}", metric] for r in RULES] for k in SIZES},
                             index=RULES)
        table["Unlimited"] = [summ.loc["Unlimited", metric]] * 3 + [summ.loc["Unlimited + merge", metric]]
        print(f"\n{label}: eviction rule x memory size\n")
        print(table.to_string())
        heatmap(table, f"{label}, eviction rule × memory size ({split})", out / f"heatmap_{metric}.png",
                xlabel="memory size (rows)", ylabel="", fmt=fmt)

    if split == "tune":
        rule, k = choose(group, summ)
        best = next(c for c in configs if group[c.name] == (rule, k)) if k else replace(base, name="Unlimited")
        if rule == "Random":
            best = replace(best, seed=0)
        best = replace(best, name=f"Combined best ({rule}, K={k})" if k else "Combined best (unlimited)")
        CHOSEN.write_text(json.dumps(asdict(best), indent=2) + "\n")
        print(f"\nChosen on tune -> {CHOSEN.relative_to(ROOT)}: {best.name}")

    # monthly charts: frozen, unlimited, and each rule at the tightest limit (where they differ most)
    show = ["Frozen only", "Unlimited"] + [f"{r}, K={SIZES[0]}" for r in RULES]
    shown_logs = {n: logs[next(c for c in group if
                               (group[c][0] if group[c][1] is None else f"{group[c][0]}, K={group[c][1]}") == n)]
                  for n in show}
    for n, log in shown_logs.items():
        log.to_csv(out / f"log_{slug(n)}.csv", index=False)
    report(split, data, out, shown_logs, summ.loc[show], mon.loc[show])


def main():
    split, data = pick_split(__doc__.splitlines()[0])
    run(split, data)


if __name__ == "__main__":
    main()
