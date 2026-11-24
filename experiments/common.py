"""Shared pieces for the experiment scripts: which data to run on, running a list of conditions,
printing, charts, and the dev-year story table.

Splits (EXPERIMENTS.md): every choice is made on tune. Val is generated and run once, after
Experiments 1-3 are settled. Test is run only by the final script, on the combined best setup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from memory import Config  # noqa: E402
from metrics import line_chart, monthly, summary  # noqa: E402
from simulation import frozen_outputs, load_transactions, run_year  # noqa: E402

TUNE = ROOT / "data" / "transactions_2026.csv"
VAL = ROOT / "data" / "transactions_val.csv"     # data_generator.py --val, only at the end

SHOWN = ["accuracy", "manual_checks", "flagged", "corrected", "from_memory", "flips",
         "adaptation_errors", "adaptation_rows", "spread_errors", "spread_rows", "memory_size"]


def pick_split(description: str) -> tuple[str, Path]:
    """--val runs on the val year; otherwise tune."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--val", action="store_true", help="run on the val year (once, after Experiments 1-3)")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    if not args.val:
        return "tune", TUNE
    if not VAL.exists():
        sys.exit("No val year yet. Generate it with: .venv/bin/python src/data_generator.py --val")
    return "val", VAL


def run_conditions(data: Path, configs: list[Config], out: Path | None = None):
    """Replay the year once per config. Returns (logs by name, summary table, monthly table).
    With out set, writes each log as out/log_<slug>.csv."""
    df = load_transactions(data)
    labels, confs, emb = frozen_outputs(list(df["text"]))
    logs, months, sums = {}, {}, {}
    for config in configs:
        log = run_year(df, labels, confs, emb, config)
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            log.to_csv(out / f"log_{slug(config.name)}.csv", index=False)
        logs[config.name] = log
        months[config.name] = monthly(log)
        sums[config.name] = summary(log)
    return logs, pd.DataFrame(sums).T, pd.concat(months, names=["condition", "month"])


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_").replace("__", "_")


def report(split: str, data: Path, out: Path, logs, summary_t, monthly_t, baseline="Frozen only"):
    """Save the tables and charts, and print them."""
    out.mkdir(parents=True, exist_ok=True)
    summary_t.to_csv(out / "summary.csv")
    monthly_t.to_csv(out / "monthly.csv")
    print(f"\n=== {split}: {data.name} ===\n")
    print(summary_t[SHOWN].to_string())
    print("\nAccuracy by month (%)\n")
    print(monthly_t["accuracy"].unstack("month").to_string())
    print("\nManual checks by month\n")
    print(monthly_t["manual_checks"].unstack("month").to_string())
    if split == "tune":
        cases = dev_case_table(logs)
        cases.to_csv(out / "cases.csv")
        print("\nDev stories: rows labeled right / rows, by month\n")
        print(cases.to_string())
    events = chart_events(data)
    order = list(logs)   # keep the condition order (and so the colors) as listed
    line_chart(monthly_t["manual_checks"].unstack("condition")[order],
               f"Manual checks per month, {split} (fewer is better)", "rows the user had to fix or answer",
               out / "manual_checks.png", baseline=baseline, events=events)
    line_chart(monthly_t["accuracy"].unstack("condition")[order], f"Accuracy per month before review, {split}",
               "% of rows labeled right", out / "accuracy.png", baseline=baseline, events=events,
               ylim=(0, 100))


def chart_events(data: Path) -> dict[int, str]:
    """Chart markers from <data>.events.json, if the generator wrote one (none for real data)."""
    path = data.with_suffix(".events.json")
    if not path.exists():
        return {}
    ev = json.loads(path.read_text())
    marks: dict[int, list[str]] = {}
    for month, text in [(ev["gift_card"][0], "gift card"), (ev["coffee_switch"], "coffee"),
                        (ev["bank_switch"], "bank"), (ev["ashley_moves_out"], "Ashley")]:
        marks.setdefault(month, []).append(text)
    return {m: " + ".join(t) for m, t in marks.items()}


def dev_case_table(logs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The tune year's three stories, month by month (rows labeled right / rows). Tune only:
    it names tune-year shops and months."""
    coffee = "PEET|BLUE BOTTLE|STRADA|PHILZ"
    rows = []
    for name, log in logs.items():
        desc = log["description"].str.upper()
        cases = {
            "Coffee (should be Coffee)": desc.str.contains(coffee) & (log["month"] >= 7),
            "Trader Joe's (should be Groceries)": desc.str.contains("TRADER JOE")
                                                  & (log["true_label"] == "Groceries") & (log["month"] >= 5),
            "Ashley (should be Social)": desc.str.contains("ASHLEY") & (log["month"] >= 11),
        }
        for case, mask in cases.items():
            for month, g in log[mask].groupby("month"):
                rows.append({"condition": name, "case": case, "month": month,
                             "cell": f"{(g['label'] == g['true_label']).sum()}/{len(g)}"})
    t = pd.DataFrame(rows).pivot_table(index=["case", "condition"], columns="month", values="cell",
                                       aggfunc="first", sort=False).fillna("")
    return t[sorted(t.columns)]
