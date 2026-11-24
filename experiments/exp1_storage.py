"""Experiment 1: what to store. (EXPERIMENTS.md)

Same reader in every condition (similarity-only kNN). Only the storage policy changes.

Run:  .venv/bin/python experiments/exp1_storage.py [path/to/transactions.csv]
Writes results/exp1/ : one log per condition, summary.csv, monthly.csv, cases.csv, two charts.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from memory import Config  # noqa: E402
from metrics import line_chart, monthly, summary  # noqa: E402
from simulation import frozen_outputs, load_transactions, run_year  # noqa: E402

CONDITIONS = [
    Config(name="Frozen only", storage="none"),
    Config(name="Corrections only", storage="corrections"),
    Config(name="Corrections + answered", storage="corrections_asked"),
    Config(name="Everything", storage="everything"),
]

# Dev shifts, used only to read the results (never by the system).
EVENTS = {5: "gift card", 7: "coffee", 10: "bank", 11: "Ashley"}
COFFEE = "PEET|BLUE BOTTLE|STRADA|PHILZ"


def case_table(logs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The three cases from the predictions table, month by month: share of rows labeled right."""
    rows = []
    for name, log in logs.items():
        cases = {
            "Coffee (should be Coffee)": log["description"].str.contains(COFFEE) & (log["month"] >= 7),
            "Trader Joe's (should be Groceries)": log["description"].str.contains("TRADER JOE")
                                                  & (log["true_label"] == "Groceries") & (log["month"] >= 5),
            "Ashley (should be Social)": log["description"].str.upper().str.contains("ASHLEY")
                                         & (log["month"] >= 11),
        }
        for case, mask in cases.items():
            sub = log[mask]
            for month, g in sub.groupby("month"):
                rows.append({"condition": name, "case": case, "month": month,
                             "right": int((g["label"] == g["true_label"]).sum()), "rows": len(g)})
    t = pd.DataFrame(rows)
    t["cell"] = t["right"].astype(str) + "/" + t["rows"].astype(str)
    t = t.pivot_table(index=["case", "condition"], columns="month", values="cell", aggfunc="first",
                      sort=False).fillna("")
    return t[sorted(t.columns)]


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "transactions_2026.csv"
    out = ROOT / "results" / "exp1"
    out.mkdir(parents=True, exist_ok=True)

    df = load_transactions(data)
    labels, confs, emb = frozen_outputs(list(df["text"]))

    logs, months, sums = {}, {}, {}
    for config in CONDITIONS:
        log = run_year(df, labels, confs, emb, config)
        slug = config.storage
        log.to_csv(out / f"log_{slug}.csv", index=False)
        logs[config.name] = log
        months[config.name] = monthly(log)
        sums[config.name] = summary(log)

    summary_t = pd.DataFrame(sums).T
    monthly_t = pd.concat(months, names=["condition", "month"])
    cases_t = case_table(logs)
    summary_t.to_csv(out / "summary.csv")
    monthly_t.to_csv(out / "monthly.csv")
    cases_t.to_csv(out / "cases.csv")

    # adaptation errors (coffee from Jul, Ashley from Nov) and spread errors (TJ labeled Gifts)
    extra = {}
    for name, log in logs.items():
        wrong = log["label"] != log["true_label"]
        coffee = log["description"].str.contains(COFFEE) & (log["month"] >= 7)
        ashley = log["description"].str.upper().str.contains("ASHLEY") & (log["month"] >= 11)
        tj = log["description"].str.contains("TRADER JOE") & (log["true_label"] == "Groceries")
        extra[name] = {"adaptation_errors": int((wrong & (coffee | ashley)).sum()),
                       "spread_errors": int((tj & (log["label"] == "Gifts")).sum())}
    summary_t = summary_t.join(pd.DataFrame(extra).T)
    summary_t.to_csv(out / "summary.csv")

    pd.set_option("display.width", 200)
    print("Summary (whole year)\n")
    print(summary_t[["accuracy", "manual_checks", "flagged", "corrected", "from_memory", "flips", "fixes",
                     "adaptation_errors", "spread_errors", "memory_size"]].to_string())
    print("\nAccuracy by month (%)\n")
    print(monthly_t["accuracy"].unstack("month").to_string())
    print("\nManual checks by month\n")
    print(monthly_t["manual_checks"].unstack("month").to_string())
    print("\nCases: rows labeled right / rows, by month\n")
    print(cases_t.to_string())

    line_chart(monthly_t["manual_checks"].unstack("condition"), "Manual checks per month (fewer is better)",
               "rows the user had to fix or answer", out / "manual_checks.png", baseline="Frozen only",
               events=EVENTS)
    line_chart(monthly_t["accuracy"].unstack("condition"), "Accuracy per month, before review",
               "% of rows labeled right", out / "accuracy.png", baseline="Frozen only",
               events=EVENTS, ylim=(0, 100))
    print(f"\nWrote {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
