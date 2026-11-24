"""Review one month of transactions by hand, the way the simulated user will.

You play the user. For each row you see what the system shows (the frozen model's label and
confidence, flagged "Check this" when the model is unsure) and decide:

    Enter        keep the shown label
    a number     change it to that category (type ? to list them)
    a name       change it to that category, or create it if it is new (e.g. Coffee)
    n            add your own note to this row first
    q            stop and save what you have reviewed so far

Memory starts empty, so every row is decided by the frozen model (the same reader as the
experiments, with nothing stored yet). Each reviewed row is written with its source
(asked / corrected / accepted).

Run:  .venv/bin/python tools/review_month_by_hand.py --month 7
      add --show-truth to see the dataset's true label after each answer
Writes results/manual_review_m<MONTH>.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from memory import Config, Memory, decide, source_of  # noqa: E402
from simulation import frozen_outputs, load_transactions  # noqa: E402
DATA = ROOT / "data" / "transactions_2026.csv"
MODEL_CATEGORIES = ["Education", "Entertainment", "Fees", "Groceries", "Healthcare", "Housing", "Income",
                    "Insurance", "Personal Care", "Restaurants", "Shopping", "Subscription", "Transfer",
                    "Transportation", "Travel", "Utilities"]


def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--month", type=int, required=True, help="1-12")
    ap.add_argument("--show-truth", action="store_true", help="show the dataset's true label after each answer")
    args = ap.parse_args()

    df = load_transactions(DATA)
    labels, confs, emb = frozen_outputs(list(df["text"]))
    rows = df[df["month"] == args.month]
    if rows.empty:
        sys.exit(f"No transactions in month {args.month}.")
    empty, config = Memory(), Config()
    cats = list(MODEL_CATEGORIES)   # the user adds her own (Social, Coffee, ...) as she goes

    def show_cats():
        print("  " + "   ".join(f"{i + 1}={c}" for i, c in enumerate(cats)))

    print(f"\nMonth {args.month}: {len(rows)} transactions. Enter = keep, number or name = change, "
          f"? = categories, n = note, q = quit.")
    show_cats()

    out = []
    for n, (i, r) in enumerate(rows.iterrows()):
        label, conf = labels[i], float(confs[i])
        decision = decide(emb[i], r["date"], label, conf, empty, config)
        money = f"+${r['amount']:,.2f} in" if r["amount"] > 0 else f"-${-r['amount']:,.2f} out"
        flag = "   ⚠ Check this" if decision.flagged else ""
        bank_note = f'   note: "{r["note"]}"' if r["note"] else ""
        print(f"\n[{n + 1}/{len(rows)}] {r['date']}  {money}  {r['description']}{bank_note}")
        print(f"  shown: {decision.label}  (model confidence {conf:.2f}){flag}")

        note = r["note"]
        while True:
            ans = ask("  > ")
            if ans == "?":
                show_cats()
                continue
            if ans.lower() == "n":
                note = ask("  note: ")
                continue
            break
        if ans.lower() == "q":
            break
        if ans == "":
            final = decision.label
        elif ans.isdigit() and 1 <= int(ans) <= len(cats):
            final = cats[int(ans) - 1]
        else:
            final = next((c for c in cats if c.lower() == ans.lower()), ans.strip().title())
            if final not in cats:
                cats.append(final)
                print(f"  new category: {final}")

        src = source_of(decision, final)
        msg = f"  stored as: {final}  [{src}]"
        if args.show_truth:
            msg += f"   dataset says: {r['true_category']}"
        print(msg)
        out.append({
            "txn_id": r["txn_id"], "month": r["month"], "date": r["date"],
            "description": r["description"], "amount": r["amount"],
            "note": note,
            "frozen_label": label, "frozen_confidence": conf,
            "shown_label": decision.label, "flagged": decision.flagged, "path": decision.path,
            "final_label": final, "source": src,
            "true_category": r["true_category"],
        })

    if not out:
        print("\nNothing reviewed.")
        return
    res = pd.DataFrame(out)
    path = ROOT / "results" / f"manual_review_m{args.month}.csv"
    res.to_csv(path, index=False)
    checks = (res["source"].isin(["corrected", "asked"])).sum()
    print(f"\nReviewed {len(res)} rows. Manual checks (corrected + asked): {checks}.")
    print(res["source"].value_counts().to_string())
    print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
