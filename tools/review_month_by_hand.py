"""Review one month of transactions by hand, the way the simulated user will.

You play the user. For each row you see what the system shows (the frozen model's label and
confidence, flagged "unsure" for transfers) and decide:

    Enter        keep the shown label
    a number     change it to that category (type ? to list them)
    a name       change it to that category, or create it if it is new (e.g. Coffee)
    n            add your own note to this row first
    q            stop and save what you have reviewed so far

There is no memory yet, so every row is decided by the frozen model. Each reviewed row is
written as it would be stored in memory, with its source (corrected / asked / model accepted).

Run:  .venv/bin/python tools/review_month_by_hand.py --month 7
      add --show-truth to see the dataset's true label after each answer
Writes results/manual_review_m<MONTH>.csv
"""

import argparse
import sys
from pathlib import Path as FsPath

import pandas as pd

ROOT = FsPath(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from memory import Decision, Path, Review, source_of  # noqa: E402
DATA = ROOT / "data" / "transactions_2026.csv"
CACHED = ROOT / "results" / "frozen_predictions_2026.csv"
MODEL_CATEGORIES = ["Education", "Entertainment", "Fees", "Groceries", "Healthcare", "Housing", "Income",
                    "Insurance", "Personal Care", "Restaurants", "Shopping", "Subscription", "Transfer",
                    "Transportation", "Travel", "Utilities"]

TRANSFER_WORDS = ("VENMO", "ZELLE", "CASH APP", "PAYPAL")
BANK_WORDS = ("PAYROLL", "AUTOPAY PPD", "SERVICE FEE", "DIR DEP")


def channel_of(raw):
    """card / transfer / bank, read from the bank text only."""
    up = raw.upper()
    if any(w in up for w in TRANSFER_WORDS):
        return "transfer"
    if any(w in up for w in BANK_WORDS):
        return "bank"
    return "card"


def frozen_predictions(rows):
    """Use the saved predictions if they cover these rows, otherwise run the model."""
    if CACHED.exists():
        cached = pd.read_csv(CACHED).set_index("txn_id")
        if set(rows["txn_id"]) <= set(cached.index):
            return [(cached.at[t, "predicted"], float(cached.at[t, "confidence"])) for t in rows["txn_id"]]
    from frozen_base import FrozenModel
    print("Running the frozen model...")
    return FrozenModel().predict(rows["description"])


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

    df = pd.read_csv(DATA, keep_default_na=False)
    df["month"] = pd.to_datetime(df["date"]).dt.month
    rows = df[df["month"] == args.month].reset_index(drop=True)
    if rows.empty:
        sys.exit(f"No transactions in month {args.month}.")
    preds = frozen_predictions(rows)
    cats = list(MODEL_CATEGORIES)   # the user adds her own (Social, Coffee, ...) as she goes

    def show_cats():
        print("  " + "   ".join(f"{i + 1}={c}" for i, c in enumerate(cats)))

    print(f"\nMonth {args.month}: {len(rows)} transactions. Enter = keep, number or name = change, "
          f"? = categories, n = note, q = quit.")
    show_cats()

    out = []
    for i, r in rows.iterrows():
        label, conf = preds[i]
        channel = channel_of(r["description"])
        decision = _decide(channel, label)
        money = f"+${r['amount']:,.2f} in" if r["amount"] > 0 else f"-${-r['amount']:,.2f} out"
        flag = "   ⚠ unsure, please check" if decision.flagged else ""
        bank_note = f'   note: "{r["note"]}"' if r["note"] else ""
        print(f"\n[{i + 1}/{len(rows)}] {r['date']}  {money}  {r['description']}  ({channel}){bank_note}")
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

        review = Review(final, note)
        src = source_of(decision, review)
        msg = f"  stored as: {final}  [{src.value}]"
        if args.show_truth:
            msg += f"   dataset says: {r['true_category']}"
        print(msg)
        out.append({
            "txn_id": r["txn_id"], "month": r["month"], "date": r["date"],
            "description": r["description"], "amount": r["amount"],
            "channel": channel, "note": note,
            "frozen_label": label, "frozen_confidence": conf,
            "shown_label": decision.label, "flagged": decision.flagged, "path": decision.path.value,
            "final_label": final, "source": src.value,
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


def _decide(channel, label):
    """The frozen-only read policy: transfers are flagged, everything else is shown as-is."""
    if channel == "transfer":
        return Decision(label, flagged=True, path=Path.ASK)
    return Decision(label, flagged=False, path=Path.FROZEN)


if __name__ == "__main__":
    main()
