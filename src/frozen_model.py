"""Frozen base model: categorizes bank transactions and never learns.

Wraps DoDataThings/distilbert-us-transaction-classifier (v1), pinned to one
commit so the "frozen" model cannot change underneath an experiment.

Usage:
    python src/frozen_model.py path/to/transactions.csv [--out predictions.csv]

The CSV needs a `raw_description` column. If it also has `true_category`
(using the model's category names), the script reports accuracy.
"""

import argparse

import pandas as pd
from transformers import pipeline

MODEL_ID = "DoDataThings/distilbert-us-transaction-classifier"
MODEL_REVISION = "e9bf5045db5d7f35d1fdb8a40d518f1658684b1f"  # pinned: keeps the model frozen


class FrozenModel:
    def __init__(self):
        self._clf = pipeline(
            "text-classification",
            model=MODEL_ID,
            revision=MODEL_REVISION,
            device="cpu",
        )
        self.categories = sorted(self._clf.model.config.id2label.values())

    def predict(self, descriptions):
        """Return (category, confidence) for each raw bank description."""
        results = self._clf(list(descriptions), batch_size=32, truncation=True)
        return [(r["label"], round(float(r["score"]), 3)) for r in results]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv")
    parser.add_argument("--out", help="where to write predictions (optional)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    model = FrozenModel()
    preds = model.predict(df["raw_description"])
    df["predicted"] = [p[0] for p in preds]
    df["confidence"] = [p[1] for p in preds]

    if "true_category" in df:
        df["correct"] = df["predicted"] == df["true_category"]
        print(f"Accuracy: {df['correct'].mean():.1%} ({df['correct'].sum()}/{len(df)})")
        print(f"Mean confidence when right: {df.loc[df['correct'], 'confidence'].mean():.2f}")
        print(f"Mean confidence when wrong: {df.loc[~df['correct'], 'confidence'].mean():.2f}")

        if "month" in df:
            monthly = df.groupby("month")["correct"].agg(["mean", "sum", "count"])
            monthly["mean"] = (monthly["mean"] * 100).round(1)
            print("\nAccuracy by month (%):")
            print(monthly.rename(columns={"mean": "accuracy", "sum": "correct", "count": "rows"}).to_string())

        wrong = df.loc[~df["correct"]]
        if len(wrong):
            print("\nMost common mistakes:")
            top = (wrong.groupby(["true_category", "predicted"]).size()
                   .sort_values(ascending=False).head(15).rename("rows"))
            print(top.to_string())

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
