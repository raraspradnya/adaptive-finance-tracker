"""Frozen base model: categorizes bank transactions and never learns.

Wraps DoDataThings/distilbert-us-transaction-classifier (v1), pinned to one
commit so the "frozen" model cannot change underneath an experiment.

Usage:
    python src/frozen_base.py path/to/transactions.csv [--out predictions.csv]

The CSV needs a `description` column. If it also has `true_category`, the script
reports accuracy (rows whose category the model does not have count as wrong).
"""

import argparse

import numpy as np
import pandas as pd
import torch
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

    def encode(self, descriptions, batch_size=32):
        """One pass per text: (labels, confidences, embeddings).

        embeddings is a float32 array of shape (n, 768), L2-normalized, so the cosine
        similarity of two rows is just their dot product. It is the [CLS] vector the
        classification head reads, so the label and the embedding come from the same pass.
        """
        model, tok = self._clf.model, self._clf.tokenizer
        id2label = model.config.id2label
        texts = list(descriptions)
        labels, confs, embs = [], [], []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = tok(texts[i:i + batch_size], padding=True, truncation=True, return_tensors="pt")
                out = model(**batch, output_hidden_states=True)
                probs = out.logits.softmax(-1)
                conf, idx = probs.max(-1)
                labels += [id2label[int(j)] for j in idx]
                confs += [round(float(c), 3) for c in conf]
                embs.append(out.hidden_states[-1][:, 0].float().numpy())
        if not embs:
            return [], [], np.zeros((0, model.config.dim), dtype=np.float32)
        emb = np.concatenate(embs).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return labels, confs, emb


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv")
    parser.add_argument("--out", help="where to write predictions (optional)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    model = FrozenModel()
    preds = model.predict(df["description"])
    df["predicted"] = [p[0] for p in preds]
    df["confidence"] = [p[1] for p in preds]

    if "true_category" in df:
        df["correct"] = df["predicted"] == df["true_category"]
        print(f"Accuracy: {df['correct'].mean():.1%} ({df['correct'].sum()}/{len(df)})")
        print(f"Mean confidence when right: {df.loc[df['correct'], 'confidence'].mean():.2f}")
        print(f"Mean confidence when wrong: {df.loc[~df['correct'], 'confidence'].mean():.2f}")

        if "date" in df:
            monthly = df.groupby(pd.to_datetime(df["date"]).dt.month.rename("month"))["correct"].agg(["mean", "sum", "count"])
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
