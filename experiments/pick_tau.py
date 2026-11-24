"""Pick the similarity cutoff (tau) on dev, once, before any experiment.

For every pair of rows, how often do two rows that are at least this similar share a label?
A good cutoff keeps that high while still giving most rows at least one earlier neighbour.

Run:  .venv/bin/python experiments/pick_tau.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from simulation import frozen_outputs, load_transactions  # noqa: E402

df = load_transactions(ROOT / "data" / "transactions_2026.csv")
_, _, emb = frozen_outputs(list(df["text"]))
labels = df["true_category"].to_numpy()
sims = emb @ emb.T
earlier = np.tril(np.ones_like(sims, dtype=bool), k=-1)   # pair (i, j) with j before i
same = labels[:, None] == labels[None, :]

rows = []
for tau in [0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97]:
    close = earlier & (sims >= tau)
    rows.append({
        "tau": tau,
        "close pairs": int(close.sum()),
        "same label %": round(100 * same[close].mean(), 1),
        "rows with an earlier neighbour %": round(100 * close.any(axis=1).mean(), 1),
    })
print(pd.DataFrame(rows).to_string(index=False))
