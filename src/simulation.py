"""Replay a year of transactions month by month (test-then-train), with a simulated user.

    for each month:
        1. decide   every row, using only memory from earlier months
        2. score    (done later from the log: label vs true_category)
        3. review   the simulated user fixes or answers every row
        4. store    reviewed rows go into memory, per the storage policy

The simulated user never rubber-stamps: whatever it is shown, its final label is true_category.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from memory import Config, Memory, decide, should_store, source_of, store, system_text

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "cache"


def load_transactions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["month"] = [d.month for d in df["date"]]
    df["text"] = [system_text(d, n) for d, n in zip(df["description"], df["note"])]
    return df.sort_values(["date", "txn_id"], kind="stable").reset_index(drop=True)


def frozen_outputs(texts: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Frozen model labels, confidences and vectors for each text, cached on disk.

    The cache key is a hash of the texts, so changing normalize() or the data recomputes it.
    """
    key = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:12]
    path = CACHE / f"frozen_{key}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return list(z["labels"]), z["confs"], z["embeddings"]
    from frozen_base import FrozenModel   # slow import; only needed on a cache miss
    labels, confs, emb = FrozenModel().encode(texts)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, labels=np.array(labels), confs=np.array(confs, dtype=np.float32), embeddings=emb)
    return labels, np.array(confs, dtype=np.float32), emb


def simulated_user(shown_label: str, true_label: str) -> str:
    """The careful user: always ends with the right label."""
    return true_label


def run_year(df: pd.DataFrame, labels: list[str], confs: np.ndarray, emb: np.ndarray,
             config: Config) -> pd.DataFrame:
    """Replay every month in order. Returns one log row per transaction."""
    memory = Memory()
    rng = np.random.default_rng(config.seed)
    log = []
    for month, rows in df.groupby("month", sort=True):
        decided = []
        for i in rows.index:                       # 1. decide with last month's memory
            d = decide(emb[i], rows.at[i, "date"], labels[i], float(confs[i]), memory, config)
            decided.append((i, d))
        for i, d in decided:                       # 3-4. review, then store
            final = simulated_user(d.label, df.at[i, "true_category"])
            source = source_of(d, final)
            stored = should_store(source, config)
            outcome = "not stored"
            if stored:
                outcome = store(memory, int(df.at[i, "txn_id"]), df.at[i, "date"], final, source,
                                emb[i], config, rng)
            log.append({
                "config": config.name, "txn_id": int(df.at[i, "txn_id"]), "date": df.at[i, "date"],
                "month": month, "description": df.at[i, "description"], "note": df.at[i, "note"],
                "text": df.at[i, "text"],
                "true_label": df.at[i, "true_category"],
                "frozen_label": labels[i], "frozen_conf": round(float(confs[i]), 3),
                "label": d.label, "flagged": d.flagged, "path": d.path,
                "share": round(d.share, 3), "neighbours": d.neighbours,
                "final_label": final, "source": source, "stored": stored, "store_outcome": outcome,
            })
        for row in log[-len(decided):]:
            row["memory_size"] = len(memory)
    return pd.DataFrame(log)
