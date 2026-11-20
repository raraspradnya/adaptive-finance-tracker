"""Checks on the shared contract in src/memory.py. Run: .venv/bin/python -m pytest -q"""

import sys
from datetime import date
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, str(FsPath(__file__).resolve().parent.parent / "src"))

from memory import (  # noqa: E402
    Config, Decision, FrozenOutput, Memory, Path, Review, Scenario, Source, Transaction,
    decide, decide_frozen_only, source_of,
)


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def vec(*head):
    v = np.zeros(768, dtype=np.float32)
    v[:len(head)] = head
    return unit(v)


def episode(label, direction="debit", counterparty="SAM LEE"):
    return dict(txn_id=0, date=date(2026, 1, 1), direction=direction, amount=10.0, channel="transfer",
                counterparty=counterparty, note="", label=label, frozen_label="Insurance",
                source=Source.CORRECTED, weight=1.0, learned_in=1)


def txn(channel):
    return Transaction(1, 7, date(2026, 7, 1), "debit", 20.0, "ZELLE TO SAM LEE", "", channel,
                       "Housing", Scenario.PREFERENCE)


def test_source_of_covers_all_four_outcomes():
    shown = Decision("Groceries", flagged=False, path=Path.FROZEN)
    assert source_of(Decision("X", True, Path.ASK), Review("X")) == Source.ASKED
    assert source_of(shown, Review("Gifts")) == Source.CORRECTED
    assert source_of(shown, Review("Groceries")) == Source.MODEL_ACCEPTED
    assert source_of(Decision("Housing", False, Path.SIMILAR_NOTE), Review("Housing")) == Source.MEMORY_ACCEPTED


def test_nearest_ranks_by_cosine_and_skips_superseded_and_other_direction():
    m = Memory()
    a = m.add(episode("Housing"), vec(1, 0))
    b = m.add(episode("Food"), vec(1, 1))
    m.add(episode("Income", direction="credit"), vec(1, 0))
    hits = m.nearest(vec(1, 0), k=5, direction="debit")
    assert [ep.episode_id for ep, _ in hits] == [a.episode_id, b.episode_id]
    m.supersede([a.episode_id], month=3)
    assert [ep.episode_id for ep, _ in m.nearest(vec(1, 0), k=5, direction="debit")] == [b.episode_id]
    assert a.superseded_in == 3 and len(m) == 3   # superseded, not deleted


def test_remove_keeps_embeddings_aligned():
    m = Memory()
    a = m.add(episode("A"), vec(1, 0))
    b = m.add(episode("B"), vec(0, 1))
    m.remove([a.episode_id])
    assert len(m) == 1 and m.embeddings.shape == (1, 768)
    assert m.nearest(vec(0, 1), k=1)[0][0] is b


def test_frozen_only_flags_transfers_and_trusts_card_purchases():
    out = FrozenOutput("Insurance", 0.95, vec(1))
    assert decide_frozen_only(txn("transfer"), out, Memory(), Config()).flagged
    card = decide(txn("card"), out, Memory(), Config(read="frozen_only"))
    assert not card.flagged and card.path == Path.FROZEN and card.label == "Insurance"
