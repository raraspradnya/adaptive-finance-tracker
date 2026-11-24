"""Checks on src/memory.py and src/simulation.py. Run: .venv/bin/python -m pytest -q"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory import (  # noqa: E402
    ACCEPTED, ASKED, CORRECTED, Config, Decision, Memory, decide, normalize, should_store, source_of,
    store,
)
from simulation import run_year  # noqa: E402


def vec(*head):
    v = np.zeros(768, dtype=np.float32)
    v[:len(head)] = head
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------- normalize

def test_normalize_makes_chase_and_bofa_match():
    pairs = [
        ("Zelle payment to Ashley Kim JPM99ABCDEFG", "ZELLE TO KIM ASHLEY CONF# ab12cd34e"),
        ("VENMO PAYMENT 1234567890 WEB ID: 3264681992", "VENMO DES:PAYMENT ID:9876543210 CO ID:3264681992 WEB"),
        ("UC BERKELEY PAYROLL PPD ID: 1946000123", "UC BERKELEY DES:PAYROLL ID:123456789 CO ID:1946000123 PPD"),
        ("PEETS COFFEE #0731 BERKELEY CA", "CHECKCARD 1004 PEETS COFFEE #0731 BERKELEY CA 12345678901"),
    ]
    for chase, bofa in pairs:
        assert normalize(chase) == normalize(bofa)


def test_normalize_keeps_meaningful_numbers():
    assert normalize("MINT MOBILE 800-683-7392") == "MINT MOBILE 800-683-7392"
    assert normalize("LYFT *RIDE 07-03") == "LYFT *RIDE 07-03"


# ---------------------------------------------------------------- reader

def test_empty_memory_falls_back_to_frozen_and_flags_when_unsure():
    m, c = Memory(), Config()
    sure = decide(vec(1), date(2026, 1, 5), "Restaurants", 0.95, m, c)
    unsure = decide(vec(1), date(2026, 1, 5), "Insurance", 0.50, m, c)
    assert (sure.label, sure.flagged, sure.path) == ("Restaurants", False, "frozen")
    assert (unsure.label, unsure.flagged, unsure.path) == ("Insurance", True, "frozen")


def test_similar_stored_row_overrides_frozen():
    m = Memory()
    m.add(1, date(2026, 3, 1), "Social", CORRECTED, vec(1, 0.01))
    d = decide(vec(1), date(2026, 4, 1), "Restaurants", 0.99, m, Config())
    assert (d.label, d.flagged, d.path) == ("Social", False, "memory")


def test_dissimilar_rows_do_not_vote():
    m = Memory()
    m.add(1, date(2026, 3, 1), "Social", CORRECTED, vec(0, 1))
    d = decide(vec(1, 0), date(2026, 4, 1), "Restaurants", 0.99, m, Config())
    assert d.path == "frozen" and d.neighbours == 0


def test_split_vote_falls_back_to_frozen():
    m = Memory()
    m.add(1, date(2026, 3, 1), "Social", CORRECTED, vec(1))
    m.add(2, date(2026, 3, 2), "Restaurants", CORRECTED, vec(1))
    d = decide(vec(1), date(2026, 4, 1), "Restaurants", 0.95, m, Config())
    assert d.path == "frozen" and abs(d.share - 0.5) < 1e-6


def _coffee_memory(config):
    m, rng = Memory(), np.random.default_rng(0)
    for k in range(3):   # three old, left-alone Restaurants rows
        store(m, k, date(2026, 1, 10), "Restaurants", ACCEPTED, vec(1), config, rng)
    store(m, 9, date(2026, 7, 10), "Coffee", CORRECTED, vec(1), config, rng)   # one recent correction
    return m


def test_trust_and_freshness_shift_the_vote():
    when = date(2026, 8, 1)
    plain_cfg, both_cfg = Config(), Config(trust_accepted=0.5, half_life_months=1)
    plain = decide(vec(1), when, "Restaurants", 0.99, _coffee_memory(plain_cfg), plain_cfg)
    both = decide(vec(1), when, "Restaurants", 0.99, _coffee_memory(both_cfg), both_cfg)
    assert plain.label == "Restaurants"
    assert both.label == "Coffee" and both.path == "memory"


# ---------------------------------------------------------------- forgetting (Experiment 3)

def _fill(config, n=5):
    """n rows on different days; row k has its own direction so nothing merges by accident."""
    m, rng = Memory(), np.random.default_rng(0)
    for k in range(n):
        v = np.zeros(768, dtype=np.float32); v[k] = 1
        src = CORRECTED if k == 0 else ACCEPTED
        store(m, k, date(2026, 1 + k, 1), f"L{k}", src, v, config, rng)
    return m


def test_capacity_is_never_exceeded():
    for rule in ("oldest", "random", "least_useful"):
        m = _fill(Config(capacity=3, eviction=rule, half_life_months=1, trust_accepted=0.25))
        assert len(m) <= 3


def test_oldest_first_drops_the_oldest():
    m = _fill(Config(capacity=3, eviction="oldest"))
    assert m.txn_ids == [2, 3, 4]


def test_least_useful_keeps_a_trusted_correction_over_a_newer_left_alone_row():
    # row 0 is a correction (trust 1); rows 1-4 are left alone (trust 0.25), half-life 6 months
    m = _fill(Config(capacity=3, eviction="least_useful", trust_accepted=0.25, half_life_months=6))
    assert 0 in m.txn_ids and len(m) == 3


def test_merging_folds_near_identical_rows_with_the_same_label():
    cfg = Config(merge_above=0.99)
    m, rng = Memory(), np.random.default_rng(0)
    assert store(m, 1, date(2026, 1, 1), "Coffee", CORRECTED, vec(1), cfg, rng) == "added"
    assert store(m, 2, date(2026, 2, 1), "Coffee", ACCEPTED, vec(1, 0.001), cfg, rng) == "merged"
    assert store(m, 3, date(2026, 3, 1), "Restaurants", ACCEPTED, vec(1), cfg, rng) == "added"
    assert len(m) == 2 and m.counts == [2, 1] and m.weights[0] == 2.0


# ---------------------------------------------------------------- writing

def test_source_of():
    assert source_of(Decision("X", True, "frozen"), "X") == ASKED
    assert source_of(Decision("X", True, "frozen"), "Y") == ASKED
    assert source_of(Decision("X", False, "frozen"), "Y") == CORRECTED
    assert source_of(Decision("X", False, "memory"), "X") == ACCEPTED


def test_storage_policies():
    assert not should_store(CORRECTED, Config(storage="none"))
    assert should_store(CORRECTED, Config(storage="corrections"))
    assert not should_store(ASKED, Config(storage="corrections"))
    assert should_store(ASKED, Config(storage="corrections_asked"))
    assert not should_store(ACCEPTED, Config(storage="corrections_asked"))
    assert should_store(ACCEPTED, Config(storage="everything"))


# ---------------------------------------------------------------- replay

def _toy_year():
    """The same shop in March and April. The frozen model calls it Restaurants; she says Social."""
    df = pd.DataFrame({
        "txn_id": [1, 2], "date": [date(2026, 3, 5), date(2026, 4, 5)], "month": [3, 4],
        "description": ["RARE BARREL", "RARE BARREL"], "note": ["", ""], "text": ["RARE BARREL"] * 2,
        "true_category": ["Social", "Social"],
    })
    emb = np.stack([vec(1, 0.001), vec(1, 0.002)])
    return df, ["Restaurants", "Restaurants"], np.array([0.99, 0.99]), emb


def test_a_march_correction_fixes_the_same_row_in_april():
    df, labels, confs, emb = _toy_year()
    log = run_year(df, labels, confs, emb, Config(storage="corrections"))
    march, april = log.iloc[0], log.iloc[1]
    assert (march["label"], march["source"], march["stored"]) == ("Restaurants", CORRECTED, True)
    assert (april["label"], april["path"], april["source"]) == ("Social", "memory", ACCEPTED)


def test_frozen_only_never_learns():
    df, labels, confs, emb = _toy_year()
    log = run_year(df, labels, confs, emb, Config(storage="none"))
    assert list(log["label"]) == ["Restaurants", "Restaurants"]
    assert list(log["memory_size"]) == [0, 0]


def test_adaptation_and_spread_come_from_the_labels():
    from metrics import _add_flags
    # COFFEE: Restaurants, then changes to Coffee in month 3 (first Coffee row can't be known yet).
    # TJ: Groceries, a one-off Gifts in month 2, then Groceries again.
    log = pd.DataFrame({
        "month":      [1, 1, 2, 2, 3, 3, 4, 4],
        "text":       ["COFFEE", "TJ", "COFFEE", "TJ", "COFFEE", "TJ", "COFFEE", "TJ"],
        "true_label": ["Restaurants", "Groceries", "Restaurants", "Gifts", "Coffee", "Groceries", "Coffee", "Groceries"],
        "label":      ["Restaurants", "Groceries", "Restaurants", "Groceries", "Restaurants", "Gifts", "Restaurants", "Groceries"],
    })
    for col, val in {"frozen_label": "X", "flagged": False, "source": "accepted", "path": "memory"}.items():
        log[col] = val
    f = _add_flags(log)
    assert list(f["adapt_row"]) == [False, False, False, False, False, False, True, False]
    assert list(f["adapt_error"]) == [False, False, False, False, False, False, True, False]
    assert list(f["spread_row"]) == [False, False, False, False, False, True, False, True]
    assert list(f["spread_error"]) == [False, False, False, False, False, True, False, False]


def test_rows_in_the_same_month_do_not_see_each_other():
    df, labels, confs, emb = _toy_year()
    df["month"] = [3, 3]
    log = run_year(df, labels, confs, emb, Config(storage="everything"))
    assert list(log["path"]) == ["frozen", "frozen"]
