"""The memory system for the categorizer: shared data types, the memory store, and the functions
each teammate fills in. Both halves of the project code against this file.

    Person A (data, write side, metrics)   Person B (read side, demo)
    - generate the year of Transactions    - extract_counterparty
    - simulate_review                      - decide (the four-check cascade, then the learned gate)
    - store, evict                         - the UI demo
    - run_year and the metrics

Data flow for one month:

    Transaction --FrozenModel.encode--> FrozenOutput
    (Transaction, FrozenOutput, Memory) --decide--> Decision        [B]
    (Transaction, Decision) --simulate_review--> Review              [A]
    (Memory, Transaction, FrozenOutput, Decision, Review) --store--> Memory gets an Episode   [A]
    after the month: evict(Memory, K, policy)                       [A]

Rules for both sides:
- decide() must never read Transaction.true_label or Transaction.scenario. They are ground truth
  for the simulated user and the metrics only.
- The frozen model is never retrained. All learning is adding, superseding and removing Episodes.
- Change this file together, not alone: it is the contract.

Status: data types, Memory, source_of and decide_frozen_only are implemented. The functions
marked TODO raise NotImplementedError until their owner fills them in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

EMBED_DIM = 768

Direction = Literal["debit", "credit"]          # debit = money out, credit = money in
Channel = Literal["card", "transfer", "bank"]   # transfer = Zelle/Venmo; bank = autopay, payroll, fees


class Scenario(str, Enum):
    """Why a row's true label is what it is. Ground truth, used only for metrics."""
    GENERAL = "general"          # true label = what most people would call it
    PREFERENCE = "preference"    # this user always files it differently
    CHANGED = "changed"          # the user changed their mind partway through the year
    DRIFT = "drift"              # the pattern itself changed (new rent amount, new roommate, ...)


class Source(str, Enum):
    """How an Episode's label was obtained, most trusted first."""
    CORRECTED = "corrected"              # shown unflagged, user changed it
    ASKED = "asked"                      # shown as a flagged guess, user confirmed or changed it
    MEMORY_ACCEPTED = "memory_accepted"  # memory labeled it unflagged, user kept it
    MODEL_ACCEPTED = "model_accepted"    # frozen model labeled it unflagged, user kept it


class Path(str, Enum):
    """Which part of the read policy decided a row."""
    RECURRING = "recurring"
    SIMILAR_NOTE = "similar_note"
    COUNTERPARTY = "counterparty"
    PAYBACK = "payback"
    GATE = "gate"        # the learned gate (Experiment 2)
    ASK = "ask"          # nothing fired on a transfer: flagged guess
    FROZEN = "frozen"    # nothing fired on a card purchase: frozen label, unflagged


# Starting trust weights. The 0.5 is tuned on January-June (0.25 / 0.5 / 0.75), then frozen.
SOURCE_WEIGHT = {
    Source.CORRECTED: 1.0,
    Source.ASKED: 1.0,
    Source.MEMORY_ACCEPTED: 0.5,
    Source.MODEL_ACCEPTED: 0.5,
}


# ---------------------------------------------------------------- data types

@dataclass(frozen=True)
class Transaction:
    """One row of a bank statement. Written by the data generator."""
    txn_id: int
    month: int                 # 1-12
    date: date
    direction: Direction
    amount: float              # always positive; direction gives the sign
    raw_description: str       # bank text, e.g. "ZELLE TO SAM LEE" or "TRADER JOE'S #552 BERKELEY CA"
    note: str                  # Venmo note, or "" (Zelle exports usually drop it)
    channel: Channel           # readable from the bank text, so the system may use it
    # ground truth: never read by decide()
    true_label: str
    scenario: Scenario


@dataclass(frozen=True)
class FrozenOutput:
    """What the frozen model returns for one transaction, from one pass."""
    label: str
    confidence: float
    embedding: np.ndarray      # shape (768,), L2-normalized


@dataclass(frozen=True)
class Decision:
    """What the system shows the user for one transaction."""
    label: str
    flagged: bool              # True = shown as "unsure, please check"
    path: Path
    evidence: str = ""         # short reason for the UI, e.g. "3 similar notes said Housing"


@dataclass(frozen=True)
class Review:
    """What the user did with a Decision. Written by the simulated user (or the UI)."""
    final_label: str
    note: str = ""             # anything the user typed while reviewing


@dataclass
class Episode:
    """One reviewed transaction stored in memory."""
    episode_id: int
    txn_id: int
    date: date
    direction: Direction
    amount: float
    channel: Channel
    counterparty: str
    note: str
    label: str
    frozen_label: str          # what the frozen model said; model-accepted episodes vote to keep it
    source: Source
    weight: float
    learned_in: int            # month stored
    superseded_in: int | None = None   # month a newer label replaced it; kept, never overwritten
    is_exception: bool = False         # a one-off that only matches near-identical rows


@dataclass
class Config:
    """One experiment condition. Defaults are the reference config."""
    storage: Literal["corrections_only", "all_equal", "weighted"] = "weighted"
    read: Literal["cascade", "gate", "frozen_only"] = "cascade"
    capacity: int | None = None        # K; None = unlimited
    eviction: Literal["fifo", "random", "priority"] = "priority"
    knn_k: int = 5
    tau: float = 0.8                   # similarity threshold, tuned on January-June
    accepted_weight: float = 0.5       # tuned on January-June
    rubber_stamp_rate: float = 0.15    # simulated user confirms without reading
    seed: int = 290


# ---------------------------------------------------------------- memory

class Memory:
    """Episodes plus an aligned embedding matrix. Cosine kNN is one matrix product."""

    def __init__(self):
        self.episodes: list[Episode] = []
        self.embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
        self._next_id = 1

    def __len__(self):
        return len(self.episodes)

    def add(self, episode_fields: dict, embedding: np.ndarray) -> Episode:
        ep = Episode(episode_id=self._next_id, **episode_fields)
        self._next_id += 1
        self.episodes.append(ep)
        self.embeddings = np.vstack([self.embeddings, embedding[None, :].astype(np.float32)])
        return ep

    def supersede(self, episode_ids, month: int):
        """Stamp episodes as replaced. They stay in memory for history but stop voting."""
        ids = set(episode_ids)
        for ep in self.episodes:
            if ep.episode_id in ids and ep.superseded_in is None:
                ep.superseded_in = month

    def remove(self, episode_ids):
        """Delete episodes outright. Only eviction (Experiment 3) does this."""
        ids = set(episode_ids)
        keep = [i for i, ep in enumerate(self.episodes) if ep.episode_id not in ids]
        self.episodes = [self.episodes[i] for i in keep]
        self.embeddings = self.embeddings[keep]

    def active_mask(self) -> np.ndarray:
        return np.array([ep.superseded_in is None for ep in self.episodes], dtype=bool)

    def nearest(self, query: np.ndarray, k: int, direction: Direction | None = None):
        """Top-k active episodes by cosine similarity: list of (Episode, similarity), best first."""
        if not self.episodes:
            return []
        scores = self.embeddings @ query
        mask = self.active_mask()
        if direction is not None:
            mask &= np.array([ep.direction == direction for ep in self.episodes])
        scores = np.where(mask, scores, -np.inf)
        top = np.argsort(scores)[::-1][:k]
        return [(self.episodes[i], float(scores[i])) for i in top if np.isfinite(scores[i])]

    def by_counterparty(self, counterparty: str, active_only: bool = True) -> list[Episode]:
        return [ep for ep in self.episodes
                if ep.counterparty == counterparty and (ep.superseded_in is None or not active_only)]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(ep) for ep in self.episodes])


# ---------------------------------------------------------------- shared helpers (implemented)

def source_of(decision: Decision, review: Review) -> Source:
    """How a reviewed row's label was obtained. Deterministic, so both sides agree."""
    if decision.flagged:
        return Source.ASKED
    if review.final_label != decision.label:
        return Source.CORRECTED
    if decision.path == Path.FROZEN:
        return Source.MODEL_ACCEPTED
    return Source.MEMORY_ACCEPTED


def decide_frozen_only(txn: Transaction, frozen: FrozenOutput, memory: Memory, config: Config) -> Decision:
    """Baseline read policy: ignore memory. Also the stand-in until decide() is ready."""
    if txn.channel == "transfer":
        return Decision(frozen.label, flagged=True, path=Path.ASK, evidence="frozen model only")
    return Decision(frozen.label, flagged=False, path=Path.FROZEN)


# ---------------------------------------------------------------- read side (Person B)

def extract_counterparty(raw_description: str) -> str:
    """TODO(B): who is on the other side, using regex rules only.

    Drop prefixes such as "ZELLE TO", "VENMO", "SQ *", "TST*", store numbers such as "#552",
    and trailing city/state. "ZELLE TO SAM LEE" -> "SAM LEE"; "TRADER JOE'S #552 BERKELEY CA"
    -> "TRADER JOE'S".
    """
    raise NotImplementedError


def decide(txn: Transaction, frozen: FrozenOutput, memory: Memory, config: Config) -> Decision:
    """TODO(B): the read policy.

    config.read == "cascade": recurring -> similar note -> consistent counterparty -> payback;
    the first check with enough evidence decides. If none fires: transfer -> flagged guess
    (Path.ASK), card or bank -> frozen label unflagged (Path.FROZEN).
    config.read == "gate": the learned logistic-regression gate (Experiment 2).
    config.read == "frozen_only": return decide_frozen_only(...).
    """
    if config.read == "frozen_only":
        return decide_frozen_only(txn, frozen, memory, config)
    raise NotImplementedError


# ---------------------------------------------------------------- write side (Person A)

def simulate_review(txn: Transaction, decision: Decision, rng: np.random.Generator, config: Config) -> Review:
    """TODO(A): the simulated user.

    Flagged rows are always answered correctly. Unflagged rows are corrected to txn.true_label,
    except that with probability config.rubber_stamp_rate the user confirms without reading.
    """
    raise NotImplementedError


def store(memory: Memory, txn: Transaction, frozen: FrozenOutput, decision: Decision,
          review: Review, config: Config) -> Episode | None:
    """TODO(A): write one reviewed row to memory, per config.storage (Experiment 1).

    Returns the new Episode, or None if the policy skips it (e.g. corrections_only on an
    accepted row). Handles one-off exceptions and superseding older episodes it contradicts.
    """
    raise NotImplementedError


def evict(memory: Memory, config: Config, month: int) -> list[int]:
    """TODO(A): if config.capacity is set and memory is over it, remove episodes per
    config.eviction (Experiment 3). Returns the removed episode_ids."""
    raise NotImplementedError


# ---------------------------------------------------------------- run log (Person A, read by both)

# One row per transaction per run. Every metric is computed from this table.
LOG_COLUMNS = [
    "run_id", "seed", "txn_id", "month", "channel", "scenario", "true_label",
    "frozen_label", "frozen_confidence",
    "label", "flagged", "path",                # the Decision
    "final_label", "source",                   # the Review and how it was stored
    "stored", "memory_size",
]


def run_year(transactions: list[Transaction], frozen: dict[int, FrozenOutput], config: Config) -> pd.DataFrame:
    """TODO(A): month by month, decide -> review -> store, then evict. Returns a LOG_COLUMNS frame."""
    raise NotImplementedError
