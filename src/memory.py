"""The memory that lets a frozen categorizer keep learning (see EXPERIMENTS.md).

The frozen model never changes. Every reviewed transaction can be stored in memory as
(vector, label, how the label was obtained, date). To label a new transaction, the reader finds
stored rows that look similar and lets them vote.

    normalize()      bank text -> a form that is the same across banks
    system_text()    what the model reads: normalized description + Venmo note
    Memory           stored rows and their vectors
    decide()         the reader: neighbours vote, or fall back to the frozen model
    source_of()      how the user's final label was obtained (asked / corrected / accepted)
    should_store()   the storage policy (Experiment 1)
    store()          write one reviewed row: optional merging, size limit and eviction (Experiment 3)
    Config           one experiment condition

decide() never sees true_category. Only the simulated user and the metrics read it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np

EMBED_DIM = 768

# How a stored label was obtained.
ASKED = "asked"            # row was flagged "Check this"; the user answered it (✓ or ✗ + pick)
CORRECTED = "corrected"    # row was not flagged; the user changed it
ACCEPTED = "accepted"      # row was not flagged; the user left it as it was

Storage = Literal["none", "corrections", "corrections_asked", "everything"]
STORED_SOURCES: dict[str, set[str]] = {
    "none": set(),                                  # frozen only: memory stays empty
    "corrections": {CORRECTED},
    "corrections_asked": {CORRECTED, ASKED},
    "everything": {CORRECTED, ASKED, ACCEPTED},
}


@dataclass(frozen=True)
class Config:
    """One experiment condition. Defaults = Experiment 1's reader with everything stored."""
    name: str = "everything"
    storage: Storage = "everything"
    tau: float = 0.93                 # neighbours: stored rows with cosine similarity >= tau
                                      # (picked on dev by experiments/pick_tau.py: the closest
                                      # different-meaning pair, two Venmo notes, is 0.92)
    share: float = 0.6                # memory answers if the top label has >= this share of the vote
    flag_below: float = 0.9           # no memory answer: flag "Check this" if frozen confidence < this
    # Experiment 2 (vote = similarity x trust x freshness). Defaults switch both off.
    trust_accepted: float = 1.0       # trust of rows the user left alone; changed/answered rows = 1.0
    half_life_months: float | None = None   # a vote halves every this many months; None = never fades
    # Experiment 3 (what to forget). Defaults: no limit, no merging.
    capacity: int | None = None       # most rows memory may hold; None = unlimited
    eviction: Literal["oldest", "random", "least_useful"] = "oldest"   # who goes when full
    merge_above: float | None = None  # merge a new row into a stored one with the same label and
                                      # similarity >= this; None = never merge
    seed: int = 0                     # for random eviction


def freshness(age_days, config: Config):
    """How much a vote of this age still counts (1.0 when freshness is off)."""
    if config.half_life_months is None:
        return np.ones_like(np.asarray(age_days, dtype=np.float64))
    return 0.5 ** (np.asarray(age_days, dtype=np.float64) / 30.44 / config.half_life_months)


def trust(source: str, config: Config) -> float:
    return config.trust_accepted if source == ACCEPTED else 1.0


# ---------------------------------------------------------------- text

# Bank wrappers that carry no meaning: card prefixes, reference and confirmation numbers, ACH codes.
_NOISE = [
    r"\bCHECKCARD \d{4}\b",     # BofA card prefix with the date (mmdd)
    r"\bCONF#\s*\S+",           # BofA Zelle confirmation
    r"\bJPM99\w+",              # Chase Zelle reference
    r"\bCO ID:\s*\S+",          # ACH company id
    r"\bID:\s*\S+",             # ACH / Venmo ids
    r"\bDES:",                  # BofA "description:" marker
    r"\b(PPD|WEB)\b",           # ACH entry codes
    r"\b\d{6,}\b",              # long reference numbers
]


def normalize(description: str) -> str:
    """Strip bank-specific wrappers so the same transaction reads the same at any bank.

    "CHECKCARD 1004 PEETS COFFEE #0731 BERKELEY CA 12345678901" -> "PEETS COFFEE #0731 BERKELEY CA"
    Zelle names are sorted, because banks disagree on FIRST LAST vs LAST FIRST:
    "Zelle payment to Ashley Kim JPM99..." and "ZELLE TO KIM ASHLEY CONF# ..." -> "ZELLE TO ASHLEY KIM"
    """
    text = description.upper().replace("ZELLE PAYMENT TO", "ZELLE TO")
    for pattern in _NOISE:
        text = re.sub(pattern, " ", text)
    text = " ".join(text.split())
    zelle = re.match(r"ZELLE TO (.+)", text)
    if zelle:
        text = "ZELLE TO " + " ".join(sorted(zelle.group(1).split()))
    return text


def system_text(description: str, note: str = "") -> str:
    """The one string the frozen model reads for a row."""
    text = normalize(description)
    return f"{text} | {note}" if note else text


# ---------------------------------------------------------------- memory

class Memory:
    """Stored rows plus an aligned matrix of their vectors (each of length 1).

    Each row has a weight: its trust when stored, or, for a merged row, the summed trust of every
    row merged into it (each aged to the merged row's date). A row's vote is
    similarity x weight x freshness(age).
    """

    def __init__(self):
        self.txn_ids: list[int] = []
        self.labels: list[str] = []
        self.sources: list[str] = []
        self.days: list[int] = []          # date as a day number, for freshness
        self.weights: list[float] = []
        self.counts: list[int] = []        # how many reviewed rows this row stands for (merging)
        self.embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
        self.seen = 0                      # rows offered for storage so far (random eviction)

    def __len__(self):
        return len(self.labels)

    def add(self, txn_id: int, when: date, label: str, source: str, embedding: np.ndarray,
            weight: float = 1.0):
        self.txn_ids.append(txn_id)
        self.labels.append(label)
        self.sources.append(source)
        self.days.append(when.toordinal())
        self.weights.append(weight)
        self.counts.append(1)
        self.embeddings = np.vstack([self.embeddings, embedding[None, :].astype(np.float32)])

    def remove(self, i: int):
        for column in (self.txn_ids, self.labels, self.sources, self.days, self.weights, self.counts):
            del column[i]
        self.embeddings = np.delete(self.embeddings, i, axis=0)

    def usefulness(self, when: date, config: Config) -> np.ndarray:
        """weight x freshness of every row, as of `when`."""
        age = when.toordinal() - np.array(self.days)
        return np.array(self.weights) * freshness(age, config)

    def neighbours(self, query: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
        """Indices and similarities of stored rows with cosine similarity >= tau."""
        if not self.labels:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        sims = self.embeddings @ query
        idx = np.flatnonzero(sims >= tau)
        return idx, sims[idx]


# ---------------------------------------------------------------- reading

@dataclass(frozen=True)
class Decision:
    """What the system shows the user for one row."""
    label: str
    flagged: bool              # shown as "Check this"
    path: str                  # "memory" or "frozen"
    share: float = 0.0         # top label's share of the vote (0 if no neighbours)
    neighbours: int = 0


def vote_weights(memory: Memory, idx: np.ndarray, sims: np.ndarray, when: date, config: Config) -> np.ndarray:
    """similarity x weight (trust) x freshness for each neighbour."""
    age = when.toordinal() - np.array([memory.days[i] for i in idx])
    return sims.astype(np.float64) * np.array([memory.weights[i] for i in idx]) * freshness(age, config)


def decide(embedding: np.ndarray, when: date, frozen_label: str, frozen_conf: float,
           memory: Memory, config: Config) -> Decision:
    """The reader. Neighbours vote; if one label wins clearly, use it. Otherwise fall back to
    the frozen model's label, flagged when the model is unsure."""
    idx, sims = memory.neighbours(embedding, config.tau)
    share = 0.0
    if len(idx):
        weights = vote_weights(memory, idx, sims, when, config)
        totals: dict[str, float] = {}
        for i, w in zip(idx, weights):
            totals[memory.labels[i]] = totals.get(memory.labels[i], 0.0) + w
        label, top = max(totals.items(), key=lambda kv: kv[1])
        total = sum(totals.values())
        share = top / total if total > 0 else 0.0     # all votes weigh 0: no evidence
        if share >= config.share:
            return Decision(label, flagged=False, path="memory", share=share, neighbours=len(idx))
    return Decision(frozen_label, flagged=frozen_conf < config.flag_below, path="frozen",
                    share=share, neighbours=len(idx))


# ---------------------------------------------------------------- writing

def source_of(decision: Decision, final_label: str) -> str:
    if decision.flagged:
        return ASKED
    return CORRECTED if final_label != decision.label else ACCEPTED


def should_store(source: str, config: Config) -> bool:
    return source in STORED_SOURCES[config.storage]


def store(memory: Memory, txn_id: int, when: date, label: str, source: str, embedding: np.ndarray,
          config: Config, rng: np.random.Generator) -> str:
    """Write one reviewed row (already accepted by should_store). Returns what happened:
    "added", "merged", "replaced" (something was evicted to make room) or "dropped"."""
    w = trust(source, config)

    # merging: fold the row into a near-identical stored row with the same label
    if config.merge_above is not None and len(memory):
        sims = memory.embeddings @ embedding
        same = np.array([lab == label for lab in memory.labels])
        sims = np.where(same, sims, -np.inf)
        j = int(np.argmax(sims))
        if sims[j] >= config.merge_above:
            age = when.toordinal() - memory.days[j]
            memory.weights[j] = memory.weights[j] * float(freshness(age, config)) + w
            memory.days[j] = when.toordinal()
            memory.counts[j] += 1
            memory.sources[j] = source
            return "merged"

    memory.seen += 1
    if config.capacity is None or len(memory) < config.capacity:
        memory.add(txn_id, when, label, source, embedding, w)
        return "added"

    if config.eviction == "random":
        # reservoir sampling: every row offered so far has the same chance of being in memory
        j = int(rng.integers(0, memory.seen))
        if j >= config.capacity:
            return "dropped"
        victim = j
    elif config.eviction == "oldest":
        victim = int(np.argmin(memory.days))
    else:   # least useful: lowest weight x freshness right now
        victim = int(np.argmin(memory.usefulness(when, config)))
    memory.remove(victim)
    memory.add(txn_id, when, label, source, embedding, w)
    return "replaced"
