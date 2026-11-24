# Experiment plan

The categorizer never changes its weights. A frozen DistilBERT labels each transaction, and an
external **memory** of reviewed transactions corrects it over time. The three experiments each
change one part of that memory and keep everything else fixed:

| # | Question | What changes | Held fixed |
|---|---|---|---|
| 1 | **What to store**: which reviewed rows go into memory? | Storage policy | Reader: similarity-only kNN · no size limit |
| 2 | **How to read**: how much does each stored row's vote count? | Vote weighting (trust, freshness) | Store everything · no size limit |
| 3 | **What to forget**: when memory is full, what goes? | Memory size and eviction rule | Winners of 1 and 2 |

Experiment 1 comes first and is the only one being built right now.

---

## Shared setup (used by all three)

### Data

| Set | What it is | Used for |
|---|---|---|
| **Dev**: `data/transactions_2026.csv` | 630 rows, Jan–Dec 2026, one Berkeley student (from `src/data_generator.py`) | Choosing settings, debugging, every chart while building |
| **Test** | A dataset made by a friend who has not seen the dev data, the code or this plan. Ideally her own anonymized bank history, labeled by her | Final numbers, replayed **once** at the end |

Dev is not "training data": nothing is trained. It is where settings are chosen and bugs are found.
Test is kept blind so the reported numbers are not fitted to events we designed.

Shifts built into dev:

| Kind | Event | Month |
|---|---|---|
| Concept | Breweries and ramen are "Social" (a category the model doesn't have) | All year |
| Concept, one-off | One $50 Trader Joe's purchase is a gift card → Gifts | May |
| Concept, permanent | Coffee shops change from Restaurants to Coffee | Jul |
| Covariate | Rent $1,225 → $1,300 · new brewery (Ale Industries) | Sep |
| Covariate | Bank switch Chase → BofA: every description changes format | Oct |
| Concept, permanent | Ashley moves out: Zelle to Ashley changes from $275 Housing to small Social payments | Nov |

Frozen model alone: 41.6% on dev overall, about 5% from October (bank switch).

### What the system sees

For each row: `normalize(description) + " | " + note`.

- `normalize()` strips bank wrappers: `CHECKCARD mmdd`, reference numbers, `CONF#`, and names
  written LAST FIRST. A Chase row and the same BofA row then look alike.
- The frozen model returns a label, a confidence, and a 768-number vector (the [CLS] embedding,
  scaled to length 1, so cosine similarity is a dot product).
- Vectors are computed once and cached, so every run takes seconds.

### The replay (test-then-train)

```
memory = empty
for month in Jan..Dec:
    1. predict  every row, using only memory from earlier months
    2. score    predictions against true_category        → this month's numbers
    3. review   the simulated user goes through the month
    4. store    reviewed rows go into memory (per the storage policy)
```

Each month is scored **before** it is learned from. During a replay the memory keeps growing;
the settings (cutoff, rules, policy) do not change.

### The simulated user

The user never rubber-stamps (the UI requires every "Check this" row to be answered before
"Save review" works). `true_category` plays the user's answers.

| The row was… | The user… | Recorded as |
|---|---|---|
| Flagged "Check this" | Answers with the right label (✓ or ✗ + pick) | `asked` |
| Not flagged, label wrong | Changes it to the right label | `corrected` |
| Not flagged, label right | Leaves it | `accepted` |

So every stored label is correct. The experiments are about which correct rows to keep and how
much to listen to them: **stability** (one-off exceptions should not spread) vs **plasticity**
(permanent changes should be picked up quickly).

### The reader (the kNN part)

1. **Neighbours**: every stored row with cosine similarity ≥ **τ** to the new row.
2. **Vote**: each neighbour votes for its label. In Experiment 1 a vote is just its similarity.
3. **Decide**: if the top label has ≥ **60%** of the vote, use it.
   Otherwise use the frozen label, and flag "Check this" when frozen confidence is < **0.9**.

τ = **0.93**, set once on dev (`experiments/pick_tau.py`) and then fixed for all experiments.
Different shops score below 0.7, but different Venmo notes score up to 0.92 ("brewery tab" vs
"concert tix"), so the cutoff sits just above that. Cost: a few variants of the same shop
("CAFFE STRADA" vs "SQ *CAFFE STRADA", 0.90) no longer match and each learns separately.

### Metrics (per month, and summed over the year)

| Metric | What it measures |
|---|---|
| **Manual checks** (headline) | Rows the user had to act on: flagged + corrected |
| **Accuracy** | Predictions vs `true_category`, before review |
| **Adaptation errors** (plasticity) | Wrong labels on rows after a permanent change (coffee from Jul, Ashley from Nov), and months until they come out right |
| **Spread errors** (stability) | Normal rows wrongly given a one-off's label (Trader Joe's rows labeled Gifts) |
| **Flips** | Rows the frozen model had right that memory turned wrong |
| **Fix rate** | Of the rows the frozen model got wrong, the share memory got right |
| **Memory size** | Stored rows |

Every chart also shows **frozen only** (no memory) as the baseline.

---

## Experiment 1: What to store

**Question:** which reviewed rows are worth keeping?

**Reader:** similarity-only kNN as above. No trust, no freshness, no size limit.

| Condition | Stored |
|---|---|
| Frozen only | Nothing (baseline) |
| Corrections only | `corrected` rows |
| Corrections + answered | `corrected` + `asked` rows |
| Everything | Every reviewed row |

Four deterministic runs (no randomness, so no seeds).

**Predictions, written before running:**

| Case | Corrections only | Corrections + answered | Everything |
|---|---|---|---|
| Coffee in Aug (should be Coffee) | ✅ Old coffee rows were never stored (the frozen model had them right) | ✅ | ❌ ~24 old Restaurants rows outvote ~4 Coffee rows |
| Trader Joe's in Jun (should be Groceries) | ❌ The only neighbour is the Gifts correction | ❌ likely | ✅ Groceries rows outvote it |
| Ashley in Dec (should be Social) | ❌ Every earlier Ashley row was a correction (the model said Transfer), so 10 Housing votes vs 1 Social | ❌ | ❌ |

Expected story: storing less makes memory plastic but unstable; storing everything makes it
stable but slow to change; and none of the three handles Ashley, which motivates Experiment 2.

**Outputs:**
1. Manual checks per month: four lines.
2. Case table: coffee, Trader Joe's and Ashley rows month by month, per condition.
3. Summary table: every metric per condition.

---

## Experiment 2: How to read

**Question:** does weighting votes by trust and freshness get both plasticity and stability?

**Storage:** everything (trust only matters when memory holds left-alone rows). No size limit.

Each neighbour's vote is **similarity × trust × freshness**:

- **Trust**: how much a stored label is believed. Rows the user changed or answered count 1.0;
  rows the user left alone count less.
- **Freshness**: how recent a stored row is. A vote halves every *N* months.

| Condition | Vote | Settings tried on dev |
|---|---|---|
| Frozen only | No memory | none |
| Similarity | similarity | none (same as Exp 1 "everything") |
| + Trust | similarity × trust | Trust for left-alone rows: 0.25 / 0.5 / 0.75 |
| + Freshness | similarity × freshness | Half-life: 1 / 3 / 6 months |
| + Both | similarity × trust × freshness | 3 × 3 combinations |

16 dev runs. The best setting per condition is chosen on dev; test is replayed with those.

**Predictions** (assuming ~4 coffee and ~4 Trader Joe's visits a month):

| Case | Similarity | + Trust | + Freshness | + Both |
|---|---|---|---|---|
| Coffee in Aug | ❌ | ❌ | ⚠️ ~50/50, flagged | ✅ |
| Trader Joe's in Jun | ✅ | ✅ | ✅ | ✅ |
| Ashley in Dec | ❌ | ❌ | ⚠️ | ✅ likely |

Expected story: trust lets corrections beat confirmations; freshness lets new evidence beat old
evidence; together they handle both kinds of case.

**Outputs:**
1. Ablation bars: the four conditions, split into adaptation errors, spread errors, manual checks.
2. Case timelines: coffee, Ashley, Trader Joe's month by month.
3. Dev heatmap: trust × freshness, to show the chosen setting is not cherry-picked.

---

## Experiment 3: What to forget

**Question:** a real app can't keep every row forever. When memory is capped, which rows should
go so the system keeps working?

**Setup:** the storage policy and reader that won Experiments 1 and 2.

| Factor | Values |
|---|---|
| Memory size *K* | 50 / 100 / 200 rows, and unlimited (dev stores up to ~630 rows a year) |
| Eviction rule | **Oldest first** (FIFO) · **Random** (reservoir sample, 5 seeds) · **Least useful** (lowest trust × freshness goes first) + **consolidation** (near-identical rows with the same label merge into one row that carries a count) |

**What to watch:**
- Does accuracy hold as *K* shrinks, and which rule degrades most gracefully?
- **Rare items**: are AMC, Ippudo and other rarely seen rows still recognized late in the year,
  or did they get evicted?
- **Old exceptions**: is the May gift card evicted, and does that help or hurt?

**Outputs:**
1. Accuracy and manual checks vs *K*, one line per eviction rule.
2. Rare-item recall in Oct–Dec per rule.

Details (exact *K* values, the consolidation threshold) get settled after Experiment 2.

---

## Fixed decisions

- The base model is frozen. No fine-tuning, no fine-tuning baseline, no Avalanche.
- No merchant-name rules: memory works only from the transaction's own text and note.
- The recurring / counterparty cascade and the learned gate are dropped.
- The budget recommender is out of scope; this project is the categorizer only.

## Planned files

```
src/memory.py                  memory store, normalize(), reader, storage, eviction
src/simulation.py              simulated user, run_year(config)
src/metrics.py                 log → metrics tables and charts
experiments/exp1_storage.py    runs Experiment 1, writes results/exp1_*
experiments/exp2_read.py
experiments/exp3_forget.py
```
