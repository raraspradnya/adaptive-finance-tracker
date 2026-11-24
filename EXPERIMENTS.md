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
| **Tune**: `data/transactions_2026.csv` | 630 rows, Jan–Dec 2026, one Berkeley student (`src/data_generator.py`) | Choosing settings, debugging, the story charts (coffee, Trader Joe's, Ashley) |
| **Val**: `data/transactions_val.csv` | **Not generated yet.** Same generator (`--val`, seed 2028); the same kinds of shift, each in a **randomly drawn month** that nobody looks at | Checking that the whole final system (all choices from Experiments 1–3) holds on a year it was never fitted to. Replayed **once** |
| **Test** | A dataset made by a friend who has not seen our data, code or this plan. Ideally her own anonymized bank history, labeled by her | Final numbers, replayed **once** at the end |

"Tune" is not training data: nothing is trained. It is where settings are chosen and bugs are found.
Each set is its own replay, starting with an empty memory.

**Order:** every choice in Experiments 1–3 is made on tune only. Then val is generated and run once
for all three experiments, then test once. An earlier val year (seed 2027) was looked at during
Experiment 1, so it was deleted.

```
tune ──► Exp 1, 2, 3 (choose everything) ──► val, once ──► test, once
```

Shifts built into tune:

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
| **Adaptation errors** (plasticity) | Rows whose label changed from the text's first label (coffee after July), counted once the new label has appeared in an earlier month, that still get labeled wrong |
| **Spread errors** (stability) | Rows back on the text's usual label that get a label the text had only as an exception earlier (Trader Joe's labeled Gifts after the gift card) |

Both are computed from the labels alone, so they work on any dataset without an event list.
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

| Condition | Vote | Settings tried on tune |
|---|---|---|
| Frozen only | No memory | none |
| Similarity | similarity | none (same as Exp 1 "everything") |
| + Trust | similarity × trust | Trust for left-alone rows: 0.1 / 0.25 / 0.5 / 0.75 |
| + Freshness | similarity × freshness | Half-life: 1 / 3 / 6 months |
| + Both | similarity × trust × freshness | 4 × 3 combinations |
| Exp 1 winner (reference) | corrections + answered stored, similarity vote | none |
| Exp 1 winner + Freshness | corrections + answered stored, similarity × freshness | Half-life: 1 / 3 / 6 months (trust does nothing here: no left-alone rows are stored) |

Result on tune: "+ Both" (trust 0.25, half-life 1 month) is best: 44 manual checks, no adaptation
or spread errors. "Exp 1 winner + Freshness" is identical to the Exp 1 winner at every half-life
(48 checks, 4 spread errors): its memory almost never holds conflicting labels for the same text,
so there is nothing for freshness to reweigh, and nothing to outvote the gift card. Takeaway:
left-alone rows are useful evidence; they only hurt when counted at full weight.

Trust 0.1 was added to the planned 0.25 / 0.5 / 0.75: the arithmetic below shows trust alone has
to be about that low to flip the coffee case.

The best setting per family is chosen on tune (fewest manual checks, then highest accuracy) and
saved to `results/exp2/chosen.json`. The val run only reads that file, so it cannot re-tune.

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

**Setup:** Experiment 2's chosen reader ("+ Both": store everything, left-alone rows trust 0.25,
votes halve every month), read from `results/exp2/chosen.json`. It is also the combined best of
Experiments 1 and 2 (Experiment 1's storage + freshness was tried and added nothing).

| Factor | Values |
|---|---|
| Memory size *K* | 50 / 100 / 200 rows, and unlimited (tune stores up to 630 rows a year) |
| Eviction rule | **Oldest first** · **Random** (reservoir sampling: every row seen so far has the same chance to stay; 5 seeds, averaged) · **Least useful** (lowest trust × freshness right now) · **Least useful + merge** (a new row first merges into a stored row with the same label and similarity ≥ 0.99; the merged row's weight is the sum of its members' trust, each aged, so votes are unchanged) |

**Rare items** are measured generically: **long-gap accuracy** = accuracy on rows whose text was
seen before but not in the previous 3 months.

**Choice for the combined best** (on tune, saved to `results/exp3/chosen.json`): the rule with the
fewest manual checks summed over the three sizes, at the smallest *K* whose manual checks stay
within 10% of unlimited memory.

**Results on tune** (manual checks per year; accuracy):

| Rule | K=50 | K=100 | K=200 | Unlimited |
|---|---|---|---|---|
| Oldest first | 112 · 82.9% | 68 · 89.8% | 46 · 93.0% | 44 · 93.3% |
| Random (mean of 5) | 145 · 78.0% | 83 · 87.4% | 53 · 91.8% | 44 · 93.3% |
| Least useful | 94 · 85.6% | 52 · 92.2% | 46 · 93.0% | 44 · 93.3% |
| **Least useful + merge** | **45 · 93.2%** | 44 · 93.3% | 44 · 93.3% | 44 · 93.3% |

- **Merging is the big win:** the year's 630 rows collapse into 67 merged rows with no loss at all,
  so even K=50 is almost free. Chosen: least useful + merge, K=50.
- **Rare items:** with K=50, oldest first remembers only 24% of long-gap rows; random 61%; least
  useful 55%; least useful + merge 97%. Random beats oldest first on rare items (the classic reason
  for reservoir sampling) but is the worst overall.
- **Old exceptions:** at K=50, oldest first and least useful evict the old Trader Joe's Groceries
  rows, so the gift card spreads again in June (0/4). Random keeps some old rows by luck, so the gift
  card is contained, but it also keeps old coffee/Ashley rows and adapts slowly.
- **Caveat:** merging works this well because the generated data repeats the same shop strings.
  Real statements vary more, so merging will compress less; the test set will show how much.

---

## At the end: val and test

Once all three experiments are settled on tune (every setting locked):

| Run | What | Shows |
|---|---|---|
| **Val** (generated then, seed 2028) | Every experiment's conditions (`--val` on each script) **and** the combined best setup (best storage + best reading + best forgetting) vs frozen only | Each ranking holds on an unseen year, and the full system works |
| **Test** (friend's data) | The combined best setup vs frozen only, once | The headline number |

The combined best and the test run are in `experiments/final.py`:
`--val` and `--test FILE.csv`.

## Fixed decisions

- The base model is frozen. No fine-tuning, no fine-tuning baseline, no Avalanche.
- No merchant-name rules: memory works only from the transaction's own text and note.
- The recurring / counterparty cascade and the learned gate are dropped.
- The budget recommender is out of scope; this project is the categorizer only.

## Files

```
src/memory.py                  normalize(), memory store, reader (vote), storage policy, merging + eviction
src/simulation.py              simulated user, run_year(config), embedding cache
src/metrics.py                 log → metrics tables, line charts, heatmaps
experiments/common.py          shared: tune/val files, running a list of conditions, reports
experiments/pick_tau.py        chooses the similarity cutoff on tune
experiments/exp1_storage.py    Experiment 1 → results/exp1/<split>/            (done on tune)
experiments/exp2_read.py       Experiment 2 → results/exp2/<split>/, chosen.json (done on tune)
experiments/exp3_forget.py     Experiment 3 → results/exp3/<split>/, chosen.json (done on tune)
experiments/final.py           combined best vs frozen only → results/final/<split>/ (tune done; val, test at the end)
```
