# Experiment plan

The categorizer never changes its weights. A frozen DistilBERT labels each transaction, and an
external **memory** of reviewed transactions corrects it over time. The three experiments each
change one part of that memory and keep everything else fixed:

| # | Question | What changes | Held fixed |
|---|---|---|---|
| 1 | **What to store**: which reviewed rows go into memory? | Storage policy | Reader: similarity-only kNN · no size limit |
| 2 | **How to read**: how much does each stored row's vote count? | Vote weighting (freshness) | Exp 1 winner's storage · no size limit |
| 3 | **What to forget**: when memory is full, what goes? | Memory size and eviction rule | Winners of 1 and 2 |

Each experiment builds on the one before: Experiment 2 starts from Experiment 1's winner, and
Experiment 3 from Experiment 2's choice.


---

## Shared setup (used by all three)

### Data

| Set | What it is | Used for |
|---|---|---|
| **Tune**: `data/transactions_2026.csv` | 630 rows, Jan–Dec 2026, one Berkeley student (`src/data_generator.py`) | Choosing settings, debugging, the story charts (coffee, Trader Joe's, Ashley) |
| **Val**: `data/transactions_val.csv` | 659 rows, generated and run once on 2026-11-24. Same generator (`--val`, seed 2028); the same kinds of shift, each in a **randomly drawn month** that nobody looks at | Checking that the whole final system (all choices from Experiments 1–3) holds on a year it was never fitted to. Replayed **once** |
| **Test** | A dataset made by a friend who has not seen our data, code or this plan. Ideally her own anonymized bank history, labeled by her | Final numbers, replayed **once** at the end |

"Tune" is not training data: nothing is trained. It is where settings are chosen and bugs are found.
Each set is its own replay, starting with an empty memory.

**Order:** every choice in Experiments 1–3 is made on tune only. Then val is generated and run once
for all three experiments, then test once. 

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

Frozen model alone: 41.6% on the raw bank text, about 5% from October (bank switch); 49.4% on the
normalized text the experiments use (the baseline in every table below).

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

**Question:** once memory holds Experiment 1's winner (corrected + answered rows), does weighting
votes by freshness help?

Each neighbour's vote is **similarity × freshness**. **Freshness**: how recent a stored row is; a
vote halves every *N* months. **Trust** (how much a stored label is believed) drops out: every
stored row is a user's own answer, so every row has the same trust.

| Condition | Vote | Role |
|---|---|---|
| Frozen only | No memory | baseline |
| Exp 1 winner | similarity | candidate |
| Exp 1 winner + Freshness | similarity × freshness, half-life 6 / 3 / 1 months | candidates |
| Everything + trust + freshness | store every row; left-alone rows count 0.25; votes halve every month | comparison only, not a candidate |

**Choice** (on tune, saved to `results/exp2/chosen.json`): fewest manual checks, then highest
accuracy; a tie goes to the simpler setup (no freshness first, then the longest half-life).

**Results on tune:**

| Condition | Accuracy | Manual checks | Spread errors |
|---|---|---|---|
| Frozen only | 49.4% | 333 | 0 |
| **Exp 1 winner** (chosen) | **92.7%** | **48** | 4 |
| + Freshness 6 / 3 / 1 mo | 92.7% | 48 | 4 |
| Everything + trust + freshness | 93.3% | 44 | 0 |

- **Freshness changes nothing.** It only matters when stored neighbours disagree, and this memory
  almost never holds two labels for the same text: when coffee became Coffee, the old Restaurants
  rows had been left alone and were never stored. The tie rule picks the plain Exp 1 winner.
- **Known limitation: the gift card spreads.** After the May gift card, memory's only Trader Joe's
  neighbour is that Gifts correction, so all 4 June Trader Joe's rows are labeled Gifts. Corrections
  alone can't tell a one-off from a lasting change.
- **The comparison row** asks whether a smarter reader would rescue "store everything": counting
  left-alone rows at low weight with fast fading. On tune it fixes exactly those 4 rows (44 vs 48)
  and is otherwise identical. Its setting was picked from an exploratory 4 × 3 trust × freshness
  grid on tune (`results/exp2/exploratory/`), where only that one setting beat 48, so its tune
  number is optimistic. Val gives the fair comparison.

---

## Experiment 3: What to forget

**Question:** a real app can't keep every row forever. When memory is capped, which rows should
go so the system keeps working?

**Setup:** Experiment 2's choice (store corrected + answered rows, similarity vote), read from
`results/exp2/chosen.json`.

| Factor | Values |
|---|---|
| Memory size *K* | 10 / 20 / 30 rows, and unlimited (tune stores up to 48 rows a year) |
| Eviction rule | **Oldest first** · **Random** (reservoir sampling: every row seen so far has the same chance to stay; 5 seeds, averaged) · **Least useful** (lowest weight × freshness right now) · **Least useful + merge** (a new row first merges into a stored row with the same label and similarity ≥ 0.99; the merged row's weight is the number of rows in it) |

**Rare items** are measured generically: **long-gap accuracy** = accuracy on rows whose text was
seen before but not in the previous 3 months.

**Choice for the combined best** (on tune, saved to `results/exp3/chosen.json`): the rule with the
fewest manual checks summed over the three sizes, at the smallest *K* whose manual checks stay
within 10% of unlimited memory.

**Results on tune** (manual checks per year; accuracy):

| Rule | K=10 | K=20 | K=30 | Unlimited |
|---|---|---|---|---|
| Oldest first | 212 · 67.5% | 171 · 74.1% | 119 · 81.9% | 48 · 92.7% |
| Random (mean of 5) | 201 · 69.8% | 103 · 84.5% | 75 · 88.8% | 48 · 92.7% |
| Least useful | 212 · 67.5% | 171 · 74.1% | 119 · 81.9% | 48 · 92.7% |
| **Least useful + merge** | 173 · 73.7% | 75 · 88.7% | **48 · 92.7%** | 48 · 92.7% |

- **Every stored row matters here.** Each one is a correction or an answer, so forgetting any of
  them costs checks: oldest first more than doubles the checks at K=30.
- **Least useful = oldest first** without merging: with no freshness and every row a user's answer,
  all rows count the same and the tie goes to the oldest.
- **Merging is the win:** the year's 48 rows collapse into 29 with no loss (repeat corrections of
  the same shop, e.g. Trader Joe's, Philz), so K=30 holds everything. Chosen: least useful + merge,
  K=30. At smaller K, merged rows with many members survive and one-off rows go first.
- **Rare items:** at K=10, oldest first gets 24% of long-gap rows right; random 49%; least useful +
  merge 52%. Random beats oldest first, as reservoir sampling is meant to, but merging does best.
- **Caveat:** merging works well because the generated data repeats the same shop strings. Real
  statements vary more; the test set will show how much merging still compresses.

**Combined best on tune:** 92.7% accuracy, 48 manual checks a year, 29 rows of memory (frozen
only: 49.4%, 333).

---

## At the end: val and test

Once all three experiments are settled on tune (every setting locked):

| Run | What | Shows |
|---|---|---|
| **Val** (generated then, seed 2028) | Every experiment's conditions (`--val` on each script) **and** the combined best setup (best storage + best reading + best forgetting) vs frozen only | Each ranking holds on an unseen year, and the full system works |
| **Test** (friend's data) | The combined best setup vs frozen only, once | The headline number |

The combined best and the test run are in `experiments/final.py`:
`--val` and `--test FILE.csv`.

### Val results (run once, 2026-11-24, no settings changed)

| | Tune | Val |
|---|---|---|
| **Exp 1** (checks a year): frozen / corrections only / corrections + answered / everything | 333 / 139 / **48** / 123 | 311 / 147 / **55** / 94 |
| **Exp 2**: Exp 1 winner / + freshness (any half-life) / everything + trust + freshness | 48 / 48 / 44 | 55 / 55 / 51 |
| **Exp 3** at K=30: oldest first / random / least useful / least useful + merge | 119 / 75 / 119 / **48** | 86 / 67 / 86 / **55** |
| **Combined best vs frozen**: accuracy, checks a year | 49.4% → 92.7%, 333 → 48 | 55.2% → 92.0%, 311 → 55 |

Every ranking from tune holds on val. The comparison row beats the Exp 1 winner by 4 checks again,
again entirely through the 4 spread errors after the gift card (0 vs 4), so the trade-off it shows
is real but small. Outputs, including the printed tables (`console.txt`), are in `results/*/val/`.

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
experiments/exp2_read.py       Experiment 2 → results/exp2/<split>/, chosen.json (done on tune; old grid in exploratory/)
experiments/exp3_forget.py     Experiment 3 → results/exp3/<split>/, chosen.json (done on tune)
experiments/final.py           combined best vs frozen only → results/final/<split>/ (tune, val done; test at the end)
```
