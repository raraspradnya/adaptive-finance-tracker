# Adaptive Finance Tracker

A personal finance tracker that learns from your corrections. When you move a transaction to a
different category, the tracker applies that to future uploads, so you stop fixing the same
mistake every month. It does this without disturbing what it already got right.

The classifier itself never changes: a frozen DistilBERT labels each transaction, and an external
**memory** of transactions you have reviewed corrects it over time. All the learning happens in
that memory, not in the model's weights.

Scope: the project covers the categorizer only. Budgets and budget recommendations are out of
scope.

## Results

Each year is replayed month by month: the system labels a month using only what it learned in
earlier months, a simulated user reviews it, and the reviewed rows go into memory.

| Year | Accuracy (frozen → with memory) | Rows the user had to check or fix a year | Memory |
|---|---|---|---|
| Tune (`data/transactions_2026.csv`, where every setting was chosen) | 49.4% → 92.7% | 333 → 48 | 29 rows |
| Val (an unseen generated year, run once) | 55.2% → 92.0% | 311 → 55 | 29 rows |
| Test (blind data from a friend) | still to run | | |

The final setup, chosen through three experiments (details and every number in
[EXPERIMENTS.md](EXPERIMENTS.md)):

1. **What to store:** only rows the user corrected or answered when asked. Rows the user left alone
   are not stored, because at full weight they outvote real changes.
2. **How to read:** a new row takes the label of near-identical stored rows (cosine similarity
   ≥ 0.93 on frozen embeddings, ≥ 60% of the vote); otherwise the frozen label, flagged for a check
   when the model is unsure. Weighting votes by freshness added nothing.
3. **What to forget:** memory holds at most 30 rows. Near-identical rows with the same label merge
   into one, and the least useful row goes first when memory is full.

Known limitation: a one-off exception (a Trader Joe's gift card labeled Gifts) spreads to the next
month's grocery trips. Storing left-alone rows at low weight fixes it (4 fewer checks on both tune
and val), at the cost of a much larger memory.

## Try the prototype

The prototype runs the frozen model only for now; the memory will be integrated soon (see the note
below).

Set up once:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Start the local server, then open http://127.0.0.1:8290:

```bash
.venv/bin/python src/server.py
```

The server runs a frozen transaction classifier
([DoDataThings/distilbert-us-transaction-classifier](https://huggingface.co/DoDataThings/distilbert-us-transaction-classifier),
downloaded on first run) on a year of synthetic transactions for one Berkeley student in
`data/transactions_2026.csv`. January to June start out reviewed. Each upload from July on is
categorized live by the model. Everything runs on this computer, and nothing is saved between
sessions.

- **See your spending.** The dashboard compares 2026 month by month and shows where a month's money
  went by category.
- **Upload a bank statement.** Rows show the bank's own text, money in (green, +) or out (−), and
  the Venmo note if there is one. Rows the model is unsure about are marked "Check this": answer
  each with ✓ (keep) or ✗ (pick the right category). "Save review" unlocks once every one is
  answered. You can change any other row or add a new category.

> **Note:** the prototype currently shows the frozen model's labels only. The continual-learning
> memory (the final setup above) is not connected to the UI yet, so your corrections don't carry
> over to the next upload. Integrating it is the next step.

To see what the frozen model does with her shifts, upload July (coffee shops are still
Restaurants), then October (the bank switch: most card purchases come back as Transfer).

## Run the experiments

Every choice is made on tune; val and test are each run once at the end.

```bash
.venv/bin/python experiments/exp1_storage.py        # what to store
.venv/bin/python experiments/exp2_read.py           # how to read (builds on Exp 1's winner)
.venv/bin/python experiments/exp3_forget.py         # what to forget (builds on Exp 2's choice)
.venv/bin/python experiments/final.py               # combined best vs frozen only
```

Add `--val` to any of them to run on the val year (already done; results in `results/*/val/`).
The blind test set goes through the final setup only:

```bash
.venv/bin/python experiments/final.py --test FILE.csv
```

The test file needs the columns `txn_id, date, description, amount, note, true_category`.
Results go to `results/<experiment>/<split>/`: logs, summary and monthly tables, charts. Exps 2 and
3 also save their choice to `results/exp2/chosen.json` and `results/exp3/chosen.json`, which the
next step reads. The first run of each year computes the frozen model's outputs once and caches
them in `results/cache/`, so later runs take seconds.

## Files

```
src/data_generator.py          synthetic years: tune (default) and val (--val); the docstring lists
                               the persona and every shift built into the year
src/frozen_base.py             the frozen model; also runs it on any CSV and reports accuracy
src/memory.py                  normalize(), the memory store, the reader (vote), storing, merging, eviction
src/simulation.py              the simulated user and the month-by-month replay
src/metrics.py                 metrics tables and charts
src/server.py                  serves the prototype UI (prototype/index.html) with the frozen model
experiments/                   pick_tau.py (similarity cutoff), exp1-3, final.py, common.py
tests/test_memory.py           checks: .venv/bin/python -m pytest -q
tools/review_month_by_hand.py  review a month in the terminal, as the simulated user would (--month 7)
eval.ipynb                     frozen-model baseline accuracy by category
EXPERIMENTS.md                 the experiment plan, every setting and every result
```

`results/frozen_predictions_2026.csv` is the frozen model on the raw bank text (41.6% accuracy).
The experiments first clean the text with `normalize()` (strips reference numbers and bank
wrappers), which lifts the frozen model to 49.4%; that is the baseline in the tables above.
