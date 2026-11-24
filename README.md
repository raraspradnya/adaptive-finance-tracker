# Adaptive Finance Tracker

A personal finance tracker that learns from your corrections. When you move a transaction to a
different category, the tracker applies that to future uploads, so you stop fixing the same
mistake every month. It does this without disturbing what it already got right.

## What you can do

- **See your spending.** The dashboard compares 2026 month by month and shows where a month's money
  went by category.
- **Upload a bank statement.** The tracker sorts each transaction into a category. Rows show the
  bank's own text, money in (green, +) or money out (−), and the Venmo note if there is one. You
  can change any row or add a new category (every category gets its own color), then accept the
  file to add it to your dashboard.

Learning from corrections comes from the external memory, which is not wired into the UI yet: for
now every row shows the frozen model's label.

Scope: the project covers the categorizer only. Budgets and budget recommendations are out of
scope.

## Try it

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
downloaded on first run) on a year of synthetic transactions for one Berkeley student in `data/transactions_2026.csv`.
January to June start out reviewed. Each upload from July on is categorized live by the model.
Everything runs on this computer, and nothing is saved between sessions.

To see what the frozen model does with her shifts, upload July (coffee shops are still
Restaurants), then October (the bank switch: most card purchases come back as Transfer).

## Project scripts

- `src/data_generator.py` writes the synthetic data (same seed, same file). Its docstring lists the
  persona and every shift built into the year.
- `tools/review_month_by_hand.py --month 7` lets you review a month by hand in the terminal, as the
  simulated user would.
- `src/frozen_base.py` runs the frozen model on any CSV and reports accuracy:
  `.venv/bin/python src/frozen_base.py data/transactions_2026.csv --out results/frozen_predictions_2026.csv`
- `src/server.py` serves the prototype UI backed by the frozen model.
- `src/memory.py` is the shared contract for the categorizer: the data types, the memory
  store, and the functions each teammate owns (`decide` on the read side; `simulate_review`,
  `store`, `evict`, `run_year` on the write side). Checks: `.venv/bin/python -m pytest -q`
