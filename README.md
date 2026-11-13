# Adaptive Finance Tracker

A personal finance tracker that learns from your corrections. When you move a transaction to a
different category, the tracker applies that to future uploads, so you stop fixing the same
mistake every month. It does this without disturbing what it already got right.

## What you can do

- **See your spending.** The dashboard compares 2026 month by month, shows where a month's money
  went by category, and tracks each category against its budget.
- **Upload a bank statement.** The tracker sorts each transaction into a category. You can change
  any row, add a new category, then accept the whole file to add it to your dashboard.
- **Teach it as you go.** When you change a category, choose whether it applies to one transaction,
  every transaction from that merchant, or similar merchants.
- **Set your budget.** The tracker proposes a budget for this month and next month. Change any
  amount, and it takes your change into account when it proposes the next month.

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
downloaded on first run) on a year of synthetic transactions in `data/transactions_2026.csv`.
January to June start out reviewed. Each upload from July on is categorized live by the model.
Everything runs on this computer, and nothing is saved between sessions.

To see the learning:

1. Go to **Upload & review** and upload the July file.
2. Click the category on a Peet's Coffee row, add a new category called Coffee, and apply it to every Peet's transaction.
3. Click **Accept all**.
4. Upload the August file. Peet's arrives as Coffee. The model alone would still say Restaurants.

## Project scripts

- `src/generate_data.py` writes the synthetic data (same seed, same files).
- `src/frozen_model.py` runs the frozen model on any CSV and reports accuracy:
  `.venv/bin/python src/frozen_model.py data/transactions_2026.csv --out results/frozen_predictions_2026.csv`
- `src/server.py` serves the prototype UI backed by the frozen model.
