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

Open `prototype/index.html` in a browser. It runs on sample data, and nothing is saved or sent.

To see the learning:

1. Go to **Upload & review** and upload the November file.
2. Click the category on a Joe's Ramen row, pick a new one, and apply it to every Joe's Ramen transaction.
3. Click **Accept all**.
4. Upload the December file. Joe's Ramen arrives in your new category.
