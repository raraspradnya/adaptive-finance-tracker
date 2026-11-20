"""Local server for the prototype UI, backed by the frozen model.

Serves prototype/index.html and a small JSON API:
    GET /api/init               categories, monthly files, and the months already reviewed
    GET /api/statement?month=N  that month's transactions, categorized live by the frozen model

Months before SEEDED_MONTHS count as already reviewed: the UI starts with their totals
(using the user's own labels) and the user uploads the remaining months one at a time.

Run:  python src/server.py   then open http://127.0.0.1:8290
Only listens on this computer.
"""

import json
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from frozen_base import FrozenModel

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "prototype" / "index.html"
HOST, PORT = "127.0.0.1", 8290
SEEDED_MONTHS = 6                      # January to June start out reviewed
NON_SPENDING = ["Income", "Transfer"]  # shown in review, left out of spending and budgets
MONTH_ABBR = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

txns = pd.read_csv(ROOT / "data" / "transactions_2026.csv", keep_default_na=False)
txns["month"] = pd.to_datetime(txns["date"]).dt.month
print("Loading the frozen model...")
model = FrozenModel()


def init_payload():
    seeded = txns[txns["month"] <= SEEDED_MONTHS]
    seeded = seeded[~seeded["true_category"].isin(NON_SPENDING)]
    spend = {
        cat: [round(-float(group.loc[group["month"] == m, "amount"].sum()), 2) + 0.0 for m in range(1, SEEDED_MONTHS + 1)]
        for cat, group in seeded.groupby("true_category")
    }
    return {
        "categories": model.categories,
        "nonSpending": NON_SPENDING,
        "customCategories": sorted(set(spend) - set(model.categories)),  # categories this user made themselves
        "seeded": SEEDED_MONTHS,
        "spend": spend,
        "files": [
            {"m": m - 1, "file": f"{'chase' if m < 10 else 'bofa'}_{MONTH_ABBR[m - 1]}_2026.csv", "n": int((txns["month"] == m).sum())}
            for m in range(1, 13)
        ],
    }


def statement_payload(month):
    rows = txns[txns["month"] == month]
    preds = model.predict(rows["description"])  # the frozen model, run live on this upload
    return [
        {
            "date": date.fromisoformat(r.date).strftime("%b %d"),
            "raw": r.description,
            "note": r.note,
            "amt": abs(float(r.amount)),
            "credit": r.amount > 0,
            "base": cat,
            "conf": conf,
        }
        for r, (cat, conf) in zip(rows.itertuples(), preds)
    ]


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, status=200):
        self._send(status, json.dumps(obj), "application/json")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/init":
            self._json(init_payload())
        elif url.path == "/api/statement":
            try:
                month = int(parse_qs(url.query)["month"][0])
                assert 1 <= month <= 12
            except (KeyError, ValueError, AssertionError):
                return self._json({"error": "month must be 1-12"}, 400)
            self._json(statement_payload(month))
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        if "/api/" in self.path:
            print(f"{self.command} {self.path} -> {args[1]}")


if __name__ == "__main__":
    print(f"Open http://{HOST}:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
