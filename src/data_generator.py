"""Generate a year of bank transactions for one simulated user.

The user: a 25-year-old Berkeley student who studies and hangs out in coffee shops and
breweries. Housing is about $1,500 a month (rent to the landlord plus her share to her roommate
Ashley), and total spending is about $2,500 a month. She is paid twice a month.

Writes a CSV with only what a bank export would show, plus the answer key:

    txn_id, date, description, amount, note, true_category

    amount         signed: negative = money out, positive = money in
    note           the Venmo note (from her Venmo export); empty for everything else
    true_category  what THIS user wants the row filed as. Never shown to the system:
                   only the simulated user and the scoring read it.

Categories are the frozen model's 16, plus three of her own: Social, Coffee, Gifts.

Shifts built into the dev year, 2026 (see EXPERIMENTS.md). The validation year (--val) has the
same kinds of shift, but each one lands in a month drawn at random from its seed:
    concept    all year  ramen and breweries are "Social" (the model has no such category)
    concept    May       one $50 Trader Joe's purchase is a gift card -> "Gifts", once
    concept    Jul       coffee shops change from "Restaurants" to "Coffee"
    covariate  Sep       rent goes from $1,225 to $1,300 (still Housing)
    covariate  Sep       a new brewery, Ale Industries, appears (still Social)
    covariate  Oct       she switches from Chase to Bank of America: every description
                         changes format (CHECKCARD prefixes, Zelle names written LAST FIRST, ...)
    concept    Nov       Ashley moved out at the end of October: Zelle to Ashley changes from
                         $275 rent share (Housing) to small payments for drinks (Social)

Run:  .venv/bin/python src/data_generator.py         -> data/transactions_2026.csv  (tune)
      .venv/bin/python src/data_generator.py --val   -> data/transactions_val.csv   (validation)
Each also writes <name>.events.json with the months its shifts happen in (used for chart markers).
Same seed -> same file.
"""

import argparse
import json
import random
import string
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Events:
    """The month each shift happens in. Defaults are the dev year."""
    coffee_switch: int = 7         # coffee becomes "Coffee"
    rent_increase: int = 9         # rent $1,225 -> $1,300
    new_brewery: int = 9           # Ale Industries appears
    bank_switch: int = 10          # Chase -> Bank of America
    ashley_moves_out: int = 11     # from this month, Zelle to Ashley is Social
    gift_card: tuple = (5, 9)      # (month, day) of the one-off Trader Joe's gift card


DEV = dict(seed=2026, year=2026, events=Events(), out=DATA / "transactions_2026.csv")


def random_events(seed):
    """Each shift in a random month, leaving months before and after it to learn from."""
    r = random.Random(seed * 7 + 1)   # its own stream, so the transactions' draws don't change
    return Events(coffee_switch=r.randint(3, 10), rent_increase=r.randint(2, 11),
                  new_brewery=r.randint(2, 11), bank_switch=r.randint(3, 11),
                  ashley_moves_out=r.randint(3, 11), gift_card=(r.randint(2, 10), r.randint(1, 28)))


# Seed 2027 was used for an earlier val year that got looked at during Experiment 1, so it was
# deleted. This one is generated only after Experiments 1-3 are settled, and run once.
VAL = dict(seed=2028, year=2028, events=random_events(2028), out=DATA / "transactions_val.csv")

LANDLORD = "BAYVIEW PROPERTY MGMT"
ASHLEY = ("ASHLEY", "KIM")
FRIENDS = [("SAM", "LEE"), ("PRIYA", "SHAH"), ("JESS", "NGUYEN")]


# Card purchases: (descriptions as the merchant sends them, visits per month,
# amount range, label for a given month, first month, last month)
def card_list(ev):
    def coffee_label(month):
        return "Coffee" if month >= ev.coffee_switch else "Restaurants"
    return [
        # coffee shops: she studies here most days
        (["PEETS COFFEE #0731 BERKELEY CA", "PEET'S COFFEE #8812"], 5, (4.5, 8.5), coffee_label, 1, 12),
        (["BLUE BOTTLE COFFEE BERKELEY"], 3, (5.0, 8.0), coffee_label, 1, 12),
        (["CAFFE STRADA BERKELEY CA", "SQ *CAFFE STRADA"], 4, (4.0, 7.5), coffee_label, 1, 12),
        (["PHILZ COFFEE BERKELEY"], 2, (5.5, 8.0), coffee_label, 1, 12),
        # breweries and ramen: her "Social" spending
        (["THE RARE BARREL BERKELEY"], 2, (18, 42), lambda m: "Social", 1, 12),
        (["FIELDWORK BREWING BERKELEY", "SQ *FIELDWORK BREWING"], 2, (16, 40), lambda m: "Social", 1, 12),
        (["TST* ALE INDUSTRIES OAKLAND"], 2, (18, 45), lambda m: "Social", ev.new_brewery, 12),
        (["IPPUDO BERKELEY", "RAMEN SHOP OAKLAND CA"], 2, (17, 30), lambda m: "Social", 1, 12),
        # ordinary restaurants
        (["CHIPOTLE 2297 BERKELEY", "SWEETGREEN BERKELEY", "CHEESE BOARD COLLECTIVE", "DOORDASH*TACOS EL GORDO"],
         5, (11, 28), lambda m: "Restaurants", 1, 12),
        # groceries
        (["TRADER JOE'S #186 BERKELEY CA"], 3, (28, 70), lambda m: "Groceries", 1, 12),
        (["BERKELEY BOWL WEST", "BERKELEY BOWL MARKETPLACE"], 2, (14, 45), lambda m: "Groceries", 1, 12),
        (["SAFEWAY #0975 BERKELEY CA"], 0.5, (12, 40), lambda m: "Groceries", 1, 12),
        # getting around
        (["CLIPPER SVC BART"], "twice", (20, 20), lambda m: "Transportation", 1, 12),
        (["LYFT *RIDE {mm}-{dd}"], 1, (9, 24), lambda m: "Transportation", 1, 12),
        # bills and subscriptions
        (["MINT MOBILE 800-683-7392"], "monthly", (30, 30), lambda m: "Utilities", 1, 12),
        (["SPOTIFY USA"], "monthly", (5.99, 5.99), lambda m: "Subscription", 1, 12),
        (["NETFLIX.COM 866-579-7172"], "monthly", (15.49, 15.49), lambda m: "Subscription", 1, 12),
        (["APPLE.COM/BILL 866-712-7753"], "monthly", (2.99, 2.99), lambda m: "Subscription", 1, 12),
        # shopping, care, the rest
        (["AMZN MKTP US*{code}", "AMAZON.COM*{code}"], 1.5, (10, 60), lambda m: "Shopping", 1, 12),
        (["TARGET T-2788 BERKELEY CA"], 0.7, (15, 60), lambda m: "Shopping", 1, 12),
        (["UNIQLO BERKELEY 0417"], 0.3, (30, 80), lambda m: "Shopping", 1, 12),
        (["SEPHORA #0412 BERKELEY"], 0.4, (20, 55), lambda m: "Personal Care", 1, 12),
        (["SQ *SHATTUCK HAIR STUDIO"], 0.3, (45, 65), lambda m: "Personal Care", 1, 12),
        (["CVS/PHARMACY #{n5}"], 0.3, (8, 30), lambda m: "Healthcare", 1, 12),
        (["AMC BAY STREET 16"], 0.4, (14, 20), lambda m: "Entertainment", 1, 12),
        (["CAL STUDENT STORE BERKELEY"], 0.3, (15, 60), lambda m: "Education", 1, 12),
    ]

# Venmo payments to friends: (note, amount range, label)
VENMO_NOTES = [
    ("ramen 🍜", (14, 26), "Social"),
    ("drinks at rare barrel", (12, 30), "Social"),
    ("brewery tab", (15, 35), "Social"),
    ("pizza night", (10, 20), "Restaurants"),
    ("uber home", (8, 18), "Transportation"),
    ("costco run", (20, 45), "Groceries"),
    ("concert tix", (35, 60), "Entertainment"),
]


# ---------------------------------------------------------------- bank formats

def digits(rng, k):
    return "".join(rng.choices(string.digits, k=k))


def code(rng, k):
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=k))


def card_text(merchant, d, rng, ev):
    if d.month < ev.bank_switch:
        return merchant
    return f"CHECKCARD {d:%m%d} {merchant} {digits(rng, 11)}"


def zelle_text(payee, d, rng, ev):
    """payee is (FIRST, LAST) for a person, or a plain string for a business."""
    if d.month < ev.bank_switch:
        name = " ".join(payee).title() if isinstance(payee, tuple) else payee.title()
        return f"Zelle payment to {name} JPM99{code(rng, 7)}"
    name = f"{payee[1]} {payee[0]}" if isinstance(payee, tuple) else payee   # BofA: LAST FIRST
    return f"ZELLE TO {name} CONF# {code(rng, 9).lower()}"


def venmo_text(d, rng, ev):
    if d.month < ev.bank_switch:
        return f"VENMO PAYMENT {digits(rng, 10)} WEB ID: 3264681992"
    return f"VENMO DES:PAYMENT ID:{digits(rng, 10)} CO ID:3264681992 WEB"


def payroll_text(d, rng, ev):
    if d.month < ev.bank_switch:
        return "UC BERKELEY PAYROLL PPD ID: 1946000123"
    return f"UC BERKELEY DES:PAYROLL ID:{digits(rng, 9)} CO ID:1946000123 PPD"


# ---------------------------------------------------------------- generation

def visits(freq, rng):
    if freq == "monthly":
        return 1
    if freq == "twice":
        return 2
    count, p, threshold = 0, rng.random(), pow(2.718281828, -freq)   # Poisson draw
    while p > threshold:
        count += 1
        p *= rng.random()
    return count


def generate(seed, year, ev):
    rng = random.Random(seed)
    rows = []
    card = card_list(ev)

    def add(d, description, amount, label, note=""):
        rows.append({"date": d, "description": description, "amount": round(amount, 2),
                     "note": note, "true_category": label})

    for month in range(1, 13):
        last_day = pd.Period(f"{year}-{month:02d}").days_in_month
        day = lambda: date(year, month, rng.randint(1, last_day))

        # income: paid on the 1st and the 15th
        for dd in (1, 15):
            d = date(year, month, dd)
            add(d, payroll_text(d, rng, ev), 1350.00, "Income")

        # rent to the landlord, by Zelle on the 1st
        d = date(year, month, 1)
        add(d, zelle_text(LANDLORD, d, rng, ev), -(1300.00 if month >= ev.rent_increase else 1225.00), "Housing")

        # Ashley: rent share until she moves out, then small payments for nights out
        if month < ev.ashley_moves_out:
            d = date(year, month, 2)
            add(d, zelle_text(ASHLEY, d, rng, ev), -275.00, "Housing")
        else:
            for _ in range(rng.randint(2, 3)):
                d = day()
                add(d, zelle_text(ASHLEY, d, rng, ev), -round(rng.uniform(12, 38), 2), "Social")

        # card purchases
        for templates, freq, (lo, hi), label, first, last in card:
            if not first <= month <= last:
                continue
            fixed = {"monthly": [3], "twice": [5, 19]}.get(freq)
            for i in range(visits(freq, rng)):
                d = date(year, month, fixed[i]) if fixed else day()
                merchant = rng.choice(templates).format(
                    mm=f"{d:%m}", dd=f"{d:%d}", code=code(rng, 6), n5=digits(rng, 5))
                add(d, card_text(merchant, d, rng, ev), -rng.uniform(lo, hi), label(month))

        # Venmo to friends, with notes
        for _ in range(visits(4, rng)):
            note, (lo, hi), label = rng.choice(VENMO_NOTES)
            d = day()
            add(d, venmo_text(d, rng, ev), -rng.uniform(lo, hi), label, note=note)

        # one-off: a Trader Joe's gift card for a friend
        if month == ev.gift_card[0]:
            d = date(year, *ev.gift_card)
            add(d, card_text("TRADER JOE'S #186 BERKELEY CA", d, rng, ev), -50.00, "Gifts")

        # flight home for the holidays
        if month == 12:
            d = date(year, 12, 6)
            add(d, card_text(f"UNITED AIRLINES {digits(rng, 10)}", d, rng, ev), -rng.uniform(280, 360), "Travel")

    df = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
    df.insert(0, "txn_id", range(1, len(df) + 1))
    df["date"] = df["date"].map(lambda d: d.isoformat())
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--val", action="store_true", help="write the validation year instead of dev")
    args = ap.parse_args()
    spec = VAL if args.val else DEV
    out = spec["out"]

    df = generate(spec["seed"], spec["year"], spec["events"])
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    out.with_suffix(".events.json").write_text(json.dumps(asdict(spec["events"]), indent=2) + "\n")

    month = pd.to_datetime(df["date"]).dt.month
    spent = -df.loc[df["amount"] < 0, "amount"].groupby(month).sum()
    print(f"{len(df)} transactions -> {out.relative_to(out.parent.parent)}")
    if args.val:   # stay blind to val: no months, no per-month or per-category breakdown
        return
    print(f"Shift months: {asdict(spec['events'])}")
    print("\nPer month: rows, money out")
    print(pd.DataFrame({"rows": df.groupby(month).size(), "spent": spent.round(0)}).to_string())
    print("\nRows per category:")
    print(df["true_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
