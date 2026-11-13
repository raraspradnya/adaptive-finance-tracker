"""Generate a year of synthetic bank transactions for one simulated user.

Writes:
    data/transactions_2026.csv   one row per transaction, January to December 2026
    data/merchants.csv           the merchant catalog the rows were drawn from

Each transaction has two labels:
    general_category  what most people would call it (the frozen model's 16 categories)
    true_category     what THIS user wants. Usually the same, but not always:
        - Joe's Ramen and breweries are "Social" for this user (a category the model doesn't have)
        - coffee shops are "Restaurants" until June, then the user switches them to "Coffee"
          (the user changes their mind mid-year, and adds another new category)

Other changes built into the stream:
    - new merchants appear partway through the year (is_new_merchant = True)
    - the same merchant shows up under different raw strings (store numbers, POS prefixes)

Columns in transactions_2026.csv:
    txn_id, month, date, raw_description, merchant_key, merchant_group, amount, direction,
    general_category, true_category, label_source, is_new_merchant

    label_source: "general"     true_category == general_category
                  "preference"  the user always files this merchant differently
                  "changed"     the user changed their mind about this merchant this year

Run:  python src/generate_data.py   (same seed -> same files)
"""

import random
import string
from pathlib import Path

import pandas as pd

SEED = 290
YEAR = 2026
COFFEE_SWITCH_MONTH = 7  # from July, coffee shops become "Coffee" for this user
OUT_DIR = Path(__file__).resolve().parent.parent / "data"

# key, group, general category, raw-string templates, mean visits per month
# ("monthly" = exactly once, "twice" = exactly twice), amount range, first month it appears,
# direction. Template placeholders: {n3} {n4} {n5} {n8} 3-8 digit numbers, {code} 6 letters/digits,
# {mm}/{dd} the transaction date.
MERCHANTS = [
    # groceries
    ("trader_joes", "grocery", "Groceries", ["TRADER JOE'S #135 BERKELEY CA", "TRADER JOE'S #198 OAKLAND CA"], 3, (30, 75), 1, "debit"),
    ("safeway", "grocery", "Groceries", ["SAFEWAY #2741 OAKLAND CA", "SAFEWAY STORE 1507 BERKELEY"], 1.5, (25, 90), 1, "debit"),
    ("berkeley_bowl", "grocery", "Groceries", ["BERKELEY BOWL WEST", "BERKELEY BOWL MARKETPLACE"], 1.5, (20, 80), 1, "debit"),
    ("costco", "grocery", "Groceries", ["COSTCO WHSE #{n4} RICHMOND CA"], 0.6, (110, 220), 1, "debit"),
    ("monterey_market", "grocery", "Groceries", ["MONTEREY MARKET"], 0.5, (12, 40), 1, "debit"),
    # restaurants
    ("chipotle", "fast_casual", "Restaurants", ["CHIPOTLE 2297 BERKELEY", "CHIPOTLE ONLINE {n4}"], 1.5, (11, 18), 1, "debit"),
    ("cheese_board", "pizza", "Restaurants", ["CHEESE BOARD COLLECTIVE", "SQ *CHEESE BOARD PIZZA"], 0.8, (10, 30), 1, "debit"),
    ("doordash", "delivery", "Restaurants", ["DOORDASH*THAI BASIL", "DOORDASH*SUNDAY GAMES", "DOORDASH*CHIPOTLE"], 1.2, (18, 45), 1, "debit"),
    ("niks_kitchen", "fast_casual", "Restaurants", ["TST* NIKS KITCHEN"], 0.7, (14, 28), 1, "debit"),
    ("blue_door_cafe", "cafe", "Restaurants", ["SQ *BLUE DOOR CAFE"], 1, (8, 16), 1, "debit"),
    # this user's "Social" spending: ramen nights and breweries
    ("joes_ramen", "ramen", "Restaurants", ["JOE'S RAMEN SF CA", "JOES RAMEN #2 SF CA"], 1.2, (20, 32), 1, "debit"),
    ("anchor_brewing", "brewery", "Restaurants", ["THE ANCHOR BREWING SF"], 0.8, (25, 55), 1, "debit"),
    ("fieldwork_brewing", "brewery", "Restaurants", ["FIELDWORK BREWING BERKELEY", "SQ *FIELDWORK BREWING"], 0.9, (20, 50), 9, "debit"),
    # coffee: Restaurants until June, then "Coffee"
    ("peets", "coffee", "Restaurants", ["PEET'S COFFEE #8812", "PEET'S COFFEE #0731", "PEETS COFFEE & TEA {n4}"], 3, (4, 9), 1, "debit"),
    ("starbucks", "coffee", "Restaurants", ["STARBUCKS STORE {n5}", "STARBUCKS #{n5} BERKELEY CA"], 1, (4, 8), 1, "debit"),
    ("blue_bottle", "coffee", "Restaurants", ["BLUE BOTTLE COFFEE 0042", "BLUE BOTTLE COFFEE OAKLAND"], 1.5, (5, 9), 4, "debit"),
    # transportation
    ("clipper", "transit", "Transportation", ["CLIPPER SVC BART"], 2, (20, 20), 1, "debit"),
    ("lyft", "rideshare", "Transportation", ["LYFT *RIDE {mm}-{dd}"], 1, (9, 28), 1, "debit"),
    ("uber", "rideshare", "Transportation", ["UBER *TRIP HELP.UBER.COM"], 1, (10, 32), 1, "debit"),
    ("shell", "gas", "Transportation", ["SHELL OIL {n8} OAKLAND"], 0.5, (35, 60), 1, "debit"),
    # subscriptions
    ("netflix", "streaming", "Subscription", ["NETFLIX.COM 866-579-7172"], "monthly", (15.49, 15.49), 1, "debit"),
    ("spotify", "streaming", "Subscription", ["SPOTIFY USA"], "monthly", (11.99, 11.99), 1, "debit"),
    ("openai", "software", "Subscription", ["OPENAI *CHATGPT SUBSCR"], "monthly", (20, 20), 3, "debit"),
    # utilities
    ("pge", "power", "Utilities", ["PG&E WEB ONLINE"], "monthly", (55, 95), 1, "debit"),
    ("comcast", "internet", "Utilities", ["COMCAST CABLE COMM 800-COMCAST"], "monthly", (79.99, 79.99), 1, "debit"),
    ("tmobile", "phone", "Utilities", ["T-MOBILE AUTOPAY"], "monthly", (65, 65), 1, "debit"),
    # shopping
    ("amazon", "online_retail", "Shopping", ["AMZN MKTP US*{code}", "AMAZON.COM*{code}"], 2, (9, 80), 1, "debit"),
    ("target", "retail", "Shopping", ["TARGET 00012345 BERKELEY CA", "TARGET T-{n4} OAKLAND"], 0.8, (15, 70), 1, "debit"),
    ("uniqlo", "clothing", "Shopping", ["UNIQLO SF 0417"], 0.8, (30, 90), 10, "debit"),
    ("ashby_flea", "market", "Shopping", ["ASHBY FLEA"], 0.3, (10, 40), 1, "debit"),
    # healthcare and personal care
    ("cvs", "pharmacy", "Healthcare", ["CVS/PHARMACY #{n5}"], 0.4, (8, 35), 1, "debit"),
    ("walgreens", "pharmacy", "Healthcare", ["WALGREENS #3456 BERKELEY CA"], 0.4, (8, 30), 1, "debit"),
    ("great_clips", "salon", "Personal Care", ["GREAT CLIPS #1234"], 0.3, (20, 28), 1, "debit"),
    ("planet_fitness", "gym", "Personal Care", ["PLANET FITNESS CLUB FEES"], "monthly", (24.99, 24.99), 1, "debit"),
    # bills, transfers, income
    ("geico", "insurance", "Insurance", ["GEICO AUTO PAY"], "monthly", (112, 112), 1, "debit"),
    ("rent", "rent", "Housing", ["AVALON BAY RENT PAYMENT"], "monthly", (2450, 2450), 1, "debit"),
    ("venmo", "p2p", "Transfer", ["VENMO PAYMENT {n10}"], 1.5, (10, 80), 1, "debit"),
    ("chase_autopay", "card_payment", "Transfer", ["CHASE CREDIT CRD AUTOPAY PPD ID: 4760039224"], "monthly", (400, 1200), 1, "debit"),
    ("payroll", "payroll", "Income", ["GUSTO PAYROLL DIR DEP"], "twice", (2100, 2100), 1, "credit"),
    ("service_fee", "bank_fee", "Fees", ["MONTHLY SERVICE FEE"], 0.2, (12, 12), 1, "debit"),
    # entertainment, travel, education
    ("amc", "movies", "Entertainment", ["AMC METREON 16 SF", "AMC BAY STREET 16"], 0.6, (14, 32), 1, "debit"),
    ("steam", "games", "Entertainment", ["STEAM GAMES 425-889-9642"], 0.3, (5, 60), 1, "debit"),
    ("cal_performances", "concerts", "Entertainment", ["CAL PERFORMANCES ZELLERBACH"], 0.2, (35, 90), 1, "debit"),
    ("united", "airline", "Travel", ["UNITED AIRLINES {n10}"], 0.25, (180, 450), 6, "debit"),
    ("marriott", "hotel", "Travel", ["MARRIOTT SAN FRANCISCO UNION SQ"], 0.1, (250, 500), 6, "debit"),
    ("coursera", "online_course", "Education", ["COURSERA.ORG"], "monthly", (49, 49), 2, "debit"),
]

SOCIAL_MERCHANTS = {"joes_ramen", "anchor_brewing", "fieldwork_brewing"}

# Display names for the UI. Not used when drawing transactions.
NAMES = {
    "trader_joes": "Trader Joe’s", "safeway": "Safeway", "berkeley_bowl": "Berkeley Bowl", "costco": "Costco",
    "monterey_market": "Monterey Market", "chipotle": "Chipotle", "cheese_board": "Cheese Board",
    "doordash": "DoorDash", "niks_kitchen": "Nik’s Kitchen", "blue_door_cafe": "Blue Door Cafe",
    "joes_ramen": "Joe’s Ramen", "anchor_brewing": "Anchor Brewing", "fieldwork_brewing": "Fieldwork Brewing",
    "peets": "Peet’s Coffee", "starbucks": "Starbucks", "blue_bottle": "Blue Bottle Coffee",
    "clipper": "Clipper (BART)", "lyft": "Lyft", "uber": "Uber", "shell": "Shell",
    "netflix": "Netflix", "spotify": "Spotify", "openai": "ChatGPT", "pge": "PG&E", "comcast": "Comcast",
    "tmobile": "T-Mobile", "amazon": "Amazon", "target": "Target", "uniqlo": "Uniqlo", "ashby_flea": "Ashby Flea",
    "cvs": "CVS Pharmacy", "walgreens": "Walgreens", "great_clips": "Great Clips", "planet_fitness": "Planet Fitness",
    "geico": "GEICO", "rent": "Avalon Bay (rent)", "venmo": "Venmo", "chase_autopay": "Chase card payment",
    "payroll": "Payroll", "service_fee": "Bank service fee", "amc": "AMC Theatres", "steam": "Steam",
    "cal_performances": "Cal Performances", "united": "United Airlines", "marriott": "Marriott", "coursera": "Coursera",
}


def user_label(key, group, general, month):
    """What this user wants the transaction filed as, and why."""
    if key in SOCIAL_MERCHANTS:
        return "Social", "preference"
    if group == "coffee" and month >= COFFEE_SWITCH_MONTH:
        return "Coffee", "changed"
    return general, "general"


def fill(template, rng, month, day):
    digits = lambda k: "".join(rng.choices(string.digits, k=k))
    return template.format(
        n3=digits(3), n4=digits(4), n5=digits(5), n8=digits(8), n10=digits(10),
        code="".join(rng.choices(string.ascii_uppercase + string.digits, k=6)),
        mm=f"{month:02d}", dd=f"{day:02d}",
    )


def visits(freq, rng):
    if freq == "monthly":
        return 1
    if freq == "twice":
        return 2
    # Poisson draw with the stdlib, so the generator needs no numpy
    count, p, threshold = 0, rng.random(), pow(2.718281828, -freq)
    while p > threshold:
        count += 1
        p *= rng.random()
    return count


def generate():
    rng = random.Random(SEED)
    rows = []
    for month in range(1, 13):
        days_in_month = pd.Period(f"{YEAR}-{month:02d}").days_in_month
        for key, group, general, templates, freq, (lo, hi), start, direction in MERCHANTS:
            if month < start:
                continue
            n = visits(freq, rng)
            fixed_days = [1] if freq == "monthly" else [1, 15] if freq == "twice" else None
            for i in range(n):
                day = fixed_days[i] if fixed_days else rng.randint(1, days_in_month)
                true, source = user_label(key, group, general, month)
                rows.append({
                    "month": month,
                    "date": f"{YEAR}-{month:02d}-{day:02d}",
                    "raw_description": fill(rng.choice(templates), rng, month, day),
                    "merchant_key": key,
                    "merchant_group": group,
                    "amount": round(rng.uniform(lo, hi), 2),
                    "direction": direction,
                    "general_category": general,
                    "true_category": true,
                    "label_source": source,
                    "is_new_merchant": start > 1,
                })
    df = pd.DataFrame(rows).sort_values(["date", "merchant_key"], kind="stable").reset_index(drop=True)
    df.insert(0, "txn_id", range(1, len(df) + 1))
    return df


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(OUT_DIR / "transactions_2026.csv", index=False)
    catalog = pd.DataFrame(
        [(k, NAMES[k], g, c, " | ".join(t), str(f), s) for k, g, c, t, f, _, s, _ in MERCHANTS],
        columns=["merchant_key", "name", "merchant_group", "general_category", "raw_templates", "visits_per_month", "first_month"],
    )
    catalog.to_csv(OUT_DIR / "merchants.csv", index=False)

    print(f"{len(df)} transactions, {df['merchant_key'].nunique()} merchants -> {OUT_DIR}")
    print("\nTransactions per month:")
    print(df.groupby("month").size().to_string())
    print("\nWhere the user's label differs from the general one:")
    print(df[df["label_source"] != "general"].groupby(["label_source", "true_category"]).size().to_string())
    print(f"\nRows from merchants that appear after January: {df['is_new_merchant'].sum()}")


if __name__ == "__main__":
    main()
