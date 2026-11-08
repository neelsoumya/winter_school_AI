#!/usr/bin/env python3
"""
generate_synthetic_data.py
--------------------------
Generates six synthetic, India-flavoured business datasets for the hackathon.

    python generate_synthetic_data.py                 # writes CSVs to ./data
    python generate_synthetic_data.py --scale 2 --seed 7 --out my_data

Every dataset has:
  * planted signal (so a good model WILL beat a naive baseline)
  * realistic mess (missing values, duplicates, outliers, censored demand)
  * a target column tied to a business cost, so models can be judged in rupees

All data is fictional. Patterns are planted by us, so real-world conclusions
must NOT be drawn from it. The point is the workflow: problem -> data -> model
-> unit economics -> pitch.
"""
import argparse
import os

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def draw_with_rate(logit, rate, rng):
    """Draw 0/1 labels from a logistic model, shifting the intercept so the
    overall positive rate is ~`rate` (e.g. 12% default, 4% fraud)."""
    lo, hi = -15.0, 15.0
    for _ in range(60):  # bisection on the intercept
        mid = (lo + hi) / 2
        if sigmoid(logit + mid).mean() < rate:
            lo = mid
        else:
            hi = mid
    p = sigmoid(logit + (lo + hi) / 2)
    return (rng.random(len(logit)) < p).astype(int)


def add_missing(df, cols, frac, rng):
    """Blank out a fraction of values to mimic messy real-world data."""
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype(float)
        df.loc[rng.random(len(df)) < frac, c] = np.nan
    return df


CITY_TIER = {
    "Mumbai": 1, "Pune": 1, "Bengaluru": 1, "Delhi NCR": 1,
    "Nagpur": 2, "Nashik": 2, "Indore": 2, "Jaipur": 2,
    "Raipur": 3, "Guwahati": 3, "Dehradun": 3,
}


# ----------------------------------------------------------------------------
# 1. Credit scoring for gig workers and small merchants (NBFC / fintech)
# ----------------------------------------------------------------------------
def make_credit(n, rng):
    city = rng.choice(list(CITY_TIER), n)
    tier = np.array([CITY_TIER[c] for c in city])
    occ_base = {
        "delivery_partner": 22000, "kirana_owner": 38000, "cab_driver": 30000,
        "tailor_artisan": 18000, "freelancer": 35000, "street_vendor": 16000,
    }
    occ = rng.choice(list(occ_base), n, p=[.25, .20, .15, .10, .15, .15])
    base = pd.Series(occ).map(occ_base).values
    tier_mult = np.where(tier == 1, 1.2, np.where(tier == 2, 1.0, 0.85))
    income = base * tier_mult * rng.lognormal(0, 0.3, n)

    unstable = np.isin(occ, ["delivery_partner", "street_vendor", "freelancer"])
    volatility = np.clip(rng.beta(2, 5, n) + 0.1 * unstable, 0, 1)
    age = np.clip(rng.normal(32, 7, n), 21, 60).round().astype(int)
    months_upi = rng.integers(1, 72, n)
    upi_txn = rng.poisson(20 + income / 1500)
    avg_upi_val = np.round(rng.lognormal(np.log(350), 0.6, n))
    bill_reg = np.round(rng.beta(5, 2, n), 2)
    existing_emi = np.round(income * rng.beta(2, 8, n) * 0.8)
    loan = rng.choice([20000, 40000, 60000, 100000, 150000, 250000], n,
                      p=[.20, .25, .20, .20, .10, .05])
    tenure = rng.choice([6, 12, 18, 24], n)
    emi_burden = (loan / tenure + existing_emi) / income

    logit = (3.0 * volatility - 2.0 * bill_reg + 2.5 * emi_burden
             - 0.015 * months_upi - 0.01 * upi_txn
             + rng.normal(0, 0.5, n))
    defaulted = draw_with_rate(logit, 0.12, rng)

    df = pd.DataFrame({
        "applicant_id": [f"A{i:06d}" for i in range(n)],
        "age": age, "city": city, "city_tier": tier, "occupation": occ,
        "monthly_income": income.round(-2), "income_volatility": volatility.round(2),
        "upi_txn_per_month": upi_txn, "avg_upi_txn_value": avg_upi_val,
        "months_on_upi": months_upi, "bill_payment_regularity": bill_reg,
        "existing_emi": existing_emi, "loan_amount": loan,
        "tenure_months": tenure, "defaulted": defaulted,
    })
    df = add_missing(df, ["monthly_income"], 0.06, rng)
    df = add_missing(df, ["bill_payment_regularity"], 0.10, rng)
    # 1% duplicate applications (a classic data quality problem)
    df = pd.concat([df, df.sample(frac=0.01, random_state=1)], ignore_index=True)
    return df


# ----------------------------------------------------------------------------
# 2. Kirana / quick-commerce dark-store demand (with stockouts = censored demand)
# ----------------------------------------------------------------------------
FESTIVALS = {  # approximate 2025 windows
    "Holi": ("2025-03-13", "2025-03-14"),
    "Eid": ("2025-03-30", "2025-04-01"),
    "Ganesh Chaturthi": ("2025-08-27", "2025-09-06"),
    "Navratri": ("2025-09-22", "2025-10-01"),
    "Diwali": ("2025-10-18", "2025-10-23"),
}


def make_demand(rng, n_stores=20):
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    cities = ["Mumbai", "Pune", "Nagpur", "Indore"]
    stores = pd.DataFrame({
        "store_id": [f"S{i:03d}" for i in range(n_stores)],
        "city": rng.choice(cities, n_stores),
        "store_size": rng.lognormal(0, 0.3, n_stores).round(2),
    })
    skus = pd.DataFrame({
        "sku": ["atta_5kg", "milk_1l", "instant_noodles", "cold_drink_2l", "umbrella"],
        "price": [280, 66, 14, 95, 350],
        "base_units": [12, 40, 25, 15, 2],
    })

    # city-level rain: heavy June-September
    rain_rows = []
    for c in cities:
        monsoon = (dates.month >= 6) & (dates.month <= 9)
        wet = rng.random(len(dates)) < np.where(monsoon, 0.65, 0.04)
        mm = np.where(wet, rng.exponential(np.where(monsoon, 25, 8)), 0).round(1)
        rain_rows.append(pd.DataFrame({"city": c, "date": dates, "rain_mm": mm}))
    rain = pd.concat(rain_rows)

    df = (stores.merge(pd.DataFrame({"date": dates}), how="cross")
                .merge(skus, how="cross")
                .merge(rain, on=["city", "date"], how="left"))
    n = len(df)

    festival = np.zeros(n, dtype=bool)
    for start, end in FESTIVALS.values():
        festival |= ((df["date"] >= start) & (df["date"] <= end)).values
    month = df["date"].dt.month.values
    weekend = (df["date"].dt.dayofweek >= 5).values
    discount = rng.choice([0, 0, 0, 5, 10, 15], n)

    mult = np.ones(n)
    mult *= np.where(weekend, 1.15, 1.0)
    mult *= np.where(df["sku"] == "cold_drink_2l", 1 + 0.7 * np.exp(-((month - 4.5) ** 2) / 3), 1)
    mult *= np.where(df["sku"] == "umbrella", 0.3 + df["rain_mm"].values / 8, 1)
    mult *= np.where(festival & df["sku"].isin(["atta_5kg", "cold_drink_2l", "instant_noodles"]), 1.4, 1)
    mult *= np.exp(0.04 * discount)

    mean = df["base_units"].values * df["store_size"].values * mult
    demand = rng.poisson(mean)
    stock = rng.poisson(mean * 1.15)          # stock on shelf (not shown to students)
    df["units_sold"] = np.minimum(demand, stock)
    df["stockout"] = (demand > stock).astype(int)   # true demand is censored!
    df["discount_pct"] = discount
    df["festival"] = festival.astype(int)
    df["price"] = (df["price"] * (1 - discount / 100)).round(1)
    return df[["store_id", "city", "store_size", "date", "sku", "price",
               "discount_pct", "festival", "rain_mm", "units_sold", "stockout"]]


# ----------------------------------------------------------------------------
# 3. D2C e-commerce: return-to-origin (RTO) prediction
# ----------------------------------------------------------------------------
def make_rto(n, rng):
    tier = rng.choice([1, 2, 3], n, p=[.35, .35, .30])
    cod_share = np.where(tier == 1, .45, np.where(tier == 2, .60, .75))
    payment = np.where(rng.random(n) < cod_share, "COD", "Prepaid")
    category = rng.choice(["fashion", "beauty", "electronics", "home_decor", "grocery"],
                          n, p=[.35, .20, .15, .15, .15])
    order_value = np.clip(rng.lognormal(np.log(700), 0.7, n), 99, 15000).round()
    discount = rng.choice([0, 10, 20, 30, 40, 50], n, p=[.25, .20, .20, .15, .10, .10])
    first = (rng.random(n) < 0.55).astype(int)
    past = np.where(first == 1, 0, rng.poisson(3, n) + 1)
    addr = np.clip(rng.beta(5, 2, n) - 0.1 * (tier == 3), 0, 1).round(2)
    dist = np.clip(rng.gamma(2, 150, n), 5, 2500).round()
    courier = rng.choice(["ShipFast", "QuickKart", "DesiDelivery", "IndiaExpress"], n)
    w = np.ones(24); w[20:24] = 3; w[8:12] = 2
    hour = rng.choice(np.arange(24), n, p=w / w.sum())
    odate = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")

    cod = (payment == "COD")
    logit = (1.3 * cod + 0.6 * first - 0.12 * np.minimum(past, 8)
             - 2.0 * (addr - 0.7) + 0.0004 * dist + 0.015 * discount
             + 0.5 * (category == "fashion") + 0.4 * (courier == "DesiDelivery")
             + 0.3 * ((hour >= 23) | (hour <= 4)) + 0.0002 * order_value * cod
             + rng.normal(0, 0.4, n))
    rto = draw_with_rate(logit, 0.20, rng)

    return pd.DataFrame({
        "order_id": [f"O{i:07d}" for i in range(n)],
        "order_date": odate.date, "city_tier": tier, "payment_mode": payment,
        "category": category, "order_value": order_value, "discount_pct": discount,
        "is_first_order": first, "past_orders": past, "address_quality": addr,
        "delivery_distance_km": dist, "courier": courier, "order_hour": hour,
        "forward_shipping_cost": (60 + dist * 0.05).round(), "rto": rto,
    })


# ----------------------------------------------------------------------------
# 4. Health insurance claims: fraud / leakage detection (imbalanced, ~4%)
# ----------------------------------------------------------------------------
def make_claims(n, rng):
    procs = {  # procedure: (typical claim Rs, typical days admitted)
        "cataract": (45000, 1), "appendectomy": (80000, 3), "knee_replacement": (250000, 5),
        "dengue": (60000, 5), "maternity": (90000, 3), "cardiac_stent": (300000, 4),
        "fracture": (100000, 4),
    }
    proc = rng.choice(list(procs), n)
    base_amt = pd.Series(proc).map({k: v[0] for k, v in procs.items()}).values
    base_days = pd.Series(proc).map({k: v[1] for k, v in procs.items()}).values

    age = np.clip(rng.normal(42, 14, n), 18, 80).astype(int)
    policy_age = rng.integers(1, 2000, n)
    hosp = rng.choice(["network_private", "network_corporate", "non_network_small", "government"],
                      n, p=[.35, .25, .30, .10])
    ratio = rng.lognormal(0, 0.35, n)
    amount = base_amt * ratio
    days = np.clip(rng.poisson(base_days), 0, 30)
    prior = rng.poisson(0.6, n)
    docs_ok = rng.random(n) > 0.10

    logit = (2.5 * (policy_age < 90) + 0.8 * (hosp == "non_network_small")
             + 0.5 * prior + 1.0 * (~docs_ok) + 1.5 * (ratio - 1)
             + rng.normal(0, 0.5, n))
    fraud = draw_with_rate(logit, 0.04, rng)
    amount = np.where(fraud == 1, amount * rng.lognormal(0.3, 0.15, n), amount)

    return pd.DataFrame({
        "claim_id": [f"C{i:06d}" for i in range(n)], "age": age,
        "policy_age_days": policy_age, "hospital_type": hosp, "procedure": proc,
        "claim_amount": amount.round(-2), "days_admitted": days,
        "prior_claims": prior, "docs_complete": docs_ok.astype(int), "is_fraud": fraud,
    })


# ----------------------------------------------------------------------------
# 5. Agri-tech: mandi price + arrivals forecasting (time series)
# ----------------------------------------------------------------------------
def make_mandi(rng, years=2):
    dates = pd.date_range("2024-01-01", periods=365 * years, freq="D")
    n = len(dates)
    doy, month = dates.dayofyear.values, dates.month.values
    monsoon = (month >= 6) & (month <= 9)
    rain = np.where(rng.random(n) < np.where(monsoon, .6, .03),
                    rng.exponential(np.where(monsoon, 30, 10)), 0).round(1)
    rain7 = pd.Series(rain).rolling(7, min_periods=1).sum().values

    commodities = {  # base Rs/quintal, seasonal amplitude, peak day-of-year, rain sensitivity
        "onion": dict(base=2200, amp=.25, peak=300, rs=.15),
        "tomato": dict(base=2000, amp=.35, peak=195, rs=.60),
        "potato": dict(base=1800, amp=.12, peak=220, rs=.04),
    }
    mandis = {"Lasalgaon": .95, "Vashi": 1.08, "Pune": 1.00, "Azadpur": 1.12}

    frames = []
    for com, p in commodities.items():
        for mandi, mf in mandis.items():
            e, noise = np.zeros(n), rng.normal(0, .02, n)
            for i in range(1, n):  # AR(1) noise = persistent price drift
                e[i] = .97 * e[i - 1] + noise[i]
            rel = 1 + p["amp"] * np.cos(2 * np.pi * (doy - p["peak"]) / 365) + e + p["rs"] * rain7 / 200
            price = np.round(p["base"] * mf * rel / 10) * 10
            arrivals = np.clip(500 * (1 - 1.2 * (rel - 1)) + rng.normal(0, 40, n), 50, None).round(1)
            frames.append(pd.DataFrame({"date": dates, "mandi": mandi, "commodity": com,
                                        "rain_mm": rain, "arrivals_tonnes": arrivals,
                                        "modal_price": price}))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop(df.sample(frac=0.03, random_state=2).index)           # mandi closed days
    typo = df.sample(frac=0.002, random_state=3).index                 # data-entry typos
    df.loc[typo, "modal_price"] *= 10
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 6. B2B SaaS for small businesses (GST billing): churn + unit economics
# ----------------------------------------------------------------------------
def make_saas(n, rng):
    plan = rng.choice(["Starter", "Growth", "Pro"], n, p=[.50, .35, .15])
    mrr = pd.Series(plan).map({"Starter": 999, "Growth": 2999, "Pro": 7999}).values
    seats = rng.poisson(pd.Series(plan).map({"Starter": 3, "Growth": 10, "Pro": 30}).values) + 1
    channel = rng.choice(["organic", "google_ads", "whatsapp_referral", "field_sales", "partner_ca"],
                         n, p=[.20, .30, .20, .15, .15])
    cac_base = pd.Series(channel).map({"organic": 500, "google_ads": 3500, "whatsapp_referral": 800,
                                       "field_sales": 6000, "partner_ca": 1500}).values
    tenure = rng.integers(1, 36, n)
    usage = rng.beta(2, 2, n)                      # hidden engagement (not given to students)
    logins = rng.poisson(2 + 25 * usage)
    adoption = np.clip(usage * 0.7 + rng.beta(2, 5, n) * 0.3, 0, 1).round(2)
    tickets = rng.poisson(1.5, n)
    pay_fail = rng.poisson(0.9 * (1 - usage))
    gst = (rng.random(n) < 0.4 + 0.4 * usage).astype(int)
    nps = np.clip(np.round(rng.normal(6 + 4 * usage - 0.3 * tickets, 2)), 0, 10)

    logit = (-3.5 * usage + 0.5 * pay_fail + 0.2 * tickets - 0.8 * gst
             + 0.5 * (plan == "Starter") - 0.03 * tenure + rng.normal(0, 0.4, n))
    churn = draw_with_rate(logit, 0.18, rng)

    return pd.DataFrame({
        "customer_id": [f"U{i:05d}" for i in range(n)], "plan": plan, "mrr": mrr,
        "acquisition_channel": channel, "cac": (cac_base * rng.lognormal(0, .2, n)).round(-1),
        "tenure_months": tenure, "seats": seats, "logins_30d": logins,
        "feature_adoption": adoption, "support_tickets": tickets,
        "payment_failures": pay_fail, "uses_gst_module": gst, "nps": nps,
        "churned_90d": churn,
    })


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="output folder")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply dataset sizes")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    s = a.scale

    jobs = {
        "credit_applications.csv": (make_credit(int(5000 * s), rng), "defaulted"),
        "kirana_demand.csv": (make_demand(rng, n_stores=max(5, int(20 * s))), "stockout"),
        "rto_orders.csv": (make_rto(int(20000 * s), rng), "rto"),
        "health_claims.csv": (make_claims(int(8000 * s), rng), "is_fraud"),
        "mandi_prices.csv": (make_mandi(rng), None),
        "saas_customers.csv": (make_saas(int(3000 * s), rng), "churned_90d"),
    }
    for fname, (df, target) in jobs.items():
        df.to_csv(os.path.join(a.out, fname), index=False)
        extra = f" | {target} rate = {df[target].mean():.1%}" if target else ""
        print(f"{fname:26s} rows = {len(df):>7,}{extra}")
    print(f"\nSaved to ./{a.out}/")


if __name__ == "__main__":
    main()
