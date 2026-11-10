"""
forecast_and_order_baseline.py
------------------------------
Baseline for the ordering + inventory part of the problem.

  Step 1  Bill-of-materials explosion: POS lines x recipes -> ingredient usage per day
  Step 2  ML forecast of next-day usage vs a naive same-weekday baseline (WAPE)
  Step 3  Replay ordering policies in the inventory simulator and compare in rupees:
            - gut-feel manager (what the historical data reflects)
            - forecast-based order-up-to policy (statistical baseline)
            - YOUR policy: plug the ML forecast (or an LLM-assisted rule) in below

    python generate_pos_data.py --out data_pos
    python forecast_and_order_baseline.py --data data_pos
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from generate_pos_data import (build_world, kpis, policy_forecast, policy_gut_feel,
                               simulate_inventory)


def usage_from_pos(data):
    """Step 1: recipe explosion. This is the core 'POS -> inventory' link of the whole system."""
    pos = pd.read_csv(os.path.join(data, "pos_order_lines.csv"), parse_dates=["timestamp"])
    rec = pd.read_csv(os.path.join(data, "recipes.csv"))
    pos["date"] = pos["timestamp"].dt.normalize()
    dish_day = pos.groupby(["outlet_id", "date", "dish_id"], as_index=False)["qty"].sum()
    u = dish_day.merge(rec, on="dish_id")
    u["usage"] = u["qty"] * u["qty_per_portion"]
    return u.groupby(["outlet_id", "date", "ingredient_id"], as_index=False)["usage"].sum()


def make_features(usage, cal):
    full = (usage.set_index(["outlet_id", "ingredient_id", "date"])["usage"].unstack("date", fill_value=0)
                 .stack().rename("usage").reset_index())
    full = full.sort_values(["outlet_id", "ingredient_id", "date"])
    g = full.groupby(["outlet_id", "ingredient_id"])["usage"]
    full["lag1"], full["lag7"], full["lag14"] = g.shift(1), g.shift(7), g.shift(14)
    full["mean7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    full["mean28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
    full["dow"], full["month"] = full["date"].dt.dayofweek, full["date"].dt.month
    cal = cal.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal["festival"] = cal["festival"].fillna("")
    cal["is_festival"] = (cal["festival"] != "").astype(int)
    cal["is_navratri"] = (cal["festival"] == "Navratri").astype(int)
    city = pd.read_csv(os.path.join(DATA, "outlets.csv"))[["outlet_id", "city"]]
    full = full.merge(city, on="outlet_id").merge(cal, on=["date", "city"], how="left")
    full["ing_code"] = full["ingredient_id"].str[3:].astype(int)
    return full.dropna(subset=["lag14", "mean28"])


def wape(y, p):
    return 100 * np.abs(y - p).sum() / np.abs(y).sum()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_pos")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--days", type=int, default=180)
    a = ap.parse_args()
    DATA = a.data

    # ---- Steps 1 and 2: usage + ML forecast ---------------------------------
    usage = usage_from_pos(DATA)
    cal = pd.read_csv(os.path.join(DATA, "calendar_context.csv"))
    df = make_features(usage, cal)
    feats = ["lag1", "lag7", "lag14", "mean7", "mean28", "dow", "month", "rain_mm", "is_festival", "is_navratri", "ing_code"]
    cut = df["date"].sort_values().iloc[int(len(df) * .75)]          # time-based split, never random
    tr, te = df[df["date"] < cut], df[df["date"] >= cut]
    # predict the ratio to the 28-day mean so one model handles kg of onion and kg of cashew
    model = HistGradientBoostingRegressor(max_iter=250, learning_rate=.06, random_state=0)
    model.fit(tr[feats], tr["usage"] / tr["mean28"].clip(lower=1e-6))
    pred = model.predict(te[feats]) * te["mean28"]
    print("Next-day usage forecast (test period, lower WAPE is better)")
    print(f"  naive same-weekday-last-week : {wape(te['usage'], te['lag7']):5.1f}%")
    print(f"  28-day mean                  : {wape(te['usage'], te['mean28']):5.1f}%")
    print(f"  gradient boosting            : {wape(te['usage'], pred):5.1f}%\n")

    # ---- Step 3: replay ordering policies in rupees -------------------------
    W = build_world(a.seed, "2025-07-01", a.days)

    def my_policy(ctx):
        """TODO (teams): use the ML forecast, supplier delay risk, price signals, or an
        LLM-assisted rule (for example 'reduce chicken orders during Navratri'). Start by
        wrapping policy_forecast with a different z (service level) or forecast_fn."""
        return policy_forecast(ctx, z=1.0)

    rows = {}
    for name, pol in [("gut-feel manager", policy_gut_feel), ("forecast order-up-to", policy_forecast),
                      ("my_policy (z=1.0)", my_policy)]:
        A, _ = simulate_inventory(W, pol)
        rows[name] = kpis(W, A)
    print("Ordering policy replay (same demand, same supplier delays)")
    print(pd.DataFrame(rows).T.to_string())
