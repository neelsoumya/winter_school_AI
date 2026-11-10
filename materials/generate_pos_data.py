#!/usr/bin/env python3
"""
generate_pos_data.py
--------------------
Synthetic world for the "Smart Kitchen Backend" hackathon problem:
a restaurant POS connected to a backend that tracks inventory, orders raw
materials (vegetables, dairy, meat, dry goods), and handles compliance + reporting.

    python generate_pos_data.py                       # writes ./data_pos
    python generate_pos_data.py --days 120 --seed 7 --out my_data

What you get (all fictional):
  Structured : outlets, menu, recipes (bill of materials), ingredients, suppliers,
               supplier_prices, calendar_context, pos_order_lines,
               purchase_orders, daily_inventory, stock_counts, purchase_invoices,
               temperature_logs, licence_registry, staff_compliance
  Text (LLM) : supplier_messages.jsonl (WhatsApp, English + Hinglish),
               delivery_notes_ocr.jsonl (noisy invoice text),
               kitchen_voice_notes.jsonl (stock-count transcripts)
  Rules      : compliance_rules_simplified.md (teaching rules, NOT legal advice)

Ground-truth columns/labels are prefixed `label_` (or live under "label" in jsonl).
The historical ordering was done by a "gut-feel manager" policy, so it is wasteful.
Your job is to beat it.

This module is also importable: build_world(), simulate_inventory(), kpis(),
policy_gut_feel(), policy_forecast() are used by forecast_and_order_baseline.py.
"""
import argparse
import json
import os
import string

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Master data
# ----------------------------------------------------------------------------
OUTLETS = [  # id, name, city, scale (relative volume), seats
    ("O1", "Andheri West", "Mumbai", 1.00, 60),
    ("O2", "Koregaon Park", "Pune", 0.80, 45),
    ("O3", "Dharampeth", "Nagpur", 0.65, 40),
]

SUPPLIERS = [  # id, name, category focus, on-time reliability, payment terms (days)
    ("SUP01", "Vashi Fresh Veg Traders", "veg", 0.88, 0),
    ("SUP02", "Green Basket Agro", "veg", 0.95, 7),
    ("SUP03", "Sahyadri Dairy Co-op", "dairy", 0.97, 7),
    ("SUP04", "Metro Meats and Poultry", "meat", 0.90, 3),
    ("SUP05", "Sunrise Grains and Provisions", "dry", 0.96, 15),
    ("SUP06", "Spice Bazaar Traders", "spice", 0.93, 15),
    ("SUP07", "ColdChain Foods Pvt Ltd", "dairy+meat", 0.92, 10),
]
CAT_SUPPLIERS = {"veg": ("SUP01", "SUP02"), "dairy": ("SUP03", "SUP07"),
                 "meat": ("SUP04", "SUP07"), "dry": ("SUP05", "SUP06"),
                 "spice": ("SUP06", "SUP05")}

# id, name, unit, category, storage, shelf_life_days, base_price(Rs/unit), price_vol,
# lead_time_days, min_order_qty, gst_rate_pct (ILLUSTRATIVE - verify current GST rules)
ING_ROWS = [
    ("ING01", "tomato", "kg", "veg", "cold", 6, 30, .35, 1, 5, 0),
    ("ING02", "onion", "kg", "veg", "ambient", 14, 30, .30, 1, 10, 0),
    ("ING03", "potato", "kg", "veg", "ambient", 21, 28, .10, 1, 10, 0),
    ("ING04", "green_chilli", "kg", "veg", "cold", 7, 80, .30, 1, 2, 0),
    ("ING05", "coriander", "kg", "veg", "cold", 3, 100, .40, 1, 1, 0),
    ("ING06", "spinach", "kg", "veg", "cold", 3, 40, .30, 1, 3, 0),
    ("ING07", "cauliflower", "kg", "veg", "cold", 7, 45, .30, 1, 5, 0),
    ("ING08", "ginger_garlic_paste", "kg", "veg", "cold", 10, 160, .15, 2, 2, 5),
    ("ING09", "paneer", "kg", "dairy", "cold", 7, 340, .05, 1, 3, 5),
    ("ING10", "milk", "l", "dairy", "cold", 3, 58, .03, 1, 10, 0),
    ("ING11", "curd", "kg", "dairy", "cold", 5, 70, .03, 1, 5, 0),
    ("ING12", "butter", "kg", "dairy", "cold", 60, 520, .05, 2, 2, 12),
    ("ING13", "cooking_cream", "l", "dairy", "cold", 30, 250, .05, 2, 2, 12),
    ("ING14", "chicken", "kg", "meat", "cold", 3, 210, .12, 1, 5, 0),
    ("ING15", "eggs", "piece", "meat", "ambient", 14, 7, .10, 1, 60, 0),
    ("ING16", "basmati_rice", "kg", "dry", "ambient", 365, 110, .05, 2, 25, 5),
    ("ING17", "atta", "kg", "dry", "ambient", 180, 42, .03, 2, 25, 0),
    ("ING18", "toor_dal", "kg", "dry", "ambient", 365, 150, .08, 2, 25, 0),
    ("ING19", "cooking_oil", "l", "dry", "ambient", 270, 135, .10, 2, 15, 5),
    ("ING20", "ghee", "kg", "dairy", "ambient", 180, 580, .04, 2, 2, 12),
    ("ING21", "cashew", "kg", "dry", "ambient", 180, 900, .10, 3, 2, 5),
    ("ING22", "red_chilli_powder", "kg", "spice", "ambient", 365, 320, .08, 3, 2, 5),
    ("ING23", "garam_masala", "kg", "spice", "ambient", 365, 600, .06, 3, 1, 5),
]
ING_COLS = ["ingredient_id", "name", "unit", "category", "storage", "shelf_life_days",
            "base_price", "price_volatility", "lead_time_days", "min_order_qty", "gst_rate_pct"]

# id, name, course, price, is_veg, popularity, recipe {ingredient name: qty per portion}
DISHES = [
    ("D01", "Paneer Butter Masala", "main", 320, 1, 1.6, {"paneer": .15, "butter": .02, "cooking_cream": .05, "tomato": .20, "onion": .08, "ginger_garlic_paste": .01, "garam_masala": .003, "red_chilli_powder": .003, "cooking_oil": .02, "coriander": .005}),
    ("D02", "Palak Paneer", "main", 300, 1, 1.0, {"paneer": .15, "spinach": .30, "onion": .08, "tomato": .10, "ginger_garlic_paste": .01, "cooking_cream": .03, "cooking_oil": .02}),
    ("D03", "Butter Chicken", "main", 380, 0, 1.8, {"chicken": .25, "butter": .03, "cooking_cream": .06, "tomato": .20, "onion": .08, "ginger_garlic_paste": .015, "garam_masala": .004, "red_chilli_powder": .003, "coriander": .005}),
    ("D04", "Chicken Biryani", "main", 350, 0, 1.7, {"chicken": .20, "basmati_rice": .15, "onion": .10, "curd": .05, "ghee": .02, "green_chilli": .005, "garam_masala": .003, "coriander": .01}),
    ("D05", "Veg Biryani", "main", 290, 1, 1.0, {"basmati_rice": .15, "cauliflower": .08, "potato": .05, "onion": .10, "curd": .05, "ghee": .02, "green_chilli": .005, "coriander": .01}),
    ("D06", "Dal Tadka", "main", 220, 1, 1.4, {"toor_dal": .08, "onion": .05, "tomato": .08, "ghee": .01, "green_chilli": .005, "coriander": .005, "red_chilli_powder": .002}),
    ("D07", "Aloo Gobi", "main", 240, 1, 0.8, {"potato": .15, "cauliflower": .15, "onion": .05, "tomato": .05, "cooking_oil": .03, "coriander": .005}),
    ("D08", "Steamed Rice", "side", 120, 1, 1.2, {"basmati_rice": .15}),
    ("D09", "Butter Naan", "bread", 60, 1, 3.0, {"atta": .10, "butter": .01, "curd": .02}),
    ("D10", "Tandoori Roti", "bread", 35, 1, 2.0, {"atta": .08}),
    ("D11", "Paneer Tikka", "starter", 340, 1, 1.1, {"paneer": .18, "curd": .06, "onion": .05, "ginger_garlic_paste": .01, "red_chilli_powder": .002}),
    ("D12", "Chicken Tikka", "starter", 360, 0, 1.1, {"chicken": .22, "curd": .06, "ginger_garlic_paste": .01, "red_chilli_powder": .002}),
    ("D13", "Masala Omelette", "snack", 120, 0, 0.6, {"eggs": 2, "onion": .03, "green_chilli": .005, "cooking_oil": .01}),
    ("D14", "Egg Curry", "main", 210, 0, 0.6, {"eggs": 2, "onion": .08, "tomato": .10, "cooking_oil": .02}),
    ("D15", "Masala Chai", "beverage", 50, 1, 2.2, {"milk": .12}),
    ("D16", "Sweet Lassi", "beverage", 90, 1, 1.2, {"curd": .20}),
    ("D17", "Kheer", "dessert", 140, 1, 0.7, {"milk": .25, "basmati_rice": .03, "cashew": .01}),
]

# approximate 2025 festival windows
FESTIVALS = {
    "Ganesh Chaturthi": ("2025-08-27", "2025-09-06"),
    "Navratri": ("2025-09-22", "2025-10-01"),
    "Diwali": ("2025-10-18", "2025-10-23"),
    "Christmas-NewYear": ("2025-12-24", "2026-01-01"),
}
FEST_DEMAND_MULT = {"Ganesh Chaturthi": 1.10, "Navratri": 1.00, "Diwali": 1.25, "Christmas-NewYear": 1.20}

STOCKOUT_PENALTY = 4.0   # lost gross margin per rupee of ingredient not available


def sigmoid_noise_ar(rng, n, rho, sigma):
    e, z = np.zeros(n), rng.normal(0, sigma, n)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + z[i]
    return e


# ----------------------------------------------------------------------------
# World building
# ----------------------------------------------------------------------------
def make_calendar(rng, days):
    month = days.month.values
    monsoon = (month >= 6) & (month <= 9)
    fest = np.array([""] * len(days), dtype=object)
    for name, (s, e) in FESTIVALS.items():
        fest[(days >= s) & (days <= e)] = name
    rows = []
    for _, _, city, _, _ in OUTLETS:
        wet = rng.random(len(days)) < np.where(monsoon, .6, .03)
        mm = np.where(wet, rng.exponential(np.where(monsoon, 25, 8)), 0).round(1)
        rows.append(pd.DataFrame({"date": days, "city": city, "rain_mm": mm, "festival": fest}))
    return pd.concat(rows, ignore_index=True)


def make_pos(rng, days, cal):
    rain = {c: g.set_index("date")["rain_mm"] for c, g in cal.groupby("city")}
    fest = cal.drop_duplicates("date").set_index("date")["festival"]
    pop = np.array([d[5] for d in DISHES])
    nonveg = np.array([d[4] == 0 for d in DISHES])
    price = np.array([d[3] for d in DISHES])
    dow_mult = np.array([.90, .85, .90, .95, 1.20, 1.35, 1.30])   # Mon..Sun
    frames, oid = [], 0
    for outlet_id, _, city, scale, _ in OUTLETS:
        for t, day in enumerate(days):
            f = fest[day]
            lam = (170 * scale * dow_mult[day.dayofweek] * FEST_DEMAND_MULT.get(f, 1.0)
                   * (1 - min(.3, rain[city][day] / 120)) * (1 + 0.0006 * t))
            n = rng.poisson(lam)
            k = 1 + np.minimum(rng.poisson(1.3, n), 5)
            order_idx = np.repeat(np.arange(n), k)
            m = len(order_idx)
            w = pop * np.where(nonveg & (f == "Navratri"), 0.5, 1.0)
            dish_idx = rng.choice(len(DISHES), m, p=w / w.sum())
            qty = 1 + (rng.random(m) < 0.2)
            hour = np.where(rng.random(n) < 0.4, rng.normal(13.2, .9, n), rng.normal(20.5, 1.2, n)).clip(11, 23.5)
            ts = day + pd.to_timedelta((hour * 60).astype(int)[order_idx], unit="m")
            ch = rng.choice(["dine_in", "takeaway", "delivery"], n, p=[.55, .15, .30])
            disc = np.where((ch == "delivery") & (rng.random(n) < .3), 10, rng.choice([0, 5], n, p=[.9, .1]))
            frames.append(pd.DataFrame({
                "order_id": [f"{outlet_id}-{oid + j:07d}" for j in order_idx], "outlet_id": outlet_id,
                "timestamp": ts, "channel": ch[order_idx], "dish_id": [DISHES[j][0] for j in dish_idx],
                "qty": qty, "unit_price": price[dish_idx], "discount_pct": disc[order_idx]}))
            oid += n
    return pd.concat(frames, ignore_index=True)


def build_world(seed=42, start="2025-07-01", n_days=180, warm=14):
    """Everything the simulator needs. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=warm), periods=n_days + warm)
    ing = pd.DataFrame(ING_ROWS, columns=ING_COLS)
    ing_names = list(ing["name"])
    D, I, O = len(days), len(ing), len(OUTLETS)

    cal = make_calendar(rng, days)
    pos = make_pos(rng, days, cal)
    pos["date"] = pos["timestamp"].dt.normalize()

    # Bill-of-materials matrix R[dish, ingredient]
    R = np.zeros((len(DISHES), I))
    for d, dish in enumerate(DISHES):
        for name, q in dish[6].items():
            R[d, ing_names.index(name)] = q
    dish_ids = [d[0] for d in DISHES]
    oids = [o[0] for o in OUTLETS]
    counts = (pos.pivot_table(index=["outlet_id", "date"], columns="dish_id", values="qty",
                              aggfunc="sum", fill_value=0)
                 .reindex(pd.MultiIndex.from_product([oids, days]), fill_value=0)[dish_ids])
    C = counts.values.reshape(O, D, len(DISHES))
    Uth = np.einsum("odk,ki->odi", C, R)             # theoretical usage from POS x recipes

    # actual usage = theoretical x portion over-use + planted shrinkage
    Uact = Uth * rng.lognormal(np.log(1.04), .04, Uth.shape)
    shrink = {("O2", "chicken"): 1.12, ("O2", "cashew"): 1.18, ("O3", "cooking_oil"): 1.09}
    for (o, name), f in shrink.items():
        Uact[oids.index(o), :, ing_names.index(name)] *= f

    # market price index per ingredient/day (seasonality + AR noise + rain)
    doy = days.dayofyear.values
    rain7 = cal[cal.city == OUTLETS[0][2]].set_index("date")["rain_mm"].rolling(7, min_periods=1).sum().values
    params = {"tomato": (.30, 200, .5), "onion": (.25, 305, .15), "green_chilli": (.20, 200, .4),
              "coriander": (.20, 200, .6), "spinach": (.15, 200, .4)}
    Pm = np.zeros((D, I))
    for i, r in ing.iterrows():
        amp, peak, rs = params.get(r["name"], (r["price_volatility"] * .5, 250, .3 if r["category"] == "veg" else 0))
        e = sigmoid_noise_ar(rng, D, .9, r["price_volatility"] * .08)
        Pm[:, i] = r["base_price"] * (1 + amp * np.cos(2 * np.pi * (doy - peak) / 365) + e + rs * rain7 / 300)
    prim = np.array([CAT_SUPPLIERS[c][0] for c in ing["category"]])
    alt = np.array([CAT_SUPPLIERS[c][1] for c in ing["category"]])
    f_p, f_a = rng.uniform(.97, 1.03, I), rng.uniform(.93, 1.10, I)
    P = Pm * f_p                                      # primary-supplier price paid
    AP = Pm * f_a * (1 + rng.normal(0, .02, Pm.shape))

    rel_map = {s[0]: s[3] for s in SUPPLIERS}
    rel = np.array([rel_map[s] for s in prim])
    delay = np.where(rng.random((O, D, I)) > rel, rng.integers(1, 3, (O, D, I)), 0)
    short = np.where(rng.random((O, D, I)) < .10, rng.uniform(.8, .95, (O, D, I)), 1.0)

    return dict(rng=rng, days=days, warm=warm, ing=ing, cal=cal, pos=pos, R=R, Uth=Uth, Uact=Uact,
                P=P, AP=AP, prim=prim, alt=alt, delay=delay, short=short)


# ----------------------------------------------------------------------------
# Ordering policies (policy(ctx) -> quantity to order today)
# ----------------------------------------------------------------------------
def order_multiple(q, moq):
    return 0.0 if q <= 1e-9 else float(moq * np.ceil(q / moq))


def forecast_daily(hist, k):
    """Naive-seasonal forecast of usage k days ahead: mean of the same weekday in past weeks."""
    n = len(hist)
    vals = [hist[n - 1 + k - 7 * m] for m in range(1, 5) if 0 <= n - 1 + k - 7 * m <= n - 1]
    return float(np.mean(vals)) if vals else float(np.mean(hist[-7:]))


def policy_gut_feel(ctx):
    """What the human manager did historically: eyeball last week, pad it, ignore what's already on order."""
    hist = ctx["hist"]
    if ctx["perishable"]:
        target = ctx["rng"].normal(1.9, .45) * hist[-7:].mean() * (ctx["lead"] + 1)
    else:
        if ctx["dow"] != 0:
            return 0.0
        target = 1.6 * hist[-28:].mean() * (7 + ctx["lead"])
    return target - ctx["on_hand"]


def policy_forecast(ctx, forecast_fn=forecast_daily, z=1.5):
    """Order-up-to policy: cover lead time + review period + safety stock, capped by shelf life.
    Plug your own ML forecast into `forecast_fn(hist, k)` (or change the signature to use ctx)."""
    hist = ctx["hist"]
    review = 1 if ctx["perishable"] else 7
    if review == 7 and ctx["dow"] != 0:
        return 0.0
    H = ctx["lead"] + review
    mu = sum(forecast_fn(hist, k) for k in range(1, H + 1))
    sigma = float(np.std(hist[-28:]))
    target = mu + z * sigma * np.sqrt(H)
    if ctx["perishable"]:
        target = min(target, sum(forecast_fn(hist, k) for k in range(1, int(ctx["shelf"]) + 1)))
    return target - ctx["on_hand"] - ctx["pipeline"]


# ----------------------------------------------------------------------------
# Inventory simulator (FIFO batches, shelf life, supplier delays, short deliveries)
# ----------------------------------------------------------------------------
def simulate_inventory(world, policy, seed=0):
    ing, U, P, days = world["ing"], world["Uact"], world["P"], world["days"]
    O, D, I = U.shape
    shelf, lead, moq = ing["shelf_life_days"].values, ing["lead_time_days"].values, ing["min_order_qty"].values
    prng = np.random.default_rng(seed)
    keys = ["open", "received", "rec_cost", "expired", "waste_cost", "stockout", "consumed", "close"]
    A = {k: np.zeros((O, D, I)) for k in keys}
    pos_list = []
    for o in range(O):
        for i in range(I):
            on_hand = float(U[o, :14, i].mean() * 2.5)
            batches, pipeline = [[on_hand, -1, P[0, i]]], []
            for d in range(D):
                A["open"][o, d, i] = on_hand
                pending = []
                for e in pipeline:
                    if e["arrive"] <= d:
                        batches.append([e["qty"], d, e["cost"]])
                        A["received"][o, d, i] += e["qty"]
                        A["rec_cost"][o, d, i] += e["qty"] * e["cost"]
                        e["po"]["received_day"], e["po"]["received_qty"] = d, e["qty"]
                    else:
                        pending.append(e)
                pipeline = pending
                alive = []
                for b in batches:
                    if d - b[1] > shelf[i]:
                        A["expired"][o, d, i] += b[0]
                        A["waste_cost"][o, d, i] += b[0] * b[2]
                    else:
                        alive.append(b)
                batches, need = alive, U[o, d, i]
                for b in batches:
                    if need <= 1e-9:
                        break
                    take = min(b[0], need)
                    b[0] -= take
                    need -= take
                    A["consumed"][o, d, i] += take
                batches = [b for b in batches if b[0] > 1e-9]
                A["stockout"][o, d, i] = max(need, 0.0)
                on_hand = float(sum(b[0] for b in batches))
                A["close"][o, d, i] = on_hand

                ctx = dict(outlet=o, ing=i, day=d, dow=days[d].dayofweek,
                           hist=U[o, max(0, d - 27):d + 1, i], on_hand=on_hand,
                           pipeline=sum(e["qty"] for e in pipeline), lead=int(lead[i]),
                           shelf=int(shelf[i]), perishable=bool(shelf[i] <= 14),
                           moq=float(moq[i]), price=P[d, i], rng=prng)
                q = order_multiple(policy(ctx), moq[i])
                if q > 0:
                    po = dict(outlet_idx=o, ing_idx=i, order_day=d, expected_day=d + int(lead[i]),
                              received_day=None, ordered_qty=q, received_qty=0.0, unit_price=P[d, i])
                    pos_list.append(po)
                    pipeline.append(dict(arrive=d + int(lead[i]) + int(world["delay"][o, d, i]),
                                         qty=q * world["short"][o, d, i], cost=P[d, i], po=po))
    return A, pos_list


def kpis(world, A):
    w = world["warm"]
    price = world["P"][w:][None, :, :]
    spend, waste = A["rec_cost"][:, w:].sum(), A["waste_cost"][:, w:].sum()
    penalty = (A["stockout"][:, w:] * price).sum() * STOCKOUT_PENALTY
    return {"purchase_spend_Rs": int(round(spend)), "wastage_Rs": int(round(waste)),
            "wastage_pct_of_spend": float(round(100 * waste / spend, 1)),
            "stockout_penalty_Rs": int(round(penalty)),
            "total_cost_Rs (wastage+stockout)": int(round(waste + penalty)),
            "service_level_pct": float(round(100 * (A["stockout"][:, w:] <= 1e-9).mean(), 1))}


# ----------------------------------------------------------------------------
# Tables from simulation results
# ----------------------------------------------------------------------------
def inventory_tables(world, A, pos_list, rng):
    w, days, ing = world["warm"], world["days"], world["ing"]
    oids, iids = [o[0] for o in OUTLETS], list(ing["ingredient_id"])
    O, D, I = A["close"].shape
    out_days = days[w:]
    Dn = D - w
    sl = lambda k: A[k][:, w:, :].ravel()
    inv = pd.DataFrame({
        "date": np.tile(np.repeat(out_days, I), O), "outlet_id": np.repeat(oids, Dn * I),
        "ingredient_id": np.tile(iids, O * Dn), "opening_qty": sl("open").round(2),
        "received_qty": sl("received").round(2), "theoretical_usage": world["Uth"][:, w:, :].ravel().round(3),
        "actual_usage": sl("consumed").round(3), "expired_qty": sl("expired").round(2),
        "stockout_qty": sl("stockout").round(3), "closing_qty": sl("close").round(2),
        "unit_cost": np.tile(world["P"][w:], (O, 1, 1)).ravel().round(2)})

    # purchase orders
    rows = []
    for k, p in enumerate(pos_list):
        if p["order_day"] < w:
            continue
        rd = p["received_day"]
        rows.append({"outlet_id": oids[p["outlet_idx"]], "ingredient_id": iids[p["ing_idx"]],
                     "supplier_id": world["prim"][p["ing_idx"]], "order_date": days[p["order_day"]],
                     "expected_date": days[0] + pd.Timedelta(days=p["expected_day"]),
                     "received_date": days[rd] if rd is not None else pd.NaT,
                     "ordered_qty": p["ordered_qty"], "received_qty": round(p["received_qty"], 2),
                     "unit_price": round(p["unit_price"], 2)})
    po = pd.DataFrame(rows).sort_values(["order_date", "outlet_id"]).reset_index(drop=True)
    po.insert(0, "po_id", [f"PO{i:06d}" for i in range(len(po))])

    # weekly physical stock counts (Mondays): expected vs counted -> shrinkage signal
    cnt = []
    for o in range(O):
        for i in range(I):
            expected = A["close"][o, w - 1, i]
            for d in range(w, D):
                expected += A["received"][o, d, i] - world["Uth"][o, d, i] - A["expired"][o, d, i]
                if days[d].dayofweek == 0:
                    counted = A["close"][o, d, i] * (1 + rng.normal(0, .02))
                    wk_use = world["Uth"][o, d - 6:d + 1, i].sum()
                    cnt.append((days[d], oids[o], iids[i], round(expected, 2), round(counted, 2),
                                round(counted - expected, 2), round(100 * (counted - expected) / max(wk_use, 1e-6), 1)))
                    expected = counted
    counts = pd.DataFrame(cnt, columns=["date", "outlet_id", "ingredient_id", "expected_qty",
                                        "counted_qty", "variance_qty", "variance_pct_of_weekly_use"])
    return inv, po, counts


def make_suppliers(rng, end):
    rows = []
    for sid, name, cat, rel, terms in SUPPLIERS:
        gstin = "" if sid == "SUP01" else "27" + "".join(rng.choice(list(string.ascii_uppercase), 5)) + \
            "".join(rng.choice(list("0123456789"), 4)) + rng.choice(list(string.ascii_uppercase)) + "1Z" + str(rng.integers(0, 9))
        lic = "" if sid == "SUP01" else str(rng.integers(10**13, 10**14))
        exp = pd.NaT if sid == "SUP01" else end + pd.Timedelta(days=int(-30 if sid == "SUP06" else rng.integers(60, 600)))
        rows.append((sid, name, cat, rel, terms, gstin, lic, exp))
    return pd.DataFrame(rows, columns=["supplier_id", "name", "category", "on_time_reliability",
                                       "payment_terms_days", "gstin", "food_licence_no", "food_licence_expiry"])


def make_invoices(rng, po, ing, supp):
    gst = ing.set_index("ingredient_id")["gst_rate_pct"]
    d = po[po.received_date.notna()].copy()
    d["taxable"] = d.received_qty * d.unit_price
    d["gst"] = d.taxable * d.ingredient_id.map(gst) / 100
    g = d.groupby(["outlet_id", "supplier_id", "received_date"], as_index=False)[["taxable", "gst"]].sum()
    g = g.rename(columns={"received_date": "invoice_date", "taxable": "taxable_value", "gst": "gst_amount"})
    g["total_amount"] = g.taxable_value + g.gst_amount
    g["supplier_gstin_on_invoice"] = g.supplier_id.map(supp.set_index("supplier_id")["gstin"])
    g["label_error_type"] = "none"
    r = rng.random(len(g))
    bad_total = r < .02
    g.loc[bad_total, "total_amount"] += rng.choice([-1, 1], bad_total.sum()) * rng.uniform(50, 500, bad_total.sum())
    g.loc[bad_total, "label_error_type"] = "total_mismatch"
    miss = (r >= .02) & (r < .03) & (g.supplier_gstin_on_invoice != "")
    g.loc[miss, "supplier_gstin_on_invoice"] = ""
    g.loc[miss, "label_error_type"] = "missing_gstin"
    g["label_has_error"] = (g.label_error_type != "none").astype(int)
    g[["taxable_value", "gst_amount", "total_amount"]] = g[["taxable_value", "gst_amount", "total_amount"]].round(2)
    g.insert(0, "invoice_id", [f"INV-{i:06d}" for i in range(len(g))])
    return g


# ----------------------------------------------------------------------------
# Compliance data
# ----------------------------------------------------------------------------
def make_temp_logs(rng, out_days):
    units = {"cold_room": (3.5, .7, 5.0), "freezer": (-19.0, 1.0, -15.0), "display_chiller": (4.0, .8, 5.0)}
    hours = np.array([2, 6, 10, 14, 18, 22])
    ts = pd.DatetimeIndex(np.repeat(out_days, 6)) + pd.to_timedelta(np.tile(hours, len(out_days)), unit="h")
    n, frames = len(ts), []
    for outlet, *_ in OUTLETS:
        for unit, (mu, sd, lim) in units.items():
            t = mu + rng.normal(0, sd, n) + np.where(np.isin(np.tile(hours, len(out_days)), [14, 22]), .5, 0)
            bump = np.zeros(n)
            for _ in range(5):                                   # random door-left-open episodes
                s, L = rng.integers(0, n - 3), rng.integers(1, 4)
                bump[s:s + L] += rng.uniform(5, 9) if unit == "freezer" else rng.uniform(2.5, 5)
            if outlet == "O2" and unit == "cold_room":           # compressor failure: 12 readings (2 days)
                bump[n // 2:n // 2 + 12] += rng.uniform(4, 7)
            t = (t + bump).round(1)
            frames.append(pd.DataFrame({"timestamp": ts, "outlet_id": outlet, "unit": unit, "temp_c": t,
                                        "label_breach": (t > lim).astype(int)}))
    df = pd.concat(frames, ignore_index=True)
    return df.drop(df.sample(frac=.03, random_state=5).index).reset_index(drop=True)   # skipped logs


def make_licences_and_staff(rng, end):
    types = ["FSSAI_licence", "Shops_and_Establishment", "Fire_NOC", "Health_trade_licence"]
    override = {("O2", "Fire_NOC"): -12, ("O3", "FSSAI_licence"): 25, ("O1", "Health_trade_licence"): 40}
    lic = []
    for o, *_ in OUTLETS:
        for t in types:
            off = override.get((o, t), int(rng.integers(90, 540)))
            num = str(rng.integers(10**13, 10**14)) if t == "FSSAI_licence" else f"{t[:3].upper()}-{rng.integers(10**5, 10**6)}"
            lic.append((o, t, num, (end + pd.Timedelta(days=off)).date()))
    lic = pd.DataFrame(lic, columns=["outlet_id", "licence_type", "licence_number", "expiry_date"])
    roles = ["cook", "cook", "cook", "helper", "helper", "waiter", "waiter", "waiter", "cashier", "manager", "store_keeper", "dishwasher"]
    st = []
    for o, *_ in OUTLETS:
        for j, r in enumerate(roles):
            f = int(rng.integers(-40, 700)) if rng.random() < .85 else int(rng.integers(-60, -1))
            m = int(rng.integers(-30, 600)) if rng.random() < .9 else int(rng.integers(-60, -1))
            st.append((f"STF-{o}-{j:02d}", o, r, (end + pd.Timedelta(days=f)).date(), (end + pd.Timedelta(days=m)).date()))
    return lic, pd.DataFrame(st, columns=["staff_id", "outlet_id", "role", "food_safety_training_expiry", "medical_fitness_expiry"])


RULES_MD = """# Simplified compliance rules (TEACHING USE ONLY, fictional and NOT legal advice)

Real compliance depends on current FSSAI regulations, state and municipal rules and GST law.
Teams should read the actual sources; these rules exist so LLM and rule-engine pipelines have something to work with.

- **R01 Cold chain.** Cold rooms and display chillers must read 5 C or below; freezers -15 C or below (alarm) with a target of -18 C. A breach lasting 2 or more consecutive readings is an *incident*; high-risk food (dairy, meat) stored in that unit needs a discard review.
- **R02 Temperature logging.** Readings are required every 4 hours while the unit is in use. Two or more consecutive missing readings (8 hours) is a record-keeping gap.
- **R03 Licences.** Every outlet must hold valid FSSAI, Shops and Establishment, Fire NOC and Health/Trade licences. Expired = critical. Expiring within 30 days = renewal action required.
- **R04 Staff.** Food handlers need a valid food-safety training certificate and a valid medical fitness certificate.
- **R05 Suppliers.** A supplier must have a valid food licence on record before the outlet places an order, except registered-exempt informal traders, who must be flagged for manager approval.
- **R06 Purchase invoices.** Invoices must show supplier, invoice number, date and the supplier GSTIN where the supplier is GST-registered. Taxable value + GST must equal the total.
- **R07 Wastage review.** Expiry wastage above 8% of purchase value for an ingredient in a week must be reviewed by the manager.
- **R08 Stock variance.** Unexplained weekly variance above 7% of weekly usage for items costing more than Rs 300 per unit requires investigation.
- **R09 FIFO and labelling.** Perishables are issued first-in-first-out and labelled with a use-by date.
- **R10 Record retention.** Keep temperature logs, invoices and stock counts for at least 2 years.
"""


# ----------------------------------------------------------------------------
# Unstructured text for LLM tasks (with ground-truth labels)
# ----------------------------------------------------------------------------
HI = {"tomato": "tamatar", "onion": "pyaaz", "potato": "aloo", "green_chilli": "hari mirch", "coriander": "dhaniya",
      "spinach": "palak", "cauliflower": "gobi", "milk": "doodh", "curd": "dahi", "eggs": "anda",
      "cooking_oil": "tel", "basmati_rice": "chawal", "chicken": "murgi"}
NUM_WORDS = {1: ("one", "ek"), 2: ("two", "do"), 3: ("three", "teen"), 4: ("four", "chaar"), 5: ("five", "paanch"),
             6: ("six", "chhe"), 7: ("seven", "saat"), 8: ("eight", "aath"), 9: ("nine", "nau"), 10: ("ten", "das"), 12: ("twelve", "barah")}


def _nm(rng, name, hindi_p=.5):
    return HI[name] if name in HI and rng.random() < hindi_p else name.replace("_", " ")


def make_supplier_messages(rng, ing, n=400):
    out = []
    pool = ing[ing.category.isin(["veg", "dairy", "meat", "dry"])]
    for k in range(n):
        r = pool.iloc[rng.integers(len(pool))]
        name, unit = r["name"], r["unit"]
        nm = _nm(rng, name)
        intent = rng.choice(["price_quote", "delay", "shortage", "substitution", "dispatch"], p=[.3, .2, .2, .15, .15])
        h = int(rng.integers(5, 10))
        label = {"intent": intent, "ingredient": name}
        if intent == "price_quote":
            price = int(round(r["base_price"] * rng.uniform(.9, 1.25)))
            text = rng.choice([f"Sir kal {nm} ka rate {price} rupay {unit} rahega, delivery {h} baje tak",
                               f"Tomorrow's rate for {nm}: Rs {price}/{unit}. Will deliver by {h}am",
                               f"{nm} {price}/{unit} kal. {h} AM tak pahunch jayega"])
            label.update(price=price, unit=unit, delivery_by=f"{h:02d}:00")
        elif intent == "delay":
            hrs = int(rng.integers(1, 6))
            text = rng.choice([f"Sir truck late hai, {nm} kal {hrs} ghante late aayega",
                               f"{nm} delivery delayed by {hrs} hours due to traffic",
                               f"Bhai {nm} aane mein {hrs} ghanta lagega, barish ki wajah se"])
            label.update(delay_hours=hrs)
        elif intent == "shortage":
            q = int(max(1, round(rng.uniform(.3, .8) * (60 if unit != "piece" else 300))))
            text = rng.choice([f"{nm} ka stock kam hai, sirf {q} {unit} de paunga",
                               f"Short supply of {nm}, can only send {q} {unit}",
                               f"Sir {nm} sirf {q} {unit} mil raha hai aaj"])
            label.update(available_qty=q, unit=unit)
        elif intent == "substitution":
            alt = pool[pool.category == r["category"]].iloc[rng.integers(len(pool[pool.category == r["category"]]))]["name"]
            price = int(round(r["base_price"] * rng.uniform(.9, 1.2)))
            text = rng.choice([f"{nm} nahi hai, {alt.replace('_', ' ')} bhej dun? rate {price}",
                               f"No {nm} today. Can send {alt.replace('_', ' ')} at Rs {price}/{unit}?"])
            label.update(substitute=alt, price=price)
        else:
            q = int(rng.integers(5, 80))
            inv = int(rng.integers(1000, 9999))
            text = rng.choice([f"{q} {unit} {nm} dispatched, invoice no {inv}",
                               f"Bhej diya {q} {unit} {nm}. Invoice {inv}"])
            label.update(qty=q, unit=unit, invoice_no=str(inv))
        out.append({"id": f"MSG{k:04d}", "source": "whatsapp", "supplier_id": CAT_SUPPLIERS[r["category"]][0],
                    "text": str(text), "label": label})
    return out


def _ocr(rng, s, p=.06):
    swap = {"0": "O", "1": "l", "5": "S", "8": "B"}
    return "".join(swap[c] if c in swap and rng.random() < p else c for c in s)


def make_delivery_notes(rng, ing, n=150):
    out = []
    for k in range(n):
        cat = rng.choice(["veg", "dairy", "meat", "dry"])
        sub = ing[ing.category == cat]
        picks = sub.iloc[rng.choice(len(sub), size=min(len(sub), int(rng.integers(3, 6))), replace=False)]
        sup = CAT_SUPPLIERS[cat][0]
        inv = f"INV-{rng.integers(1000, 9999)}"
        lines, items, total = [f"{dict((s[0], s[1]) for s in SUPPLIERS)[sup].upper()}", f"INV NO: {inv}"], [], 0.0
        for j, (_, r) in enumerate(picks.iterrows(), 1):
            q = int(rng.integers(3, 60)) if r["unit"] != "piece" else int(rng.choice([60, 120, 180, 240]))
            rate = round(r["base_price"] * rng.uniform(.9, 1.2), 2)
            amt = round(q * rate, 2)
            total += amt
            lines.append(_ocr(rng, f"{j}. {r['name'].replace('_', ' ').upper()}  {q} {r['unit'].upper()} @ {rate:.2f} = {amt:.2f}"))
            items.append({"ingredient": r["name"], "qty": q, "rate": rate, "amount": amt})
        lines.append(f"TOTAL (before GST) {total:.2f}")
        out.append({"id": f"DN{k:04d}", "source": "invoice_ocr", "supplier_id": sup, "text": "\n".join(lines),
                    "label": {"invoice_no": inv, "items": items, "total_before_gst": round(total, 2)}})
    return out


def make_voice_notes(rng, ing, n=150):
    out = []
    pool = ing[ing.category.isin(["veg", "dairy", "meat"])]
    for k in range(n):
        picks = pool.iloc[rng.choice(len(pool), size=int(rng.integers(4, 8)), replace=False)]
        parts, counts = [], {}
        for _, r in picks.iterrows():
            q = int(rng.integers(1, 13)) if r["unit"] != "piece" else int(rng.choice([30, 60, 90]))
            counts[r["name"]] = q
            word = NUM_WORDS.get(q)
            qs = str(q) if (word is None or rng.random() < .5) else word[int(rng.random() < .5)]
            u = {"kg": "kilo", "l": "litre", "piece": ""}[r["unit"]]
            parts.append(f"{_nm(rng, r['name'])} {qs} {u}".strip())
        text = rng.choice(["Aaj closing: ", "Stock count: ", "Kitchen mein abhi: "]) + ", ".join(parts)
        out.append({"id": f"VN{k:04d}", "source": "voice_transcript", "text": text, "label": {"counts": counts}})
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_pos")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--start", default="2025-07-01")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    save = lambda df, name: (df.to_csv(os.path.join(a.out, name), index=False), print(f"{name:28s} {len(df):>8,} rows"))

    W = build_world(a.seed, a.start, a.days)
    rng = np.random.default_rng(a.seed + 1)
    w, days, ing = W["warm"], W["days"], W["ing"]
    out_days = days[w:]

    A, pos_list = simulate_inventory(W, policy_gut_feel)
    inv, po, counts = inventory_tables(W, A, pos_list, rng)
    supp = make_suppliers(rng, out_days[-1])
    inv_df = make_invoices(rng, po, ing, supp)
    lic, staff = make_licences_and_staff(rng, out_days[-1])

    pos = W["pos"][W["pos"].timestamp >= out_days[0]].drop(columns="date")
    sp = pd.concat([pd.DataFrame({"date": out_days, "ingredient_id": r["ingredient_id"], "supplier_id": s,
                                  "price": (W["P"] if k == 0 else W["AP"])[w:, i].round(2)})
                    for i, r in ing.iterrows() for k, s in enumerate([W["prim"][i], W["alt"][i]])])

    save(pd.DataFrame(OUTLETS, columns=["outlet_id", "name", "city", "volume_scale", "seats"]), "outlets.csv")
    save(pd.DataFrame([(d[0], d[1], d[2], d[3], d[4]) for d in DISHES],
                      columns=["dish_id", "name", "course", "price", "is_veg"]), "menu.csv")
    save(pd.DataFrame([(d[0], i, q) for d in DISHES for i, q in d[6].items()],
                      columns=["dish_id", "ingredient", "qty_per_portion"])
         .assign(ingredient_id=lambda x: x.ingredient.map(dict(zip(ing["name"], ing["ingredient_id"]))))
         [["dish_id", "ingredient_id", "ingredient", "qty_per_portion"]], "recipes.csv")
    save(ing, "ingredients.csv")
    save(supp, "suppliers.csv")
    save(sp, "supplier_prices.csv")
    save(W["cal"][W["cal"].date >= out_days[0]], "calendar_context.csv")
    save(pos, "pos_order_lines.csv")
    save(po, "purchase_orders.csv")
    save(inv, "daily_inventory.csv")
    save(counts, "stock_counts.csv")
    save(inv_df, "purchase_invoices.csv")
    save(make_temp_logs(rng, out_days), "temperature_logs.csv")
    save(lic, "licence_registry.csv")
    save(staff, "staff_compliance.csv")
    write_jsonl(os.path.join(a.out, "supplier_messages.jsonl"), make_supplier_messages(rng, ing))
    write_jsonl(os.path.join(a.out, "delivery_notes_ocr.jsonl"), make_delivery_notes(rng, ing))
    write_jsonl(os.path.join(a.out, "kitchen_voice_notes.jsonl"), make_voice_notes(rng, ing))
    with open(os.path.join(a.out, "compliance_rules_simplified.md"), "w", encoding="utf-8") as f:
        f.write(RULES_MD)
    print("\nHistorical (gut-feel) policy KPIs:", kpis(W, A))
    print(f"Saved to ./{a.out}/")


if __name__ == "__main__":
    main()
