"""
backend_skeleton.py
-------------------
A minimal backend that a POS system could talk to. It shows the SHAPE of the product;
teams replace the stubs with better models, LLM calls and real integrations.

    pip install fastapi uvicorn pandas numpy
    python generate_pos_data.py --out data_pos
    DATA_DIR=data_pos uvicorn backend_skeleton:app --reload      # then open http://127.0.0.1:8000/docs

Endpoints
  POST /pos/orders                       POS webhook: record a sale, deplete inventory via recipes (BOM)
  GET  /inventory/{outlet_id}            current book stock per ingredient
  GET  /recommendations/orders/{outlet}  suggested purchase orders grouped by supplier (human approves)
  POST /purchase-orders/approve          human-in-the-loop approval -> creates POs
  GET  /compliance/{outlet_id}           findings from the rule engine
  GET  /reports/daily/{outlet_id}        sales, theoretical food cost, wastage, top dishes
"""
import os
import sqlite3
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from compliance_engine import run_all
from generate_pos_data import forecast_daily, order_multiple

DATA = os.environ.get("DATA_DIR", "data_pos")
app = FastAPI(title="Smart Kitchen Backend (teaching skeleton)")


def load_db():
    """Load CSVs into an in-memory SQLite DB. In production this is Postgres + events from the POS."""
    db = sqlite3.connect(":memory:", check_same_thread=False)
    for name in ["menu", "recipes", "ingredients", "suppliers", "daily_inventory", "pos_order_lines", "purchase_orders"]:
        pd.read_csv(os.path.join(DATA, f"{name}.csv")).to_sql(name, db, index=False)
    # live book stock starts from the last closing balance
    last = pd.read_sql("SELECT outlet_id, ingredient_id, closing_qty AS on_hand FROM daily_inventory "
                       "WHERE date = (SELECT MAX(date) FROM daily_inventory)", db)
    last.to_sql("book_stock", db, index=False)
    pd.DataFrame(columns=["po_id", "outlet_id", "ingredient_id", "supplier_id", "qty", "status"]).to_sql("approved_pos", db, index=False)
    return db


DB = load_db()


class OrderLine(BaseModel):
    dish_id: str
    qty: int = 1


class PosOrder(BaseModel):
    order_id: str
    outlet_id: str
    lines: List[OrderLine]


class PoLine(BaseModel):
    outlet_id: str
    ingredient_id: str
    supplier_id: str
    qty: float


@app.post("/pos/orders")
def pos_order(o: PosOrder):
    """Deplete book stock using the recipe (bill of materials). This is THE link from POS to inventory."""
    depleted = {}
    for ln in o.lines:
        rows = DB.execute("SELECT ingredient_id, qty_per_portion FROM recipes WHERE dish_id=?", (ln.dish_id,)).fetchall()
        if not rows:
            raise HTTPException(404, f"Unknown dish {ln.dish_id}")
        for ing, q in rows:
            depleted[ing] = depleted.get(ing, 0) + q * ln.qty
    for ing, q in depleted.items():
        DB.execute("UPDATE book_stock SET on_hand = on_hand - ? WHERE outlet_id=? AND ingredient_id=?", (q, o.outlet_id, ing))
    DB.commit()
    return {"order_id": o.order_id, "depleted": {k: round(v, 3) for k, v in depleted.items()}}


@app.get("/inventory/{outlet_id}")
def inventory(outlet_id: str):
    df = pd.read_sql("SELECT b.ingredient_id, i.name, i.unit, ROUND(b.on_hand,2) AS on_hand FROM book_stock b "
                     "JOIN ingredients i USING(ingredient_id) WHERE outlet_id=?", DB, params=(outlet_id,))
    if df.empty:
        raise HTTPException(404, "Unknown outlet")
    return df.to_dict("records")


@app.get("/recommendations/orders/{outlet_id}")
def recommend(outlet_id: str, z: float = 1.5):
    """STUB: statistical order-up-to using the last 28 days of usage. Replace with your ML + LLM logic."""
    hist = pd.read_sql("SELECT ingredient_id, date, actual_usage FROM daily_inventory WHERE outlet_id=? ORDER BY date",
                       DB, params=(outlet_id,))
    ing = pd.read_sql("SELECT * FROM ingredients", DB).set_index("ingredient_id")
    sup = pd.read_sql("SELECT supplier_id, name FROM suppliers", DB).set_index("supplier_id")["name"]
    stock = pd.read_sql("SELECT ingredient_id, on_hand FROM book_stock WHERE outlet_id=?", DB, params=(outlet_id,)).set_index("ingredient_id")["on_hand"]
    lines = []
    for iid, g in hist.groupby("ingredient_id"):
        h = g["actual_usage"].values[-28:]
        r = ing.loc[iid]
        perishable = r["shelf_life_days"] <= 14
        H = int(r["lead_time_days"]) + (1 if perishable else 7)
        target = sum(forecast_daily(h, k) for k in range(1, H + 1)) + z * h.std() * np.sqrt(H)
        qty = order_multiple(target - max(stock.get(iid, 0), 0), r["min_order_qty"])
        if qty > 0:
            lines.append({"ingredient_id": iid, "name": r["name"], "qty": qty, "unit": r["unit"], "reason": f"cover {H} days + safety stock"})
    return {"outlet_id": outlet_id, "lines": lines, "note": "Needs human approval via POST /purchase-orders/approve"}


@app.post("/purchase-orders/approve")
def approve(lines: List[PoLine]):
    n0 = DB.execute("SELECT COUNT(*) FROM approved_pos").fetchone()[0]
    for k, ln in enumerate(lines):
        DB.execute("INSERT INTO approved_pos VALUES (?,?,?,?,?,?)",
                   (f"APO{n0 + k:05d}", ln.outlet_id, ln.ingredient_id, ln.supplier_id, ln.qty, "sent_to_supplier"))
    DB.commit()
    return {"created": len(lines), "status": "sent_to_supplier (stub: wire to WhatsApp/email/ERP)"}


@app.get("/compliance/{outlet_id}")
def compliance(outlet_id: str):
    findings, today = run_all(DATA)
    f = findings[(findings["outlet_id"] == outlet_id) | (findings["outlet_id"] == "ALL")]
    return {"as_of": str(today.date()), "count": len(f), "findings": f.head(50)[["rule", "severity", "outlet_id", "subject", "detail"]].to_dict("records")}


@app.get("/reports/daily/{outlet_id}")
def daily_report(outlet_id: str, date: str):
    sales = pd.read_sql("SELECT p.dish_id, m.name, SUM(p.qty) AS qty, SUM(p.qty*p.unit_price*(1-p.discount_pct/100.0)) AS revenue "
                        "FROM pos_order_lines p JOIN menu m USING(dish_id) WHERE p.outlet_id=? AND DATE(p.timestamp)=? "
                        "GROUP BY p.dish_id", DB, params=(outlet_id, date))
    inv = pd.read_sql("SELECT SUM(theoretical_usage*unit_cost) AS theoretical_cogs, SUM(actual_usage*unit_cost) AS actual_cogs, "
                      "SUM(expired_qty*unit_cost) AS wastage FROM daily_inventory WHERE outlet_id=? AND DATE(date)=?",
                      DB, params=(outlet_id, date))
    if sales.empty:
        raise HTTPException(404, "No data for that outlet/date")
    rev = float(sales["revenue"].sum())
    r = inv.iloc[0]
    return {"date": date, "revenue": round(rev), "theoretical_food_cost_pct": round(100 * r["theoretical_cogs"] / rev, 1),
            "actual_food_cost_pct": round(100 * r["actual_cogs"] / rev, 1), "wastage_Rs": round(float(r["wastage"])),
            "top_dishes": sales.sort_values("qty", ascending=False).head(5)[["name", "qty"]].to_dict("records")}
