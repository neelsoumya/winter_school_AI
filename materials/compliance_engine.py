"""
compliance_engine.py
--------------------
The "traditional" half of compliance: deterministic rules over structured data.
It finds problems reliably and cheaply. An LLM is then used only to explain,
prioritise and draft actions (see llm_pipeline_template.py).

Rules implemented (see compliance_rules_simplified.md; teaching rules, NOT legal advice):
  R01 cold-chain incidents    R02 missing temperature logs   R03 licence expiry
  R04 staff certificates      R05 supplier licences          R06 invoice reconciliation
  R07 wastage review          R08 stock variance

    python compliance_engine.py --data data_pos
"""
import argparse
import os

import numpy as np
import pandas as pd

LIMITS = {"cold_room": 5.0, "display_chiller": 5.0, "freezer": -15.0}


def _finding(rule, severity, outlet, subject, detail, **extra):
    return {"rule": rule, "severity": severity, "outlet_id": outlet, "subject": subject, "detail": detail, **extra}


def check_cold_chain(temp):
    out = []
    t = temp.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"])
    t = t.sort_values(["outlet_id", "unit", "timestamp"])
    for (o, u), g in t.groupby(["outlet_id", "unit"]):
        breach = (g["temp_c"] > LIMITS[u]).astype(int).values
        run, start = 0, None
        for k, b in enumerate(breach):
            if b:
                run += 1
                start = start if start is not None else g["timestamp"].iloc[k]
            if (not b or k == len(breach) - 1) and run:
                if run >= 2:
                    out.append(_finding("R01", "critical" if run >= 4 else "high", o, u,
                                        f"{run} consecutive breaches starting {start:%d %b %H:%M}, "
                                        f"max {g['temp_c'].iloc[max(0, k - run):k + 1].max():.1f} C",
                                        consecutive_readings=run, start=str(start)))
                run, start = 0, None
        # missing logs: gaps > 8 hours
        gaps = g["timestamp"].diff().dt.total_seconds().div(3600)
        for ts, gap in zip(g["timestamp"][gaps > 8], gaps[gaps > 8]):
            out.append(_finding("R02", "medium", o, u, f"No readings for {gap:.0f} h ending {ts:%d %b %H:%M}"))
    return out


def check_licences(lic, today):
    out = []
    for _, r in lic.iterrows():
        days = (pd.to_datetime(r["expiry_date"]) - today).days
        if days < 0:
            out.append(_finding("R03", "critical", r["outlet_id"], r["licence_type"], f"EXPIRED {-days} days ago"))
        elif days <= 30:
            out.append(_finding("R03", "high", r["outlet_id"], r["licence_type"], f"Expires in {days} days"))
    return out


def check_staff(staff, today):
    out = []
    for col, label in [("food_safety_training_expiry", "food-safety training"), ("medical_fitness_expiry", "medical fitness")]:
        bad = staff[pd.to_datetime(staff[col]) < today]
        for o, g in bad.groupby("outlet_id"):
            out.append(_finding("R04", "high", o, label, f"{len(g)} staff with expired {label}: {', '.join(g['staff_id'])}"))
    return out


def check_suppliers(supp, po, today):
    out = []
    used = set(po["supplier_id"])
    for _, r in supp.iterrows():
        if r["supplier_id"] not in used:
            continue
        if pd.isna(r["food_licence_expiry"]) or r["food_licence_no"] != r["food_licence_no"]:
            out.append(_finding("R05", "medium", "ALL", r["name"], "No food licence on record: needs manager approval"))
        elif pd.to_datetime(r["food_licence_expiry"]) < today:
            out.append(_finding("R05", "high", "ALL", r["name"], f"Supplier food licence expired {r['food_licence_expiry']}"))
    return out


def check_invoices(inv, supp):
    reg = set(supp.loc[supp["gstin"].fillna("") != "", "supplier_id"])
    out = []
    bad_total = inv[(inv["taxable_value"] + inv["gst_amount"] - inv["total_amount"]).abs() > 1.0]
    for _, r in bad_total.iterrows():
        diff = r["total_amount"] - (r["taxable_value"] + r["gst_amount"])
        out.append(_finding("R06", "medium", r["outlet_id"], r["invoice_id"], f"Total off by Rs {diff:+.0f}", error_type="total_mismatch"))
    miss = inv[(inv["supplier_id"].isin(reg)) & (inv["supplier_gstin_on_invoice"].fillna("") == "")]
    for _, r in miss.iterrows():
        out.append(_finding("R06", "medium", r["outlet_id"], r["invoice_id"], "GSTIN missing on invoice from registered supplier", error_type="missing_gstin"))
    return out


def check_wastage(inv_daily, ing):
    out = []
    d = inv_daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["week"] = d["date"].dt.to_period("W")
    d["spend"] = d["received_qty"] * d["unit_cost"]
    d["waste"] = d["expired_qty"] * d["unit_cost"]
    g = d.groupby(["outlet_id", "ingredient_id"])[["spend", "waste"]].sum()
    g["pct"] = 100 * g["waste"] / g["spend"].clip(lower=1)
    names = ing.set_index("ingredient_id")["name"]
    for (o, i), r in g[(g["pct"] > 8) & (g["waste"] > 2000)].iterrows():
        out.append(_finding("R07", "medium", o, names[i], f"Expiry wastage {r['pct']:.1f}% of purchases (Rs {r['waste']:,.0f})"))
    return out


def check_variance(counts, ing, threshold=7.0, min_price=300):
    price = ing.set_index("ingredient_id")["base_price"]
    names = ing.set_index("ingredient_id")["name"]
    c = counts[counts["ingredient_id"].map(price) > min_price]
    g = c.groupby(["outlet_id", "ingredient_id"])["variance_pct_of_weekly_use"].agg(["mean", "count"])
    out = []
    for (o, i), r in g[g["mean"].abs() > threshold].iterrows():
        out.append(_finding("R08", "high", o, names[i], f"Average weekly stock variance {r['mean']:+.1f}% over {int(r['count'])} counts"))
    return out


def run_all(data_dir, today=None):
    rd = lambda n: pd.read_csv(os.path.join(data_dir, n))
    temp, lic, staff = rd("temperature_logs.csv"), rd("licence_registry.csv"), rd("staff_compliance.csv")
    supp, po, inv = rd("suppliers.csv"), rd("purchase_orders.csv"), rd("purchase_invoices.csv")
    daily, counts, ing = rd("daily_inventory.csv"), rd("stock_counts.csv"), rd("ingredients.csv")
    today = pd.Timestamp(today) if today is not None else pd.to_datetime(temp["timestamp"]).max().normalize()
    findings = (check_cold_chain(temp) + check_licences(lic, today) + check_staff(staff, today)
                + check_suppliers(supp, po, today) + check_invoices(inv, supp)
                + check_wastage(daily, ing) + check_variance(counts, ing))
    df = pd.DataFrame(findings)
    order = {"critical": 0, "high": 1, "medium": 2}
    return df.sort_values(by="severity", key=lambda s: s.map(order)).reset_index(drop=True), today


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_pos")
    a = ap.parse_args()
    df, today = run_all(a.data)
    print(f"As of {today:%d %b %Y}: {len(df)} findings\n")
    print(df.groupby(["rule", "severity"]).size().unstack(fill_value=0))
    print("\nTop findings:")
    print(df[df.severity.isin(["critical", "high"])].head(12)[["rule", "severity", "outlet_id", "subject", "detail"]].to_string(index=False))
