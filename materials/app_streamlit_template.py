"""
app_streamlit_template.py
-------------------------
A minimal, deployable Streamlit app: "RTO Risk Copilot" for a D2C brand.

Run locally:      streamlit run app_streamlit_template.py
Deploy:           rename to app.py, add requirements.txt, push to a Hugging Face Space
                  (SDK = Streamlit). See HACKATHON_GUIDE.md.

The pattern every team should copy:
    1. user inputs  ->  2. model prediction  ->  3. money impact  ->  4. recommended action
"""
import os
import subprocess
import sys

import streamlit as st

from baseline_model import predict_one, rto_cost, train_rto_model

DATA = "data/rto_orders.csv"

st.set_page_config(page_title="RTO Risk Copilot", page_icon="📦")
st.title("📦 RTO Risk Copilot")
st.caption("Predict which COD orders will come back, and decide what to do before shipping.")

# Make the app self-sufficient on a fresh Hugging Face Space
if not os.path.exists(DATA):
    st.info("No data found yet. Generate the synthetic dataset first.")
    if st.button("Generate synthetic data"):
        subprocess.run([sys.executable, "generate_synthetic_data.py"], check=True)
        st.rerun()
    st.stop()


@st.cache_resource(show_spinner="Training baseline model...")
def load_model():
    return train_rto_model(DATA)


model, metrics = load_model()

with st.sidebar:
    st.header("Model quality")
    st.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")
    st.metric("PR-AUC", f"{metrics['pr_auc']:.2f}", help=f"Random guessing = {metrics['base_rate']:.2f}")
    st.header("Business levers")
    threshold = st.slider("Intervene if P(RTO) above", 0.1, 0.9, 0.35, 0.05)
    incentive = st.number_input("Prepaid incentive per order (₹)", 0, 300, 40)
    success = st.slider("Chance customer switches to prepaid", 0.0, 1.0, 0.4, 0.05)

st.subheader("Order details")
c1, c2 = st.columns(2)
with c1:
    payment = st.selectbox("Payment mode", ["COD", "Prepaid"])
    category = st.selectbox("Category", ["fashion", "beauty", "electronics", "home_decor", "grocery"])
    courier = st.selectbox("Courier", ["ShipFast", "QuickKart", "DesiDelivery", "IndiaExpress"])
    tier = st.selectbox("City tier", [1, 2, 3])
    value = st.number_input("Order value (₹)", 99, 15000, 899)
    discount = st.slider("Discount %", 0, 50, 20, 10)
with c2:
    first = st.checkbox("First-time customer", True)
    past = 0 if first else st.number_input("Past orders", 1, 50, 3)
    addr = st.slider("Address quality score", 0.0, 1.0, 0.6, 0.05)
    dist = st.number_input("Delivery distance (km)", 5, 2500, 400)
    hour = st.slider("Order hour (0-23)", 0, 23, 22)
    ship = st.number_input("Forward shipping cost (₹)", 40, 300, 80)

row = dict(payment_mode=payment, category=category, courier=courier, city_tier=tier,
           order_value=value, discount_pct=discount, is_first_order=int(first),
           past_orders=past, address_quality=addr, delivery_distance_km=dist,
           order_hour=hour)

p = predict_one(model, row)
loss_if_rto = rto_cost(ship)
expected_loss = p * loss_if_rto
st.subheader("Result")
m1, m2, m3 = st.columns(3)
m1.metric("P(RTO)", f"{p:.0%}")
m2.metric("Loss if returned", f"₹{loss_if_rto:,.0f}")
m3.metric("Expected loss", f"₹{expected_loss:,.0f}")

if p >= threshold and payment == "COD":
    saving = success * expected_loss - incentive
    verdict = "worth it" if saving > 0 else "NOT worth it"
    st.warning(f"High risk. Offer a prepaid incentive of ₹{incentive}. "
               f"Expected net benefit: ₹{saving:,.0f} ({verdict}).")
else:
    st.success("Low enough risk. Ship as normal.")

st.caption("Synthetic data, for teaching only.")
