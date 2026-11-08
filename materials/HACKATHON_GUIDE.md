# From Model to Venture: AI Hackathon for Indian Startups

A one-day workshop plus hackathon for masters-in-AI students in Mumbai, working in mixed teams with MBA and Finance students.

---

## 1. Purpose

Students leave having done one thing end to end: **take a messy business problem, build a working AI solution, prove it makes money, deploy it, and pitch it in 5 minutes.**

### Learning outcomes

By the end, each student should be able to:

1. **Start from the problem, not the technology.** Write a one-paragraph problem statement with a named customer and a rupee cost of the problem.
2. **Think like a founder.** Ask "what job could this create or what business could this become?" and not only "what role could I get?"
3. **Work across domains.** Translate between CS (models), Finance (unit economics, risk) and MBA (customer, market, go-to-market).
4. **Handle realistic data.** Missing values, duplicates, class imbalance, censored demand, leakage.
5. **Judge a model in rupees, not accuracy.** Connect precision, recall and thresholds to cost and profit.
6. **Ship.** Deploy a working demo on Hugging Face Spaces (Streamlit or Gradio).
7. **Pitch.** Tell the story in 5 minutes to a sceptical audience.

---

## 2. Suggested schedule

### Day 1: Workshop (about 7 hours)

| Time | Session | Notes |
|---|---|---|
| 09:30 | Welcome and the startup mindset | Problems, not technologies. Employment creators, not only employees |
| 10:15 | Guest lecture 1 | Founder journey: what actually went wrong in year one |
| 11:00 | Break and team formation | Mixed teams of 4 to 5 (see section 3) |
| 11:15 | Problem-first canvas (section 4) | Teams pick a case study and fill the canvas |
| 12:15 | Data walkthrough | Run the generator, EDA live, show the planted messiness |
| 13:00 | Lunch | |
| 14:00 | Baseline model + business cost (live code) | Use `baseline_model.py` |
| 15:00 | Deploy in 30 minutes | Streamlit or Gradio on Hugging Face Spaces |
| 15:45 | Guest lecture 2 | Problems to expect in startups: money, hiring, regulation, customers |
| 16:30 | Teams start building | Mentors rotate |
| 17:30 | Check-in and homework | Each team posts a one-line hypothesis |

### Day 2: Hackathon (about 8 hours; or run it a week later)

| Time | Session |
|---|---|
| 09:30 | Kick-off, rules, rubric |
| 10:00 | Build sprint 1 (model + unit economics) |
| 12:30 | Mentor office hours: CS, Finance and MBA mentors on separate tables |
| 14:00 | Build sprint 2 (app + deploy + deck) |
| 16:00 | Freeze. Rehearse the pitch |
| 16:30 | 5-minute pitches + 2 minutes of Q&A each |
| 18:00 | Judging, feedback, awards |

**Guest speakers:** Saikat, Urgin, Madhuparna and others. Give each a prompt (section 10).

---

## 3. Team structure

Mixed teams of 4 to 5: **2 CS/AI + 1 Finance + 1 MBA (+ 1 flexible)**.

| Role | Owns | Typical deliverable |
|---|---|---|
| **CS / AI** | Data cleaning, model, deployment | Working app, metrics, error analysis |
| **Finance** | Unit economics, risk, pricing, break-even | Cost of the problem in ₹, payback period, sensitivity table |
| **MBA** | Customer, market size, competition, go-to-market | Customer interviews (even 3), TAM/SAM/SOM, pricing, pitch story |
| **Everyone** | The problem statement and the pitch | One-page canvas |

**Rule:** no one is allowed to say "that's not my part". The CS students must explain the model to Finance in plain language. Finance and MBA students must read the data dictionary and run the notebook at least once.

---

## 4. The problem-first canvas

Teams fill this in before touching a model. Print it or share it as a doc.

```
1. Who has the problem?            (a named customer: "a 12-store kirana chain owner in Pune")
2. What is the problem, exactly?   (one sentence, no technology words)
3. What does it cost them today?   (₹ per month, with your assumption stated)
4. How do they solve it today?     (spreadsheet, gut feel, WhatsApp, nothing)
5. What decision will our AI change?  (a decision, not a prediction)
6. What data do we need, and can we get it?
7. How will we know it worked?     (metric in ₹, not accuracy)
8. Who pays, how much, and why?    (pricing hypothesis)
9. What could go wrong?            (bias, privacy, regulation, wrong predictions)
10. If this works, what is the business? (product, team, jobs created)
```

> **Coaching line for mentors:** "Your model predicts something. What decision does it change, and who is paid to make that decision?"

---

## 5. Quick start

```bash
# 1. Install
pip install pandas numpy scikit-learn streamlit gradio matplotlib

# 2. Generate all six datasets (fixed seed so every team gets the same data)
python generate_synthetic_data.py --seed 42 --out data

# 3. Baseline model for the RTO case study
python baseline_model.py

# 4. Run the demo app
streamlit run app_streamlit_template.py
```

Use `--scale 2` for larger datasets. Use a different `--seed` per team if you want teams to have different data (useful if you want to prevent copying).

### Files

| File | Purpose |
|---|---|
| `generate_synthetic_data.py` | Creates all six CSVs with planted signal and realistic mess |
| `baseline_model.py` | Readable baseline: encode, fit, evaluate, compute ₹ cost |
| `app_streamlit_template.py` | Deployable app with the pattern: input, prediction, ₹ impact, action |
| `HACKATHON_GUIDE.md` | This document |

---

## 6. Case studies

Each case is designed so that **CS, Finance and MBA students each have real work to do.** Students may also propose their own problem (encourage this) and use the generator as a template.

### Case 1. GigCredit: lending to gig workers and small merchants

**Startup context.** An NBFC-fintech wants to lend ₹20,000 to ₹2.5 lakh to delivery partners, drivers, kirana owners and artisans who have no CIBIL history but have UPI transaction trails.

**The problem.** Defaults are about 12%. Traditional underwriting rejects most of these customers. Approve too loosely and the book bleeds; approve too tightly and growth stalls.

**Data:** `credit_applications.csv` (about 5,000 rows)
`age, city, city_tier, occupation, monthly_income, income_volatility, upi_txn_per_month, avg_upi_txn_value, months_on_upi, bill_payment_regularity, existing_emi, loan_amount, tenure_months, defaulted`

| Role | Tasks |
|---|---|
| **CS** | Predict default; handle missing income (6%) and bill regularity (10%); remove duplicate applications; calibrate probabilities |
| **Finance** | Expected loss = PD x LGD x exposure. Choose an approval cut-off that maximises profit given interest rate, cost of funds, collection cost. Build a profit-vs-threshold curve |
| **MBA** | Segment customers (which occupations to target?), pricing by risk band, partnership strategy (Swiggy/Zomato partners, kirana distributors), customer acquisition |

**Traps and discussion.**
- Is `city_tier` a fair feature? Does the model discriminate indirectly? (Discuss fairness and proxies.)
- Rejected applicants have no outcome. What does that do to future training data (selection bias)?
- Regulation: discuss RBI digital-lending rules and the DPDP Act 2023 (check the current text; they change).

**Stretch.** Explain each rejection in plain Hindi or Marathi using feature contributions.

---

### Case 2. DarkStore Demand: never run out of milk, never overstock umbrellas

**Startup context.** A quick-commerce / kirana-tech startup supplies 20 neighbourhood stores in Mumbai, Pune, Nagpur and Indore with five SKUs.

**The problem.** Stockouts lose sales; overstock kills perishables and cash. Demand moves with weekends, festivals (Holi, Eid, Ganesh Chaturthi, Navratri, Diwali), monsoon rain and discounts.

**Data:** `kirana_demand.csv` (about 36,500 rows)
`store_id, city, store_size, date, sku, price, discount_pct, festival, rain_mm, units_sold, stockout`

| Role | Tasks |
|---|---|
| **CS** | Forecast daily demand per store and SKU; compare naive baseline, gradient boosting and a time-series model; evaluate on a time-based split (never random) |
| **Finance** | Working-capital cost of inventory, margin lost per stockout, wastage cost for milk; choose a service level (for example 95%) and compute safety stock cost |
| **MBA** | Which stores and SKUs to prioritise? Supplier negotiation, pricing and promotions, expansion to new neighbourhoods |

**Traps and discussion.**
- **Censored demand:** when `stockout = 1`, `units_sold` is lower than true demand. A model trained naively under-forecasts. How should they correct for this?
- Umbrellas sell almost only when it rains. Is rain forecast available at order time?

**Stretch.** Turn forecasts into a reorder recommendation with a cost-minimising order quantity.

---

### Case 3. COD-Guard: stopping Return-to-Origin losses for a D2C brand

**Startup context.** A D2C fashion and beauty brand ships 20,000 orders, many cash-on-delivery, across tier 1, 2 and 3 cities. About 20% come back undelivered.

**The problem.** Each RTO costs forward + reverse shipping + handling, plus a lost sale. Blocking COD loses customers. Which orders deserve an intervention?

**Data:** `rto_orders.csv` (20,000 rows)
`order_date, city_tier, payment_mode, category, order_value, discount_pct, is_first_order, past_orders, address_quality, delivery_distance_km, courier, order_hour, forward_shipping_cost, rto`

| Role | Tasks |
|---|---|
| **CS** | Start from `baseline_model.py`; improve it; threshold tuning; explain predictions |
| **Finance** | Cost per RTO, cost of a prepaid incentive, net benefit of intervening per risk band; break-even incentive |
| **MBA** | Which action: call to confirm, WhatsApp verification, prepaid discount, courier switch? Customer experience impact, courier negotiation |

**Traps and discussion.**
- Accuracy is misleading at 20% base rate. Use PR-AUC and the ₹ metric.
- Late-night orders and a specific courier show higher RTO. Is it cause or correlation?
- A working demo already exists: `app_streamlit_template.py`.

**Stretch.** A/B test design: how many orders do you need to prove the intervention works?

---

### Case 4. ClaimSense: fraud and leakage in health insurance claims

**Startup context.** An insurtech / TPA processes thousands of cashless and reimbursement claims. Around 4% are fraudulent or inflated.

**The problem.** Manual review is slow and costly. Which claims should be fast-tracked and which should be investigated?

**Data:** `health_claims.csv` (about 8,000 rows)
`age, policy_age_days, hospital_type, procedure, claim_amount, days_admitted, prior_claims, docs_complete, is_fraud`

| Role | Tasks |
|---|---|
| **CS** | Imbalanced classification (4%); anomaly detection as an alternative; rank claims by risk |
| **Finance** | Review cost per claim vs expected fraud savings; how many claims can the investigation team handle (capacity constraint)? Loss ratio impact |
| **MBA** | Hospital network strategy, customer experience for honest claimants (false positives hurt!), partnership with hospitals |

**Traps and discussion.**
- What does a false positive cost in customer trust, versus a missed fraud in rupees?
- Compare claim amount to the typical amount for the procedure. Is that feature leaking the label?

**Stretch.** Queue design: fast-track, standard, investigate. Compute savings per queue.

---

### Case 5. MandiMitra: price forecasting for farmers and traders

**Startup context.** An agri-tech startup advises farmers and FPOs on when and where to sell onion, tomato and potato across Lasalgaon, Vashi, Pune and Azadpur mandis.

**The problem.** Prices swing with seasonality and heavy rain. A farmer who sells at the wrong time or in the wrong mandi loses a large share of income.

**Data:** `mandi_prices.csv` (about 8,500 rows; two years of daily data)
`date, mandi, commodity, rain_mm, arrivals_tonnes, modal_price`

| Role | Tasks |
|---|---|
| **CS** | 7-day and 30-day forecasts; lag and rolling features (for example 7-day rain); clean typo outliers (prices x10) and missing days |
| **Finance** | Value of better timing per quintal; transport cost vs price gain between mandis; storage cost and spoilage for tomato vs potato; farmer-level income impact |
| **MBA** | Who pays: farmers (unlikely), FPOs, traders, banks? Distribution in rural India, vernacular and voice interface, trust |

**Traps and discussion.**
- `arrivals_tonnes` is observed on the same day as price. Using it to predict tomorrow's price? Check for leakage.
- Which language and channel would a real farmer use (WhatsApp, voice)?

**Stretch.** "Sell now or wait?" recommendation with a confidence band.

---

### Case 6. BillBuddy: churn and unit economics for a GST-billing SaaS

**Startup context.** A B2B SaaS sells GST invoicing and accounting to small shops across three plans (₹999, ₹2,999, ₹7,999 per month) acquired through ads, field sales, WhatsApp referrals and chartered accountant (CA) partners.

**The problem.** 18% of customers churn in 90 days. Customer acquisition cost (CAC) differs wildly by channel. Where should the next ₹10 lakh go?

**Data:** `saas_customers.csv` (3,000 rows)
`plan, mrr, acquisition_channel, cac, tenure_months, seats, logins_30d, feature_adoption, support_tickets, payment_failures, uses_gst_module, nps, churned_90d`

| Role | Tasks |
|---|---|
| **CS** | Churn model; which behaviours predict churn early; explainability for the customer-success team |
| **Finance** | LTV = ARPU x gross margin / churn; LTV:CAC and payback period by channel; cohort view; effect of cutting churn by 3 points |
| **MBA** | Channel strategy, pricing and plan design, retention playbook (who calls whom, with what offer), positioning vs Tally, Zoho and Vyapar |

**Traps and discussion.**
- Is NPS a cause of churn or a symptom? Is it available early enough to be useful?
- Which customers are worth saving? A churn prediction with no retention action has no value.

**Stretch.** A "save list" for the customer-success team with expected ₹ saved per call.

---

### Bring-your-own problem

Allow teams to propose their own with approval from a mentor. Ideas: ONDC seller onboarding, UPI merchant lending, electricity-bill anomaly detection for MSMEs, clinic appointment no-shows, school-fee default risk, last-mile delivery ETA for Mumbai traffic, WhatsApp commerce lead scoring. They can adapt `generate_synthetic_data.py` to produce their own data.

---

## 7. What every team must deliver

| Deliverable | Detail |
|---|---|
| **Problem canvas** | One page (section 4) |
| **Working model** | Beats a naive baseline; evaluated on a proper split |
| **Unit economics** | A table showing ₹ impact, with assumptions and a sensitivity analysis |
| **Deployed demo** | Live link on Hugging Face Spaces (Streamlit or Gradio) |
| **5-minute pitch** | Deck or live demo, same structure for everyone |
| **One-page risks** | Bias, privacy, regulation, failure modes |

---

## 8. Deploying on Hugging Face Spaces

Course reference for deployment: <https://neelsoumya.github.io/visualization_lecture/exercises/exercise_week7.html#huggingface>

**Steps**

1. Create a free account at huggingface.co and click **New Space**.
2. Choose SDK **Streamlit** or **Gradio**.
3. Upload `app.py` (rename `app_streamlit_template.py`), `baseline_model.py`, `generate_synthetic_data.py` and `requirements.txt`.
4. The Space builds automatically. Share the URL.

**`requirements.txt`**
```
pandas
numpy
scikit-learn
streamlit
```
(For Gradio use `gradio` instead of `streamlit`.)

**`README.md` header on the Space (Streamlit)**
```
---
title: RTO Risk Copilot
emoji: 📦
sdk: streamlit
app_file: app.py
---
```
For Gradio use `sdk: gradio` and `app_file: app.py`.

**Tips:** the app builds data on first run, so no CSV upload is needed. Keep training under a minute. Never put real personal data in a public Space.

### Gradio version of the same app

```python
# app.py (Gradio)
import gradio as gr
from baseline_model import train_rto_model, predict_one, rto_cost

model, metrics = train_rto_model("data/rto_orders.csv")

def score(payment, category, courier, tier, value, discount, first, past, addr, dist, hour, ship):
    row = dict(payment_mode=payment, category=category, courier=courier, city_tier=int(tier),
               order_value=value, discount_pct=discount, is_first_order=int(first),
               past_orders=past, address_quality=addr, delivery_distance_km=dist, order_hour=int(hour))
    p = predict_one(model, row)
    return f"{p:.0%}", f"₹{p * rto_cost(ship):,.0f}"

demo = gr.Interface(
    fn=score,
    inputs=[
        gr.Dropdown(["COD", "Prepaid"], value="COD", label="Payment mode"),
        gr.Dropdown(["fashion", "beauty", "electronics", "home_decor", "grocery"], value="fashion", label="Category"),
        gr.Dropdown(["ShipFast", "QuickKart", "DesiDelivery", "IndiaExpress"], value="ShipFast", label="Courier"),
        gr.Dropdown([1, 2, 3], value=2, label="City tier"),
        gr.Number(value=899, label="Order value (₹)"),
        gr.Slider(0, 50, value=20, step=10, label="Discount %"),
        gr.Checkbox(value=True, label="First-time customer"),
        gr.Number(value=0, label="Past orders"),
        gr.Slider(0, 1, value=0.6, label="Address quality"),
        gr.Number(value=400, label="Distance (km)"),
        gr.Slider(0, 23, value=22, step=1, label="Order hour"),
        gr.Number(value=80, label="Forward shipping cost (₹)"),
    ],
    outputs=[gr.Textbox(label="P(RTO)"), gr.Textbox(label="Expected loss")],
    title="RTO Risk Copilot",
)
demo.launch()
```

---

## 9. The 5-minute pitch

| Time | Slide | What to say |
|---|---|---|
| 0:00 | **Problem** | Name the customer and the ₹ pain. No technology words |
| 0:45 | **Today's solution and why it fails** | What do they do now? |
| 1:15 | **Our solution** | The decision our AI changes |
| 1:45 | **Live demo** | 60 seconds. Use a prepared example |
| 2:45 | **Proof** | Metric in ₹, baseline vs model, one honest limitation |
| 3:30 | **Business model** | Who pays, how much, unit economics (Finance leads) |
| 4:15 | **Go-to-market and team** | First 10 customers: how do you get them? (MBA leads) |
| 4:45 | **The ask** | What would you do with the next ₹25 lakh and 12 months? |

**Rule:** every team member speaks. CS covers the demo and proof, Finance covers economics, MBA covers problem and market.

---

## 10. Guest lecture prompts

Ask each founder to cover:

1. How did you choose the problem? How many ideas did you throw away?
2. The first customer: how did you find them, and what did they actually say?
3. Money: how did you fund the first 12 months? What would you do differently?
4. The hardest hiring decision.
5. A time the technology worked but the business did not.
6. Regulation and trust in India (RBI, SEBI, IRDAI, DPDP) and how you handled them.
7. Advice for students who want to create jobs, not only take one.

---

## 11. Judging rubric (100 points)

| Criterion | Points | What good looks like |
|---|---|---|
| Problem clarity and customer insight | 20 | Named customer, ₹ cost, evidence of customer conversations |
| Technical quality | 20 | Proper validation, sensible baseline, handles data issues, honest metrics |
| Business and financial reasoning | 20 | Unit economics, sensitivity, clear assumptions |
| Working deployed demo | 15 | Live link, usable, solves the stated decision |
| Pitch and teamwork | 15 | Clear story, every member contributes, cross-domain collaboration visible |
| Risks, ethics and realism | 10 | Fairness, privacy, regulation, limitations acknowledged |

Bonus (up to 5): a team that proposes and builds a credible problem of its own.

---

## 12. Instructor notes

- **The data is synthetic and we planted the patterns.** Say this on day one. Findings from it are not facts about India. The skill being practised is the workflow.
- **Ask "so what?" constantly.** Every chart and metric should be followed by "what decision does that change?"
- **Make Finance and MBA students feel essential.** Seat them in the middle of the team, give them separate mentor tables, and score business reasoning as heavily as the model.
- **Share the unit-economics habit.** Ask for a sensitivity table: "What if the intervention works half as well as you hope?"
- **Common failure modes:** random splits on time series; accuracy on imbalanced data; leakage (for example `arrivals_tonnes` on the same day); ignoring that the approve/reject decision changes future data; no action attached to a prediction.
- **Reproducibility:** the same `--seed` gives every team identical data. Use different seeds if you want to discourage copying.
- **Hugging Face Spaces free tier sleeps when idle.** Ask teams to open their app 10 minutes before pitching.

## 13. Employability: what students can say afterwards

- "I built and deployed an AI product for a real Indian business problem in 48 hours."
- "I translated a model into ₹ impact, with unit economics and a sensitivity analysis."
- "I worked in a cross-functional team with MBA and finance peers and pitched to founders."
- They have a live portfolio link, a one-page canvas and a recorded pitch to show recruiters or investors.

Encourage a follow-up: each team writes a 500-word post on what they built and what they would do next, and shares the Space link on LinkedIn.

## 14. Optional pre-work

- Skim two founder stories (any Indian startup, for example from podcasts or case studies).
- Read the data dictionary for the chosen case in section 6.
- Install Python packages from section 5 and run `generate_synthetic_data.py` before the workshop, so day one is not spent debugging installs.
