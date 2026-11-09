# Smart Kitchen Backend: AI for Restaurant and Retail Operations

**A cross-disciplinary hackathon problem for CS/AI, MBA, Finance, Hospitality, Food Technology and Law students.**

---

## 1. The problem in one paragraph

A restaurant group runs three outlets (Mumbai, Pune, Nagpur). Their **point-of-sale (POS)** system records every dish sold, but nothing connects those sales to what happens in the **kitchen store room**. Managers order vegetables, dairy, chicken and dry goods **by gut feel**, over WhatsApp, often forgetting what is already on order. Tomatoes rot, chicken runs out on Saturday night, cashew nuts quietly disappear, cold-room temperature logs are filled in on paper, and licences expire without anyone noticing. The owner finds out about all of this only at month-end.

**Your task:** design, prototype and pitch a **backend that plugs into the POS** and does four things automatically:

1. **Inventory tracking:** know what is in the store room right now, from sales and deliveries.
2. **Ordering:** decide what to buy, from whom, and when, so that the kitchen neither runs out nor wastes.
3. **Compliance:** monitor food-safety, licence, staff-certificate and invoice rules, and raise the right alert to the right person.
4. **Reporting:** give the owner a daily, trustworthy picture in plain language (food cost %, wastage, variance, compliance status).

You will do this by combining **large language models (open or closed source)** with **traditional machine learning, optimisation and rule engines**, and by working with people from other disciplines. The domain knowledge of a chef, a food-safety officer, a CA or a procurement manager is as important as your model.

This is a real problem for hospitality and retail businesses in India (restaurants, cloud kitchens, bakeries, grocery and quick-commerce stores) and the same architecture applies to all of them.

---

## 2. Why this is a good startup problem

| Observation | Why it matters |
|---|---|
| Food cost is typically one of the largest cost lines of a restaurant | Small percentage improvements are large rupee amounts |
| Margins are thin | An owner will pay for a tool that visibly saves money |
| Suppliers communicate by WhatsApp, voice notes and paper invoices | A big opportunity for LLMs (messy, multilingual, Hinglish text) |
| Compliance is manual and consequences are severe | Rules + alerts beat paper registers |
| POS vendors already exist but few close the loop to ordering and compliance | A gap for a focused startup |
| The same pattern (sales -> stock -> order -> compliance) repeats in retail | The market is larger than restaurants |

**Founder question for every team:** *"Who exactly would pay for this, how much, and what would they stop doing the day they start using it?"*

---

## 3. Users

| Persona | Job to be done | Pain today |
|---|---|---|
| **Owner / multi-outlet operator** | Know food cost, wastage and compliance risk across outlets | Hears about problems weeks late |
| **Outlet manager** | Approve daily orders, handle supplier issues | Orders from memory, WhatsApp chaos |
| **Chef / store keeper** | Count stock, plan prep, avoid 86-ing dishes | Counts on paper; no link to sales |
| **Supplier (often informal)** | Receive clear orders, get paid on time | Last-minute changes, unclear quantities |
| **Auditor / food-safety inspector** | See complete records on demand | Missing temperature logs, expired papers |
| **CA / accountant** | Reconcile invoices and GST | Manual entry, mismatched totals |

---

## 4. System to build

```
                 +-----------------------+
  Customer  ---> |   POS (billing app)   |  dish sold, qty, channel, discount
                 +-----------+-----------+
                             | events (webhook)
                             v
 +--------------------------------------------------------------------+
 |                         SMART KITCHEN BACKEND                       |
 |                                                                      |
 |  [1] INVENTORY     recipes (BOM) x sales -> book stock; stock counts |
 |  [2] ORDERING      forecast -> order-up-to -> choose supplier        |
 |  [3] INGESTION     LLM: WhatsApp / OCR invoice / voice note -> JSON  |
 |  [4] COMPLIANCE    rule engine -> findings -> LLM explains/prioritises|
 |  [5] REPORTING     daily briefing, food cost %, anomaly + chat Q&A   |
 +----------+---------------------------+-----------------------------+
            |                           |
            v                           v
   Manager approves orders       Owner dashboard / WhatsApp briefing
   (human in the loop)           Suppliers get structured POs
```

**Principle:** *LLMs read and write language; rules, ML and optimisation make the decisions that cost money.* Every consequential action (placing an order, discarding stock, reporting a breach) goes through a human or a deterministic check.

---

## 5. What is provided

Run `python generate_pos_data.py --out data_pos` (about 30 seconds). All data is **fictional**. Patterns were planted by us. Do not draw conclusions about real restaurants.

### Code

| File | Purpose |
|---|---|
| `generate_pos_data.py` | Generates all data below, plus an **inventory simulator** (FIFO batches, shelf life, supplier delays, short deliveries) and ordering policies |
| `compliance_engine.py` | Deterministic rule checks (R01 to R08) over structured data |
| `forecast_and_order_baseline.py` | Recipe explosion, ML usage forecast vs naive, and ordering-policy replay in rupees |
| `llm_pipeline_template.py` | Extraction from messy text and a compliance briefing, with `mock` (rules), `anthropic` (closed) and `ollama` (open source) providers, plus evaluation against labels and a grounding check |
| `backend_skeleton.py` | FastAPI skeleton: POS webhook, inventory, order recommendations, approvals, compliance, daily report |

### Data (180 days, 3 outlets, 17 dishes, 23 ingredients, 7 suppliers)

| File | Rows (approx.) | What it is |
|---|---|---|
| `pos_order_lines.csv` | 189,000 | One row per dish sold: order, outlet, timestamp, channel (dine-in, takeaway, delivery), qty, price, discount |
| `menu.csv`, `recipes.csv` | 17 / 82 | Dishes and the **bill of materials**: kg or litres of each ingredient per portion |
| `ingredients.csv` | 23 | Unit, storage type, **shelf life**, base price, price volatility, lead time, minimum order qty, GST rate (illustrative) |
| `suppliers.csv`, `supplier_prices.csv` | 7 / 8,280 | Reliability, payment terms, GSTIN, food licence and expiry; daily price quotes from a primary and an alternate supplier |
| `calendar_context.csv` | 540 | Rain (mm) and festivals by city and date |
| `purchase_orders.csv` | 5,300 | History of what was ordered, expected and received (late and short deliveries included) |
| `daily_inventory.csv` | 12,400 | Opening, received, theoretical usage (from POS), actual usage, expired, stockout and closing quantities per outlet, ingredient and day |
| `stock_counts.csv` | 1,700 | Weekly physical counts vs expected stock |
| `purchase_invoices.csv` | 1,700 | Supplier invoices with taxable value, GST and total |
| `temperature_logs.csv` | 9,400 | Cold room, freezer and display chiller readings every 4 hours (some logs missing) |
| `licence_registry.csv`, `staff_compliance.csv` | 12 / 36 | Outlet licences with expiry; staff training and medical certificates |
| `supplier_messages.jsonl` | 400 | WhatsApp-style messages (English and Hinglish) with ground-truth labels |
| `delivery_notes_ocr.jsonl` | 150 | Noisy OCR text of invoices with labels |
| `kitchen_voice_notes.jsonl` | 150 | Stock-count transcripts with labels |
| `compliance_rules_simplified.md` | 10 rules | Simplified **teaching** rules (not legal advice) |

Columns starting with `label_` and the `label` field in the `.jsonl` files are **ground truth** for evaluation. Do not use them as model inputs.

### Baseline numbers (seed 42, 180 days, same demand and supplier behaviour)

| Policy | Wastage (Rs) | Wastage % of spend | Stockout penalty (Rs) | Wastage + stockout (Rs) |
|---|---|---|---|---|
| Gut-feel manager (historical data) | 8.1 lakh | 7.9% | 3.1 lakh | 11.2 lakh |
| Forecast order-up-to (statistical baseline) | 0.3 lakh | 0.4% | 0.7 lakh | 1.1 lakh |

Stockout penalty = unavailable ingredient value x 4 (a proxy for lost gross margin). Purchase spend is about Rs 1 crore over 180 days across the three outlets.

> **Be sceptical of the baseline.** The simulator is kinder than reality: demand patterns are clean, suppliers are not strategic, and prices are not manipulated. Part of the challenge is to **stress-test your own policy** (supplier strike, price spike, festival surge, new menu item).

---

## 6. Modules and tasks

Each module lists the work by discipline. Teams do not need to finish every module. **Depth in two or three modules plus a convincing end-to-end story beats shallow coverage of all.**

### Module 1: POS to inventory (the backbone)

| Role | Tasks |
|---|---|
| CS / data | Recipe explosion (POS lines x BOM -> theoretical usage). Reconcile with `actual_usage`. Detect and explain gaps |
| Hospitality / chef | Are the recipe quantities realistic? What about trimming loss, yield, staff meals, tasting, prep waste? |
| Finance | Theoretical vs actual food cost %. Which gap is acceptable? |

**Discussion:** The stock count says there is less chicken than the POS predicts. Is it over-portioning, waste, theft, or a wrong recipe? How would you tell them apart, and how would you avoid wrongly accusing a staff member?

### Module 2: Demand forecasting

- Forecast **dish demand** or **ingredient usage** per outlet and day. Use calendar, rain, festival, lags. Compare naive, statistical and ML methods using a **time-based split**.
- Think about how to forecast *new* dishes with no history, and about promotions and delivery-app campaigns.
- Evaluate in **rupees**, not only WAPE: what is the cost of a 10% under-forecast of chicken vs a 10% over-forecast of coriander?

### Module 3: Ordering and procurement optimisation

- Decide **how much to order, when, and from which supplier**, per ingredient. Respect shelf life, lead time, minimum order quantity, and supplier delays and short deliveries.
- Baselines in `generate_pos_data.py`: `policy_gut_feel` and `policy_forecast`. Replace the forecast with your ML model, use the alternate supplier price (`supplier_prices.csv`), add price-risk logic (buy dry goods forward when cheap), or build a perishability-aware optimiser.
- **Questions for Finance and MBA:** working capital cost, payment terms (SUP01 wants cash; SUP05 gives 15 days), supplier concentration risk, negotiating power.

### Module 4: LLM ingestion of messy real-world inputs

- Convert WhatsApp messages, OCR invoices and voice-note transcripts into structured JSON the backend can use. Evaluate **field-level accuracy** against labels.
- Compare: rule-based `mock` baseline, a closed-source model, an open-source model (for example Llama, Qwen or Mistral via Ollama or Hugging Face), and a small fine-tuned or few-shot model.
- Compare on **accuracy, cost per 1,000 messages, latency, privacy, Hinglish ability, and failure modes** (invalid JSON, hallucinated numbers).
- Real WhatsApp is far messier than our templates. **Write harder test cases** (typos, voice-to-text errors, mixed units, sarcasm, two items in one message) and report where each approach breaks.

### Module 5: Compliance

- The rule engine already finds issues (cold-chain incidents, missing logs, licence expiry, staff certificates, supplier licences, invoice mismatches, wastage, stock variance).
- Your job: **prioritise** (the baseline produces dozens of alerts; owners ignore noisy systems), **explain** in plain English or Hindi, **route** to the right person with a deadline, and **keep an audit trail**.
- Use an LLM to draft the briefing, but add a **grounding check** (every number must come from the findings) and a human sign-off for anything with legal or safety consequences.
- Domain experts: what does a real inspector ask for? What are the actual current FSSAI, state and GST requirements? Replace our simplified rules with the real ones and cite them.

### Module 6: Reporting and conversational analytics

- Daily owner briefing (sales, food cost %, wastage, top variances, compliance status, three recommended actions).
- "Ask your data": natural language questions to SQL (for example *"Which outlet wasted the most paneer last month?"*). Evaluate correctness against SQL you write by hand, and guard against wrong or unsafe queries.
- Anomaly detection on stock variance and temperature series.

### Module 7: Backend and product

- Extend `backend_skeleton.py`: authentication, per-outlet access, audit log, idempotent webhook, persistence, error handling.
- Design the manager's approval screen (Streamlit or Gradio is fine) and the owner's WhatsApp briefing.
- Deploy a demo (for example on Hugging Face Spaces). Think about the offline case: what happens when the internet drops during dinner service?

---

## 7. Where to use an LLM and where not to

| Need | Best tool | Why |
|---|---|---|
| Parse a Hinglish supplier message | LLM (compare with regex baseline) | Messy language, many variants |
| Read a blurry invoice | OCR + LLM or vision model | Layout and noise |
| Forecast tomorrow's tomato usage | Gradient boosting or time-series model | Numeric, needs calibration and cost-based evaluation |
| Choose order quantities | Optimisation / policy simulation | Constraints and costs; must be auditable |
| Detect a temperature breach | Deterministic rule | Cheap, exact, legally defensible |
| Explain alerts to an owner | LLM (grounded) | Language generation |
| Decide whether to discard food | **Human**, informed by rules | Safety and liability |
| Answer "why did food cost rise?" | LLM over verified numbers (SQL or analysis) | Never let the LLM do the arithmetic from memory |

**Open vs closed models: compare, do not assume.** Consider cost at scale, data privacy (supplier prices and sales are sensitive), latency, reliability, language quality, ability to run on a modest server, and vendor lock-in.

**Guardrails to implement:** JSON schema validation and retries, grounding checks, confidence thresholds with human review, logging of every LLM call, and a regression set of labelled examples you re-run whenever you change a prompt or model.

---

## 8. Evaluation

### Technical

| Area | Metric |
|---|---|
| Forecasting | WAPE vs naive, time-based split, per ingredient class |
| Ordering | Wastage (Rs), stockout penalty (Rs), service level, purchase spend, total cost; replay in the simulator with several seeds |
| Extraction | Field-level accuracy per task and per language style; invalid-JSON rate |
| Compliance | Precision and recall of findings against the planted issues; alert volume per outlet per day |
| Briefing | Grounding (no ungrounded numbers), usefulness rated by a domain expert |

### Business

- Annual savings per outlet and **payback period**.
- Pricing model (per outlet per month? percentage of savings?) and break-even customer count.
- Adoption risk: who has to change behaviour, and what do they lose?

---

## 9. Team structure

Mixed teams of 5 to 6.

| Role | Contribution |
|---|---|
| CS / AI (2) | Models, LLM pipeline, evaluation, backend |
| Finance (1) | Unit economics, working capital, ROI, pricing |
| MBA / operations (1) | Customer discovery, go-to-market, change management |
| Domain expert: hospitality, food technology, supply chain or law (1) | Reality check on recipes, procurement, food-safety and tax rules |
| Flexible (design or data) (1) | UX, dashboard, demo, pitch |

**Rule:** at least one customer conversation (a real restaurant owner, chef or store manager) before the final pitch. Even three short calls will change your design.

---

## 10. Deliverables

1. **Problem canvas** (one page): named customer, rupee cost of the problem, decision your system changes.
2. **Working prototype**: runs end to end on the synthetic data for at least two modules.
3. **Evaluation report**: numbers in rupees against the baselines, plus a stress test.
4. **LLM comparison** (if you used LLMs): at least two models or approaches, with cost, accuracy and failure examples.
5. **Unit economics and pricing** one-pager.
6. **Risk and compliance note**: privacy, fairness to staff, safety liability, regulation, failure modes.
7. **5-minute pitch + live demo.**

### Judging rubric (100 points)

| Criterion | Points |
|---|---|
| Problem understanding and customer insight | 15 |
| Technical quality (validation, baselines, honest metrics) | 20 |
| Smart use of LLMs vs traditional methods, with guardrails | 15 |
| End-to-end working demo | 15 |
| Business case and unit economics | 15 |
| Cross-disciplinary contribution (domain expert and finance visible) | 10 |
| Risk, ethics and realism | 10 |

---

## 11. Quick start

```bash
pip install pandas numpy scikit-learn fastapi uvicorn httpx requests
# optional: pip install anthropic     (closed-source LLM)
# optional: install Ollama and run `ollama pull llama3.1`   (open-source LLM)

python generate_pos_data.py --out data_pos                  # data + historical policy KPIs
python compliance_engine.py --data data_pos                 # rule-based findings
python forecast_and_order_baseline.py --data data_pos       # forecast + policy comparison
python llm_pipeline_template.py --data data_pos --provider mock
export ANTHROPIC_API_KEY=...                                 # for closed-source runs
python llm_pipeline_template.py --data data_pos --provider anthropic --model <model-name> --n 30
DATA_DIR=data_pos uvicorn backend_skeleton:app --reload      # API docs at /docs
```

Use `--seed` to give each team a different world, and `--days` for a longer or shorter history.

---

## 12. Risks and ethics to discuss

- **Food safety and liability.** If the system says "stock is fine" and someone falls ill, who is responsible? AI recommends; a trained human decides.
- **Staff surveillance and fairness.** Variance analysis can wrongly implicate staff. Report patterns and investigate with humans; do not auto-accuse.
- **Privacy and data protection.** Staff certificates and medical fitness are personal data; check the obligations under India's DPDP Act 2023 and sector rules. Supplier prices and sales are commercially sensitive when sent to a third-party LLM.
- **Hallucination.** An invented quantity in a purchase order is real money. Validate, ground and log.
- **Supplier power.** Informal suppliers on WhatsApp may be excluded by a system that demands structured digital orders. How do you onboard them without breaking relationships?
- **Regulation.** Real FSSAI, state, municipal and GST requirements differ from our simplified rules and change over time. Read the primary sources.
- **Dependence.** What happens when the LLM API is down or the internet fails?

---

## 13. Pitch outline (5 minutes)

| Time | Content |
|---|---|
| 0:00 | The customer and the rupee pain (for example "an owner losing Rs 3 lakh per outlet per half-year to wastage") |
| 0:45 | How it is done today and why that fails |
| 1:15 | Your system: what decision changes |
| 1:45 | Live demo (60 seconds): a POS sale, a low-stock alert, an order recommendation, a compliance briefing |
| 2:45 | Proof: numbers against the baseline, one honest limitation |
| 3:30 | LLMs vs traditional ML: what you chose and why |
| 4:00 | Business model, pricing, first 10 customers |
| 4:45 | The ask |

---

## Appendix (instructors only): planted patterns and traps

*Do not share with students before the hackathon ends.*

**Demand**
- Weekends are 20 to 35% busier; Diwali week +25%, Christmas/New Year +20%; heavy rain reduces footfall up to 25%.
- During Navratri, non-vegetarian dishes sell at about half the usual rate (chicken, eggs).
- Delivery orders (30%) get discounts more often.

**Inventory and shrinkage**
- Outlet O2 loses about 12% extra chicken and about 18% extra cashew; outlet O3 about 9% extra cooking oil. All outlets over-use about 4% (portioning). Weekly `stock_counts.csv` shows the signal; random count noise is about 2%.
- Gut-feel policy ignores quantities already on order and pads orders heavily, so late deliveries cause double ordering and over-stocking of perishables.
- Supplier SUP01 (informal veg trader) delivers late about 12% of the time; about 10% of deliveries across suppliers are short (80 to 95% of the order).

**Prices**
- Tomato and coriander prices react strongly to rain over the previous week; onion peaks in Oct/Nov.
- The alternate supplier's price is 0.93 to 1.10 of market (sometimes cheaper); reliability is in `suppliers.csv`, but the order history only uses the primary supplier.

**Compliance**
- O2 cold room has a 12-reading (about 2 day) compressor failure around 29 Sep (**critical**). All outlets have random 1 to 3 reading door-open episodes, which creates many alerts (alert fatigue is a deliberate design challenge).
- Licences: O2 Fire NOC expired 12 days before the end of the data; O3 FSSAI licence expires in 25 days; O1 Health/Trade licence in 40 days.
- About 15 to 20% of staff certificates are expired; supplier SUP06's food licence is expired; SUP01 has no GSTIN or licence on record.
- About 2% of invoices have totals that do not reconcile; about 1% miss the supplier GSTIN (`label_error_type`).

**Text data limitations**
- Supplier messages, voice notes and OCR text are **template-generated**, so a regex baseline scores very high on supplier messages and voice notes. OCR noise (letters replacing digits) is the main difficulty; for example a leading "l." instead of "1." breaks the regex baseline on some invoices. Ask teams to create harder test sets, since the reported LLM-vs-regex gap on easy data says little about real WhatsApp.

**Simplifications**
- Compliance rules are fictional and simplified; GST rates in `ingredients.csv` are illustrative only. Demand is unaffected by stockouts. There is one primary supplier per ingredient in the order history. Encourage teams to challenge all of this.
