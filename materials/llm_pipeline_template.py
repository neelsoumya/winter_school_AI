"""
llm_pipeline_template.py
------------------------
Where LLMs earn their place in the hospitality backend, and where they do not.

  Task A  Extract structured data from messy text (supplier WhatsApp, OCR invoices, kitchen voice notes)
  Task B  Turn deterministic compliance findings into a prioritised manager briefing (with a grounding check)

Providers (swap with --provider):
  mock       rule-based extractor, runs offline: this is the BASELINE your LLM has to beat
  anthropic  closed-source API   (needs `pip install anthropic` and ANTHROPIC_API_KEY; model via --model)
  ollama     open-source local   (needs Ollama running, e.g. `ollama pull llama3.1`; model via --model)

    python generate_pos_data.py --out data_pos
    python llm_pipeline_template.py --data data_pos --provider mock
    python llm_pipeline_template.py --data data_pos --provider anthropic --model claude-sonnet-5-5 --n 30
    python llm_pipeline_template.py --data data_pos --provider ollama --model llama3.1 --n 30

Design rule: LLMs read and write language; rules and ML make the decisions that cost money.
"""
import argparse
import json
import os
import re

import pandas as pd

from compliance_engine import run_all

# ----------------------------------------------------------------------------
# Provider adapters: each returns complete(prompt) -> str
# ----------------------------------------------------------------------------
def get_complete_fn(provider, model):
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY

        def complete(prompt):
            r = client.messages.create(model=model, max_tokens=1000, temperature=0,
                                       messages=[{"role": "user", "content": prompt}])
            return r.content[0].text
        return complete
    if provider == "ollama":
        import requests

        def complete(prompt):
            r = requests.post("http://localhost:11434/api/generate", timeout=120,
                              json={"model": model, "prompt": prompt, "stream": False,
                                    "options": {"temperature": 0}})
            return r.json()["response"]
        return complete
    return None   # mock


def parse_json(text):
    """LLMs wrap JSON in prose or code fences. Be tolerant, but return None when it really fails."""
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Task A: extraction prompts (LLM) and rule-based mock (baseline)
# ----------------------------------------------------------------------------
INGREDIENTS = ["tomato", "onion", "potato", "green_chilli", "coriander", "spinach", "cauliflower",
               "ginger_garlic_paste", "paneer", "milk", "curd", "butter", "cooking_cream", "chicken", "eggs",
               "basmati_rice", "atta", "toor_dal", "cooking_oil", "ghee", "cashew", "red_chilli_powder", "garam_masala"]

PROMPTS = {
    "supplier_message": """You read WhatsApp messages from vegetable/dairy/meat suppliers to a restaurant (English or Hinglish).
Return ONLY JSON. Allowed ingredient values: {ings}.
Schema: {{"intent": "price_quote|delay|shortage|substitution|dispatch", "ingredient": "<id>",
 "price": number|null, "delivery_by": "HH:MM"|null, "delay_hours": number|null,
 "available_qty": number|null, "substitute": "<id>"|null, "qty": number|null, "invoice_no": "string"|null}}
For substitution, "ingredient" is the item that is NOT available. Use null for fields that do not apply.
Message: {text}""",
    "delivery_note": """Below is noisy OCR text of a supplier invoice (letters may replace digits, e.g. O for 0, l for 1, S for 5).
Return ONLY JSON: {{"invoice_no": "string", "items": [{{"ingredient": "<id>", "qty": number, "rate": number, "amount": number}}], "total_before_gst": number}}
Allowed ingredient values: {ings}. Fix OCR digit errors.
Text:
{text}""",
    "voice_note": """A kitchen staff member dictated a closing stock count (English/Hinglish, numbers may be words).
Return ONLY JSON: {{"counts": {{"<ingredient id>": number}}}}. Allowed ingredient ids: {ings}.
Text: {text}""",
}

LEX = {"tomato": ["tomato", "tamatar"], "onion": ["onion", "pyaaz"], "potato": ["potato", "aloo"],
       "green_chilli": ["green chilli", "hari mirch"], "coriander": ["coriander", "dhaniya"],
       "spinach": ["spinach", "palak"], "cauliflower": ["cauliflower", "gobi"], "paneer": ["paneer"],
       "milk": ["milk", "doodh"], "curd": ["curd", "dahi"], "chicken": ["chicken", "murgi"], "eggs": ["eggs", "anda"],
       "cooking_oil": ["cooking oil", "tel"], "basmati_rice": ["basmati rice", "chawal"], "atta": ["atta"],
       "butter": ["butter"], "cooking_cream": ["cooking cream"], "ginger_garlic_paste": ["ginger garlic paste"],
       "toor_dal": ["toor dal"], "ghee": ["ghee"], "cashew": ["cashew"], "red_chilli_powder": ["red chilli powder"],
       "garam_masala": ["garam masala"]}
NUMW = {"one": 1, "ek": 1, "two": 2, "do": 2, "three": 3, "teen": 3, "four": 4, "chaar": 4, "five": 5, "paanch": 5,
        "six": 6, "chhe": 6, "seven": 7, "saat": 7, "eight": 8, "aath": 8, "nine": 9, "nau": 9, "ten": 10, "das": 10, "twelve": 12, "barah": 12}


def find_ings(text):
    t, hits = text.lower(), []
    for ing, words in LEX.items():
        for w in words:
            for m in re.finditer(r"\b" + re.escape(w) + r"\b", t):
                hits.append((m.start(), ing))
    return [i for _, i in sorted(hits)]


def to_num(tok):
    tok = tok.replace("O", "0").replace("o", "0").replace("l", "1").replace("S", "5").replace("B", "8")
    try:
        return float(tok)
    except ValueError:
        return None


class MockExtractor:
    """Regex + keyword baseline. Cheap and surprisingly decent. If your LLM cannot beat this, rethink."""

    def supplier_message(self, text):
        t, nums, ings = text.lower(), [int(x) for x in re.findall(r"\d+", text)], find_ings(text)
        if "dispatched" in t or "bhej diya" in t: intent = "dispatch"
        elif "nahi hai" in t or "can send" in t or "no " in t[:3]: intent = "substitution"
        elif any(k in t for k in ["kam hai", "short supply", "sirf", "can only", "mil raha"]): intent = "shortage"
        elif any(k in t for k in ["late", "delayed", "ghant"]): intent = "delay"
        else: intent = "price_quote"
        out = {"intent": intent, "ingredient": ings[0] if ings else None}
        if intent == "price_quote" and nums: out.update(price=nums[0], delivery_by=f"{nums[-1]:02d}:00")
        elif intent == "delay" and nums: out["delay_hours"] = nums[0]
        elif intent == "shortage" and nums: out["available_qty"] = nums[0]
        elif intent == "substitution": out.update(substitute=ings[1] if len(ings) > 1 else None, price=nums[-1] if nums else None)
        elif intent == "dispatch" and nums: out.update(qty=nums[0], invoice_no=str(nums[-1]))
        return out

    def delivery_note(self, text):
        items, inv, total = [], None, None
        for line in text.split("\n"):
            m = re.match(r"^\d+\.\s+(.+?)\s+(\S+)\s+(KG|L|PIECE)\s+@\s+(\S+)\s+=\s+(\S+)$", line)
            if m:
                ing = find_ings(m.group(1))
                items.append({"ingredient": ing[0] if ing else None, "qty": to_num(m.group(2)),
                              "rate": to_num(m.group(4)), "amount": to_num(m.group(5))})
            if line.startswith("INV NO:"):
                inv = "INV-" + re.sub(r"\D", lambda c: c.group(), line.split("INV-")[-1]).replace("O", "0").replace("l", "1").replace("S", "5").replace("B", "8")
            if line.startswith("TOTAL"):
                total = to_num(line.split()[-1])
        return {"invoice_no": inv, "items": items, "total_before_gst": total}

    def voice_note(self, text):
        counts = {}
        for part in text.split(":", 1)[-1].split(","):
            ings = find_ings(part)
            num = next((float(x) if x.isdigit() else NUMW[x] for x in part.lower().split() if x.isdigit() or x in NUMW), None)
            if ings and num is not None:
                counts[ings[0]] = num
        return {"counts": counts}


class LLMExtractor:
    def __init__(self, complete):
        self.complete = complete

    def _run(self, task, text):
        return parse_json(self.complete(PROMPTS[task].format(ings=", ".join(INGREDIENTS), text=text))) or {}

    def supplier_message(self, t): return self._run("supplier_message", t)
    def delivery_note(self, t): return self._run("delivery_note", t)
    def voice_note(self, t): return self._run("voice_note", t)


# ----------------------------------------------------------------------------
# Evaluation against ground-truth labels
# ----------------------------------------------------------------------------
def close(a, b):
    try:
        return abs(float(a) - float(b)) < 0.51
    except (TypeError, ValueError):
        return a == b


def score(task, pred, gold):
    """Fraction of gold fields recovered correctly (field-level accuracy)."""
    if task == "supplier_message":
        keys = [k for k in gold if k != "unit"]
        return sum(close(pred.get(k), gold[k]) for k in keys) / len(keys)
    if task == "delivery_note":
        pi = {i.get("ingredient"): i for i in pred.get("items", [])}
        fields = [close(pred.get("invoice_no"), gold["invoice_no"]), close(pred.get("total_before_gst"), gold["total_before_gst"])]
        for it in gold["items"]:
            p = pi.get(it["ingredient"], {})
            fields += [close(p.get("qty"), it["qty"]), close(p.get("rate"), it["rate"])]
        return sum(fields) / len(fields)
    pc, gc = pred.get("counts", {}), gold["counts"]
    return sum(close(pc.get(k), v) for k, v in gc.items()) / len(gc)


def evaluate(ex, data, n):
    res = {}
    for task, fname in [("supplier_message", "supplier_messages.jsonl"), ("delivery_note", "delivery_notes_ocr.jsonl"),
                        ("voice_note", "kitchen_voice_notes.jsonl")]:
        rows = [json.loads(l) for l in open(os.path.join(data, fname), encoding="utf-8")][:n]
        scores, fails = [], []
        for r in rows:
            try:
                p = getattr(ex, task)(r["text"])
            except Exception as e:           # API errors, timeouts
                p = {}
            s = score(task, p, r["label"])
            scores.append(s)
            if s < 1 and len(fails) < 2:
                fails.append((r["text"][:90].replace("\n", " | "), p, r["label"]))
        res[task] = (sum(scores) / len(scores), fails)
    return res


# ----------------------------------------------------------------------------
# Task B: compliance briefing (rules find, LLM explains) + grounding check
# ----------------------------------------------------------------------------
BRIEF_PROMPT = """You are the compliance assistant for a restaurant group. Using ONLY the findings below
(and the rule IDs), write a morning briefing for the owner: at most 180 words, ordered by urgency, each
point with the outlet, what is wrong, and one concrete action. Do not invent numbers, dates or outlets.

FINDINGS:
{findings}

RULES (excerpt):
{rules}
"""


def numbers_in(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def ungrounded_numbers(output, source):
    """Guardrail: any number in the LLM output that does not appear in the source is suspect."""
    return sorted(numbers_in(output) - numbers_in(source))


def daily_briefing(findings, rules_md, complete=None, top=15):
    sev = {"critical": 0, "high": 1, "medium": 2}
    f = findings.sort_values(by="severity", key=lambda s: s.map(sev)).head(top)
    src = "\n".join(f"[{r.rule}/{r.severity}] {r.outlet_id} {r.subject}: {r.detail}" for r in f.itertuples())
    if complete is None:    # mock: deterministic template
        text = "Morning compliance briefing:\n" + "\n".join(f"- ({r.severity.upper()}) {r.outlet_id}: {r.subject}. {r.detail}." for r in f.itertuples())
    else:
        text = complete(BRIEF_PROMPT.format(findings=src, rules=rules_md[:1800]))
    return text, ungrounded_numbers(text, src + rules_md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_pos")
    ap.add_argument("--provider", default="mock", choices=["mock", "anthropic", "ollama"])
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "claude-sonnet-5-5"))
    ap.add_argument("--n", type=int, default=40, help="examples per extraction task")
    a = ap.parse_args()

    complete = get_complete_fn(a.provider, a.model)
    ex = MockExtractor() if complete is None else LLMExtractor(complete)

    print(f"== Task A: extraction accuracy ({a.provider}) ==")
    for task, (acc, fails) in evaluate(ex, a.data, a.n).items():
        print(f"{task:18s} field accuracy = {acc:.1%}")
        for text, pred, gold in fails[:1]:
            print(f"   e.g. '{text}'\n        pred={pred}\n        gold={gold}")

    print("\n== Task B: compliance briefing ==")
    findings, today = run_all(a.data)
    rules = open(os.path.join(a.data, "compliance_rules_simplified.md"), encoding="utf-8").read()
    text, bad = daily_briefing(findings, rules, complete)
    print(text)
    print("\nUngrounded numbers (possible hallucinations):", bad or "none")
