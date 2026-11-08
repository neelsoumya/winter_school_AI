"""
baseline_model.py
-----------------
A deliberately simple, readable baseline for the RTO (return-to-origin) case study.
Students: copy this file, change the dataset + columns, keep the structure:

    load data -> split -> pipeline (encode + model) -> metrics -> business cost

Run it directly to see baseline metrics:
    python baseline_model.py
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CAT = ["payment_mode", "category", "courier"]
NUM = ["order_value", "discount_pct", "city_tier", "is_first_order", "past_orders",
       "address_quality", "delivery_distance_km", "order_hour"]
TARGET = "rto"


def rto_cost(forward_shipping_cost, handling=30):
    """Rupees lost if a parcel is returned: we pay shipping twice + handling."""
    return 2 * forward_shipping_cost + handling


def train_rto_model(csv_path="data/rto_orders.csv", seed=42):
    df = pd.read_csv(csv_path)
    X, y = df[CAT + NUM], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y,
                                              random_state=seed)
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", "passthrough", NUM),
    ])
    model = Pipeline([("prep", prep),
                      ("clf", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                                             random_state=seed))])
    model.fit(X_tr, y_tr)

    p = model.predict_proba(X_te)[:, 1]
    metrics = {
        "roc_auc": roc_auc_score(y_te, p),
        "pr_auc": average_precision_score(y_te, p),   # better than accuracy when classes are imbalanced
        "base_rate": float(y_te.mean()),
    }
    return model, metrics


def predict_one(model, row: dict) -> float:
    """Return P(RTO) for a single order given as a dict of feature values."""
    return float(model.predict_proba(pd.DataFrame([row])[CAT + NUM])[:, 1][0])


if __name__ == "__main__":
    m, metrics = train_rto_model()
    print({k: round(v, 3) for k, v in metrics.items()})
