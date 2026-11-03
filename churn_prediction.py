
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# 1. Set project paths
BASE_DIR = Path(__file__).resolve().parent

TRANSACTIONS_FILE = (
    BASE_DIR / "cleaned" / "transactions_cleaned.csv"
)
CUSTOMERS_FILE = BASE_DIR / "data" / "customers.csv"
OUTPUT_FILE = BASE_DIR / "cleaned" / "customer_churn_predictions.csv"

# 2. Load data
transactions = pd.read_csv(TRANSACTIONS_FILE)
customers = pd.read_csv(CUSTOMERS_FILE)

transactions["InvoiceDate"] = pd.to_datetime(
    transactions["InvoiceDate"], errors="coerce"
)

transactions["TotalSales"] = pd.to_numeric(
    transactions["TotalSales"], errors="coerce"
)

# 3. Keep completed transactions for purchase behaviour
completed = transactions[
    transactions["OrderStatus"].str.lower() == "completed"
].copy()

# Use a fixed date so the results are reproducible.
# Features: 180 days before 18 July 2026
# Target: purchases during the following 90 days
feature_start = pd.Timestamp("2026-01-19")
snapshot_date = pd.Timestamp("2026-07-18")
label_end = pd.Timestamp("2026-10-16")

feature_data = completed[
    (completed["InvoiceDate"] >= feature_start)
    & (completed["InvoiceDate"] < snapshot_date)
].copy()

future_data = completed[
    (completed["InvoiceDate"] >= snapshot_date)
    & (completed["InvoiceDate"] < label_end)
].copy()

# 4. Create customer behaviour features
features = feature_data.groupby("CustomerID").agg(
    LastPurchase=("InvoiceDate", "max"),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalSales", "sum"),
    AverageOrderValue=("TotalSales", "mean"),
    CategoriesPurchased=("Category", "nunique")
)

features["Recency"] = (
    snapshot_date - features["LastPurchase"]
).dt.days

features = features.drop(columns=["LastPurchase"])

# Include customers with no purchases in the feature period
features = features.reindex(customers["CustomerID"])
features.index.name = "CustomerID"

features["Frequency"] = features["Frequency"].fillna(0)
features["Monetary"] = features["Monetary"].fillna(0)
features["AverageOrderValue"] = (
    features["AverageOrderValue"].fillna(0)
)
features["CategoriesPurchased"] = (
    features["CategoriesPurchased"].fillna(0)
)
features["Recency"] = features["Recency"].fillna(181)

# 5. Create the target using future purchases
future_frequency = future_data.groupby("CustomerID")[
    "InvoiceNo"
].nunique()

features["Churn"] = (
    ~features.index.isin(future_frequency.index)
).astype(int)

# 6. Prepare the model data
X = features.drop(columns=["Churn"])
y = features["Churn"]

if y.nunique() < 2:
    raise ValueError(
        "The dataset must contain both churn and non-churn customers."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 7. Train the model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# 8. Evaluate the model
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nCUSTOMER CHURN MODEL")
print("-" * 40)
print("Total customers:", len(features))
print("Churn customers:", int(y.sum()))
print("Non-churn customers:", int((y == 0).sum()))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

if y_test.nunique() == 2:
    print(
        "ROC-AUC:",
        round(roc_auc_score(y_test, probabilities), 3)
    )

# 9. Predict churn risk for all customers
features["ChurnRisk"] = model.predict_proba(X)[:, 1]

features["RiskLevel"] = pd.cut(
    features["ChurnRisk"],
    bins=[-0.001, 0.30, 0.60, 1.00],
    labels=["Low", "Medium", "High"]
)

features.reset_index().to_csv(OUTPUT_FILE, index=False)

print("\nFeature Importance:")
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importance)

print("\nPredictions saved to:")
print(OUTPUT_FILE)
