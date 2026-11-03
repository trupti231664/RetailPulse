
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Find the project folder and load cleaned data
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "cleaned" / "transactions_cleaned.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# 2. Check the required columns
required_columns = [
    "CustomerID",
    "InvoiceNo",
    "InvoiceDate",
    "TotalSales",
    "OrderStatus"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# 3. Prepare the data
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

df["TotalSales"] = pd.to_numeric(
    df["TotalSales"], errors="coerce"
)

df = df.dropna(
    subset=[
        "CustomerID",
        "InvoiceNo",
        "InvoiceDate",
        "TotalSales"
    ]
).copy()

# Use completed transactions for customer purchasing behaviour
df = df[
    df["OrderStatus"].astype(str).str.strip().str.lower()
    == "completed"
].copy()

if df.empty:
    raise ValueError("No completed transactions are available.")

# 4. Calculate RFM values
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda dates:
             (snapshot_date - dates.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalSales", "sum")
).reset_index()

# 5. Keep customers with valid positive spending
rfm = rfm[
    (rfm["Frequency"] > 0) &
    (rfm["Monetary"] > 0)
].copy()

if len(rfm) < 4:
    raise ValueError(
        "At least four eligible customers are needed for this clustering."
    )

# 6. Prepare RFM features for machine learning
features = rfm[["Recency", "Frequency", "Monetary"]].copy()

# Reduce the effect of very large values
features["Recency"] = features["Recency"].astype(float)
features["Frequency"] = features["Frequency"].astype(float)
features["Monetary"] = features["Monetary"].astype(float)

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 7. Apply K-Means clustering
model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = model.fit_predict(scaled_features)

# 8. Save customer segmentation results
OUTPUT_DIR = PROJECT_DIR / "cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "customer_segments.csv"
rfm.to_csv(OUTPUT_PATH, index=False)

# 9. Display the results
print("\nCUSTOMER SEGMENTATION COMPLETED")
print("--------------------------------")
print(f"Customers analysed: {len(rfm):,}")
print(f"Number of clusters: {rfm['Cluster'].nunique()}")

print("\nCustomers in each cluster:")
print(rfm["Cluster"].value_counts().sort_index())

print("\nAverage RFM values by cluster:")
print(
    rfm.groupby("Cluster")[
        ["Recency", "Frequency", "Monetary"]
    ].mean().round(2)
)

print(f"\nResults saved to: {OUTPUT_PATH}")
print("\nFirst five customer records:")
print(rfm.head())
