
from pathlib import Path
import pandas as pd

# Keep raw data unchanged
RAW_DIR = Path("data")
CLEAN_DIR = Path("cleaned")
CLEAN_DIR.mkdir(exist_ok=True)

# Load raw transactions
df = pd.read_csv(RAW_DIR / "transactions.csv")

print("RAW DATA SUMMARY")
print("Rows:", len(df))
print("Exact duplicate rows:", df.duplicated().sum())
print("Missing values:", df.isna().sum().sum())

# 1. Remove exact duplicate rows
df = df.drop_duplicates().copy()

# 2. Remove rows missing essential identifiers
df = df.dropna(
    subset=["InvoiceNo", "CustomerID", "ProductID", "InvoiceDate"]
).copy()

# 3. Convert columns to appropriate data types
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

for col in ["Quantity", "UnitPrice", "DiscountPercent", "TotalSales"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. Remove rows with invalid essential numeric values
df = df.dropna(
    subset=["InvoiceDate", "Quantity", "UnitPrice",
            "DiscountPercent", "TotalSales"]
).copy()

# 5. Fill missing PaymentMethod values
df["PaymentMethod"] = df["PaymentMethod"].fillna(
    df["PaymentMethod"].mode()[0]
)

# Keep negative quantities for returns.
# Remove zero quantities and negative unit prices.
df = df[
    (df["Quantity"] != 0) &
    (df["UnitPrice"] >= 0) &
    (df["DiscountPercent"].between(0, 100))
].copy()

# 5. Save cleaned transactions
df.to_csv(CLEAN_DIR / "transactions_cleaned.csv", index=False)

print("\nCLEANING SUMMARY")
print("Rows after cleaning:", len(df))
print("Duplicates remaining:", df.duplicated().sum())
print("Missing values remaining:", df.isna().sum().sum())
print("Saved to:", CLEAN_DIR / "transactions_cleaned.csv")
