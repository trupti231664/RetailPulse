
# ============================================================
# RETAILPULSE - EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- FILE PATHS -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CLEAN_DIR = BASE_DIR / "cleaned"
ANALYSIS_DIR = BASE_DIR / "analysis"
CHART_DIR = ANALYSIS_DIR / "charts"

ANALYSIS_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)

# -------------------- LOAD DATA ------------------------------

print("\n========== RETAILPULSE EDA ==========\n")

transactions = pd.read_csv(
    CLEAN_DIR / "transactions_cleaned.csv",
    parse_dates=["InvoiceDate"]
)

products = pd.read_csv(DATA_DIR / "products.csv")
customers = pd.read_csv(DATA_DIR / "customers.csv")

print("DATA LOADED SUCCESSFULLY")
print("Transactions:", len(transactions))
print("Products:", len(products))
print("Customers:", len(customers))

# -------------------- MERGE DATA -----------------------------

# Remove category/name columns from transactions before merging
df = transactions.drop(
    columns=["Category", "ProductName"],
    errors="ignore"
).merge(
    products[["ProductID", "Category", "ProductName"]],
    on="ProductID",
    how="left",
    validate="many_to_one"
)

# Merge customer information if CustomerID exists
customer_columns = [
    col for col in ["CustomerID", "Gender", "Age", "City",
                    "CustomerSegment"]
    if col in customers.columns
]

if "CustomerID" in customer_columns:
    df = df.merge(
        customers[customer_columns].drop_duplicates("CustomerID"),
        on="CustomerID",
        how="left",
        validate="many_to_one"
    )

# Ensure numeric columns are numeric
for col in ["Quantity", "UnitPrice", "TotalSales",
            "DiscountPercent"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Create useful date columns
df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.month
df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
df["Weekday"] = df["InvoiceDate"].dt.day_name()

# -------------------- DATA QUALITY ---------------------------

print("\n========== 1. DATA QUALITY ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Duplicate rows:", df.duplicated().sum())
print("Missing values:", df.isna().sum().sum())

quality_report = pd.DataFrame({
    "MissingValues": df.isna().sum(),
    "MissingPercent": (
        df.isna().mean() * 100
    ).round(2),
    "DataType": df.dtypes.astype(str)
})

quality_report.to_csv(
    ANALYSIS_DIR / "data_quality_report.csv"
)

# -------------------- OVERALL SALES --------------------------

print("\n========== 2. OVERALL SALES ==========")

total_sales = df["TotalSales"].sum()
average_sales = df["TotalSales"].mean()
total_quantity = df["Quantity"].sum()
unique_invoices = df["InvoiceNo"].nunique()

print(f"Total recorded sales: {total_sales:,.2f}")
print(f"Average transaction row sales: {average_sales:,.2f}")
print(f"Total net quantity across rows: {total_quantity:,.0f}")
print(f"Unique invoice numbers: {unique_invoices:,}")

summary = pd.DataFrame({
    "Metric": [
        "Transaction Rows",
        "Unique Invoices",
        "Total Recorded Sales",
        "Average Sales Per Row",
        "Total Quantity"
    ],
    "Value": [
        len(df),
        unique_invoices,
        total_sales,
        average_sales,
        total_quantity
    ]
})

summary.to_csv(
    ANALYSIS_DIR / "overall_sales_summary.csv",
    index=False
)

# -------------------- CATEGORY ANALYSIS ----------------------

print("\n========== 3. SALES BY CATEGORY ==========")

category_sales = (
    df.groupby("Category", dropna=False)
    .agg(
        TotalSales=("TotalSales", "sum"),
        UnitsSold=("Quantity", "sum"),
        TransactionRows=("ProductID", "size")
    )
    .sort_values("TotalSales", ascending=False)
)

print(category_sales.to_string())

category_sales.to_csv(
    ANALYSIS_DIR / "sales_by_category.csv"
)

plt.figure(figsize=(12, 7))
category_sales["TotalSales"].sort_values().plot(kind="barh")
plt.title("Total Recorded Sales by Product Category")
plt.xlabel("Recorded Sales")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(CHART_DIR / "sales_by_category.png", dpi=150)
plt.close()

# -------------------- MONTHLY SALES TREND --------------------

print("\n========== 4. MONTHLY SALES TREND ==========")

monthly_sales = (
    df.groupby("YearMonth")
    .agg(
        TotalSales=("TotalSales", "sum"),
        UnitsSold=("Quantity", "sum"),
        TransactionRows=("ProductID", "size")
    )
    .reset_index()
)

print(monthly_sales.head(12).to_string(index=False))

monthly_sales.to_csv(
    ANALYSIS_DIR / "monthly_sales.csv",
    index=False
)

plt.figure(figsize=(15, 6))
plt.plot(
    monthly_sales["YearMonth"],
    monthly_sales["TotalSales"],
    marker="o",
    markersize=2
)
plt.title("Monthly Recorded Sales Trend")
plt.xlabel("Year-Month")
plt.ylabel("Recorded Sales")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(CHART_DIR / "monthly_sales_trend.png", dpi=150)
plt.close()

# -------------------- TOP 10 PRODUCTS ------------------------

print("\n========== 5. TOP 10 PRODUCTS BY QUANTITY ==========")

top_products = (
    df.groupby(["ProductID", "ProductName"], dropna=False)
    .agg(
        UnitsSold=("Quantity", "sum"),
        TotalSales=("TotalSales", "sum")
    )
    .sort_values("UnitsSold", ascending=False)
    .head(10)
)

print(top_products.to_string())

top_products.to_csv(
    ANALYSIS_DIR / "top_10_products.csv"
)

plt.figure(figsize=(12, 7))
top_products["UnitsSold"].sort_values().plot(kind="barh")
plt.title("Top 10 Products by Net Quantity")
plt.xlabel("Net Quantity")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig(CHART_DIR / "top_10_products.png", dpi=150)
plt.close()

# -------------------- SALES BY YEAR --------------------------

print("\n========== 6. SALES BY YEAR ==========")

yearly_sales = (
    df.groupby("Year")
    .agg(
        TotalSales=("TotalSales", "sum"),
        UnitsSold=("Quantity", "sum"),
        TransactionRows=("ProductID", "size")
    )
    .reset_index()
)

print(yearly_sales.to_string(index=False))

yearly_sales.to_csv(
    ANALYSIS_DIR / "yearly_sales.csv",
    index=False
)

plt.figure(figsize=(10, 6))
sns.barplot(data=yearly_sales, x="Year", y="TotalSales")
plt.title("Total Recorded Sales by Year")
plt.tight_layout()
plt.savefig(CHART_DIR / "yearly_sales.png", dpi=150)
plt.close()

# -------------------- DISCOUNT ANALYSIS ----------------------

print("\n========== 7. DISCOUNT ANALYSIS ==========")

if "DiscountPercent" in df.columns:
    discount_analysis = (
        df.groupby("DiscountPercent", dropna=False)
        .agg(
            TotalSales=("TotalSales", "sum"),
            TransactionRows=("ProductID", "size")
        )
        .reset_index()
        .sort_values("DiscountPercent")
    )

    print(discount_analysis.head(15).to_string(index=False))

    discount_analysis.to_csv(
        ANALYSIS_DIR / "discount_analysis.csv",
        index=False
    )

# -------------------- ORDER STATUS ---------------------------

print("\n========== 8. ORDER STATUS ==========")

if "OrderStatus" in df.columns:
    order_status = (
        df.groupby("OrderStatus", dropna=False)
        .agg(
            TransactionRows=("ProductID", "size"),
            TotalSales=("TotalSales", "sum")
        )
        .sort_values("TransactionRows", ascending=False)
    )

    print(order_status.to_string())

    order_status.to_csv(
        ANALYSIS_DIR / "order_status_analysis.csv"
    )

# -------------------- CUSTOMER ANALYSIS ----------------------

print("\n========== 9. CUSTOMER ANALYSIS ==========")

if "CustomerID" in df.columns:
    customer_analysis = (
        df.groupby("CustomerID")
        .agg(
            TransactionRows=("ProductID", "size"),
            TotalSales=("TotalSales", "sum"),
            TotalQuantity=("Quantity", "sum")
        )
        .sort_values("TotalSales", ascending=False)
    )

    print("Top 10 customers by recorded sales:")
    print(customer_analysis.head(10).to_string())

    customer_analysis.to_csv(
        ANALYSIS_DIR / "customer_analysis.csv"
    )

# -------------------- WEEKDAY ANALYSIS -----------------------

print("\n========== 10. SALES BY WEEKDAY ==========")

weekday_sales = (
    df.groupby("Weekday")
    .agg(
        TotalSales=("TotalSales", "sum"),
        TransactionRows=("ProductID", "size")
    )
    .reset_index()
)

weekday_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

weekday_sales["Weekday"] = pd.Categorical(
    weekday_sales["Weekday"],
    categories=weekday_order,
    ordered=True
)

weekday_sales = weekday_sales.sort_values("Weekday")
weekday_sales.to_csv(
    ANALYSIS_DIR / "weekday_sales.csv",
    index=False
)

plt.figure(figsize=(10, 6))
sns.barplot(data=weekday_sales, x="Weekday", y="TotalSales")
plt.title("Total Recorded Sales by Weekday")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(CHART_DIR / "weekday_sales.png", dpi=150)
plt.close()

# -------------------- FINISH ---------------------------------

print("\n========== EDA COMPLETED ==========")
print("Analysis results saved in:", ANALYSIS_DIR)
print("Charts saved in:", CHART_DIR)
print("All done!")



