import random
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# RETAILPULSE: BEHAVIOUR-BASED SYNTHETIC RETAIL DATA
# This imitates retail patterns; it is not real company data.
# =========================================================

SEED = 20261017
random.seed(SEED)
np.random.seed(SEED)

CUSTOMER_COUNT = 5_000
PRODUCT_COUNT = 1_000
START_DATE = pd.Timestamp("2022-01-01")
END_DATE = pd.Timestamp("2026-10-15")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# The churn-model label asks: did the customer make a completed
# purchase during the 90 days after the snapshot date?
SNAPSHOT_DATE = END_DATE - pd.Timedelta(days=90)
FEATURE_START = SNAPSHOT_DATE - pd.Timedelta(days=180)

products_by_category = {
    "Electronics": ["Wireless Mouse", "Bluetooth Speaker", "Keyboard", "Laptop Stand", "Webcam", "Headphones", "Power Bank"],
    "Clothing": ["Cotton T-Shirt", "Denim Jeans", "Casual Shirt", "Summer Dress", "Hoodie", "Formal Trousers"],
    "Footwear": ["Running Shoes", "Casual Sneakers", "Sandals", "Formal Shoes", "Slippers"],
    "Groceries": ["Rice Pack", "Wheat Flour", "Tea Pack", "Coffee Powder", "Cooking Oil", "Snack Pack"],
    "Beauty and Personal Care": ["Face Wash", "Body Lotion", "Shampoo", "Moisturizer", "Sunscreen", "Hair Oil"],
    "Home and Living": ["Ceramic Mug", "Storage Basket", "Table Lamp", "Cushion Cover", "Wall Clock", "Bedsheet Set"],
    "Sports and Fitness": ["Yoga Mat", "Resistance Band", "Skipping Rope", "Gym Gloves", "Water Shaker"],
    "Books": ["Python Programming Book", "Business Book", "Data Science Handbook", "Novel", "Cookbook"],
    "Stationery": ["Notebook", "Ball Pen Set", "Marker Set", "Office Diary", "Sticky Notes"],
    "Baby Products": ["Baby Blanket", "Feeding Bottle", "Baby Shampoo", "Diaper Pack", "Baby Carrier"],
    "Garden and Outdoor": ["Plant Pot", "Garden Gloves", "Watering Can", "Garden Hose", "Garden Tool Set"],
    "Automotive Accessories": ["Car Phone Holder", "Seat Cover", "Car Charger", "Car Cleaning Kit", "Bike Helmet"],
    "Travel Accessories": ["Travel Backpack", "Suitcase", "Travel Pillow", "Passport Holder", "Packing Cubes"],
    "Jewellery and Accessories": ["Necklace", "Silver Earrings", "Bracelet", "Pendant", "Fashion Ring", "Watch"],
    "Toys and Games": ["Puzzle Set", "Building Blocks", "Board Game", "Soft Toy", "Educational Toy"],
    "Pet Supplies": ["Dog Food", "Cat Food", "Pet Bed", "Pet Collar", "Pet Toy"],
    "Kitchen Appliances": ["Electric Kettle", "Mixer Grinder", "Toaster", "Hand Blender", "Rice Cooker"],
    "Mobile Accessories": ["Phone Case", "Screen Protector", "Wireless Charger", "Phone Holder", "Memory Card"],
    "Furniture": ["Office Chair", "Study Table", "Bookshelf", "Coffee Table", "Wardrobe"],
    "Health and Wellness": ["Heating Pad", "Digital Thermometer", "First Aid Kit", "Massage Roller", "Yoga Block"],
    "Office Supplies": ["Printer Paper", "Calculator", "Desk Organizer", "File Folder", "Label Printer"],
    "Home Decor": ["Wall Art", "Photo Frame", "Artificial Plant", "Decorative Lights", "Curtains"],
    "Food and Bakery": ["Cookies Box", "Bread Loaf", "Muffin Pack", "Biscuit Pack", "Cake Box"],
    "Beverages": ["Green Tea", "Fruit Juice", "Mineral Water", "Cold Coffee", "Coffee Can"],
    "Hardware and Tools": ["Screwdriver Set", "Hammer", "Measuring Tape", "Tool Box", "Wrench Set"],
    "Electrical Supplies": ["LED Bulb", "Extension Board", "Power Strip", "Electrical Tape", "Smart Plug"],
    "Arts and Crafts": ["Paint Brush Set", "Watercolor Set", "Sketchbook", "Craft Paper", "Acrylic Paint"],
    "Party Supplies": ["Birthday Balloons", "Gift Wrap", "Party Banner", "Paper Plates", "Gift Bag"],
    "Musical Instruments": ["Acoustic Guitar", "Keyboard Piano", "Flute", "Ukulele", "Drumsticks"],
    "Eco-Friendly Products": ["Reusable Shopping Bag", "Bamboo Toothbrush", "Steel Straw Set", "Compost Bin", "Reusable Lunch Box"],
}

price_ranges = {
    "Electronics": (199, 250000), "Clothing": (199, 50000), "Footwear": (299, 70000),
    "Groceries": (20, 10500), "Beauty and Personal Care": (79, 25000), "Home and Living": (99, 50000),
    "Sports and Fitness": (150, 60000), "Books": (100, 30000), "Stationery": (10, 8000),
    "Baby Products": (99, 50000), "Garden and Outdoor": (50, 80000), "Automotive Accessories": (100, 120000),
    "Travel Accessories": (300, 150000), "Jewellery and Accessories": (99, 200000), "Toys and Games": (99, 40000),
    "Pet Supplies": (100, 60000), "Kitchen Appliances": (500, 150000), "Mobile Accessories": (49, 50000),
    "Furniture": (1000, 500000), "Health and Wellness": (100, 70000), "Office Supplies": (30, 100000),
    "Home Decor": (99, 100000), "Food and Bakery": (20, 20000), "Beverages": (20, 10000),
    "Hardware and Tools": (50, 100000), "Electrical Supplies": (30, 80000), "Arts and Crafts": (30, 50000),
    "Party Supplies": (20, 30000), "Musical Instruments": (500, 500000), "Eco-Friendly Products": (50, 50000),
}

# 1. Build product catalogue
category_names = list(products_by_category)
product_rows = []

for i in range(PRODUCT_COUNT):
    category = category_names[i % len(category_names)]
    product_name = random.choice(products_by_category[category])
    low, high = price_ranges[category]
    # Log-uniform pricing creates more low/mid-price products than extreme prices.
    unit_price = round(float(np.exp(np.random.uniform(np.log(low), np.log(high)))), 2)
    unit_cost = round(unit_price * random.uniform(0.45, 0.80), 2)

    product_rows.append({
        "ProductID": f"PRD{i + 1:05d}",
        "ProductName": f"{product_name} Model {i + 1:04d}",
        "Category": category,
        "Brand": f"Brand{random.randint(1, 80):02d}",
        "UnitPrice": unit_price,
        "UnitCost": unit_cost,
        "ProfitPerUnit": round(unit_price - unit_cost, 2),
        "SupplierLeadDays": random.randint(1, 30),
    })

products_df = pd.DataFrame(product_rows)
product_lookup = products_df.set_index("ProductID").to_dict("index")
products_by_cat = {
    cat: group["ProductID"].tolist()
    for cat, group in products_df.groupby("Category")
}

# 2. Build customers with different shopping profiles
first_names = ["Aarav", "Aditi", "Aisha", "Ananya", "Arjun", "Diya", "Ishaan", "Kabir", "Meera", "Neha",
               "Priya", "Rahul", "Riya", "Rohan", "Sana", "Sneha", "Tanvi", "Varun", "Vikram", "Zoya"]
last_names = ["Sharma", "Patel", "Shah", "Verma", "Singh", "Joshi", "Kulkarni", "Mehta", "Gupta", "Rao",
              "Khan", "Desai", "Pawar", "Iyer", "Nair"]
cities = [
    ("Pune", "Maharashtra"), ("Mumbai", "Maharashtra"), ("Nagpur", "Maharashtra"),
    ("Nashik", "Maharashtra"), ("Bengaluru", "Karnataka"), ("Mysuru", "Karnataka"),
    ("Hyderabad", "Telangana"), ("Chennai", "Tamil Nadu"), ("Coimbatore", "Tamil Nadu"),
    ("Delhi", "Delhi"), ("Jaipur", "Rajasthan"), ("Ahmedabad", "Gujarat"),
    ("Surat", "Gujarat"), ("Kolkata", "West Bengal"), ("Kochi", "Kerala"),
    ("Indore", "Madhya Pradesh"), ("Bhopal", "Madhya Pradesh"), ("Lucknow", "Uttar Pradesh"),
    ("Chandigarh", "Chandigarh"), ("Bhubaneswar", "Odisha")
]

# Each profile affects purchase frequency and discount preference.
profiles = {
    "Loyal": {"weight": 0.22, "monthly_orders": 2.8, "churn_chance": 0.08, "discount": "normal"},
    "Regular": {"weight": 0.38, "monthly_orders": 1.5, "churn_chance": 0.22, "discount": "normal"},
    "Occasional": {"weight": 0.25, "monthly_orders": 0.55, "churn_chance": 0.38, "discount": "normal"},
    "Discount Seeker": {"weight": 0.10, "monthly_orders": 1.0, "churn_chance": 0.32, "discount": "high"},
    "At Risk": {"weight": 0.05, "monthly_orders": 0.8, "churn_chance": 0.78, "discount": "normal"},
}
profile_names = list(profiles)
profile_weights = [profiles[p]["weight"] for p in profile_names]

customer_rows = []
customer_behaviour = {}

for i in range(CUSTOMER_COUNT):
    customer_id = f"CUST{i + 1:05d}"
    city, state = random.choice(cities)
    profile = random.choices(profile_names, weights=profile_weights, k=1)[0]
    signup_date = START_DATE - pd.Timedelta(days=random.randint(1, 900))
    preferred_category = random.choice(category_names)

    # A portion of customers stops buying before the dataset end.
    if random.random() < profiles[profile]["churn_chance"]:
        active_until = START_DATE + pd.Timedelta(
            days=random.randint(150, max(151, (END_DATE - START_DATE).days - 30))
        )
    else:
        active_until = END_DATE

    customer_rows.append({
        "CustomerID": customer_id,
        "CustomerName": f"{random.choice(first_names)} {random.choice(last_names)}",
        "Age": random.randint(18, 75),
        "Gender": random.choices(["Female", "Male", "Other"], weights=[48, 49, 3], k=1)[0],
        "City": city,
        "State": state,
        "CustomerSegment": profile,
        "SignupDate": signup_date.strftime("%Y-%m-%d"),
    })

    customer_behaviour[customer_id] = {
        "profile": profile,
        "active_until": active_until,
        "monthly_orders": profiles[profile]["monthly_orders"],
        "preferred_category": preferred_category,
        "discount_type": profiles[profile]["discount"],
    }

customers_df = pd.DataFrame(customer_rows)

# 3. Generate transactions by customer and month
transactions = []
invoice_number = 1
payment_methods = ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking", "Wallet"]
sales_channels = ["Online", "In-Store", "Mobile App", "Marketplace"]

for customer in customer_rows:
    cid = customer["CustomerID"]
    behaviour = customer_behaviour[cid]
    profile = behaviour["profile"]
    month_start = START_DATE

    while month_start <= behaviour["active_until"]:
        month_end = min(month_start + pd.offsets.MonthEnd(0), behaviour["active_until"], END_DATE)
        days_in_month = max(1, (month_end - month_start).days + 1)

        # Seasonal lift around festive months.
        seasonal_factor = 1.25 if month_start.month in [10, 11, 12] else 1.0
        expected_orders = (
            behaviour["monthly_orders"] * days_in_month / 30.4 * seasonal_factor
        )
        order_count = int(np.random.poisson(expected_orders))

        for _ in range(order_count):
            day_offset = random.randint(0, max(0, (month_end - month_start).days))
            order_date = month_start + pd.Timedelta(days=day_offset)
            order_date += pd.Timedelta(hours=random.randint(8, 21), minutes=random.randint(0, 59))
            order_date = min(order_date, END_DATE)

            if random.random() < 0.50:
                product_id = random.choice(products_by_cat[behaviour["preferred_category"]])
            else:
                product_id = random.choice(products_df["ProductID"].tolist())

            product = product_lookup[product_id]
            unit_price = float(product["UnitPrice"])
            quantity = random.choices([1, 2, 3, 4, 5], weights=[55, 25, 12, 6, 2], k=1)[0]

            if behaviour["discount_type"] == "high":
                discount_percent = random.choices(
                    [0, 5, 10, 15, 20, 25],
                    weights=[15, 15, 20, 20, 18, 12],
                    k=1
                )[0]
            else:
                discount_percent = random.choices(
                    [0, 5, 10, 15, 20, 25],
                    weights=[45, 20, 15, 10, 7, 3],
                    k=1
                )[0]

            gross_sales = round(quantity * unit_price, 2)
            discount_amount = round(gross_sales * discount_percent / 100, 2)
            total_sales = round(gross_sales - discount_amount, 2)
            status = random.choices(
                ["Completed", "Returned", "Cancelled"],
                weights=[88, 7, 5],
                k=1
            )[0]

            if status == "Returned":
                quantity = -quantity
                gross_sales = round(quantity * unit_price, 2)
                discount_amount = round(gross_sales * discount_percent / 100, 2)
                total_sales = round(gross_sales - discount_amount, 2)
            elif status == "Cancelled":
                total_sales = 0.0

            transactions.append({
                "InvoiceNo": f"INV{invoice_number:07d}",
                "CustomerID": cid,
                "ProductID": product_id,
                "ProductName": product["ProductName"],
                "Category": product["Category"],
                "InvoiceDate": order_date.strftime("%Y-%m-%d %H:%M:%S"),
                "Quantity": quantity,
                "UnitPrice": unit_price,
                "DiscountPercent": discount_percent,
                "GrossSales": gross_sales,
                "DiscountAmount": discount_amount,
                "TotalSales": total_sales,
                "PaymentMethod": random.choice(payment_methods),
                "SalesChannel": random.choice(sales_channels),
                "OrderStatus": status,
                "City": customer["City"],
                "State": customer["State"],
            })
            invoice_number += 1

        month_start = month_start + pd.offsets.MonthBegin(1)

transactions_df = pd.DataFrame(transactions)
transactions_df["InvoiceDate"] = pd.to_datetime(transactions_df["InvoiceDate"])
transactions_df = transactions_df.sort_values("InvoiceDate").reset_index(drop=True)

# 4. Create inventory snapshot
inventory_rows = []
for product in product_rows:
    stock = random.randint(0, 500)
    reorder_level = random.randint(20, 100)
    inventory_rows.append({
        "ProductID": product["ProductID"],
        "ProductName": product["ProductName"],
        "Category": product["Category"],
        "StockQuantity": stock,
        "ReorderLevel": reorder_level,
        "StockStatus": "Reorder Required" if stock <= reorder_level else "In Stock",
        "InventoryDate": END_DATE.strftime("%Y-%m-%d"),
    })
inventory_df = pd.DataFrame(inventory_rows)

# 5. Add a customer-level churn status for reporting.
completed = transactions_df[transactions_df["OrderStatus"] == "Completed"].copy()
last_purchase = completed.groupby("CustomerID")["InvoiceDate"].max()
customers_df["LastPurchaseDate"] = customers_df["CustomerID"].map(last_purchase)
customers_df["DaysSinceLastPurchase"] = (
    END_DATE - pd.to_datetime(customers_df["LastPurchaseDate"])
).dt.days
customers_df["ChurnStatus"] = np.where(
    customers_df["DaysSinceLastPurchase"].isna()
    | (customers_df["DaysSinceLastPurchase"] > 90),
    "Churned",
    "Active",
)
customers_df["ChurnLabel"] = (customers_df["ChurnStatus"] == "Churned").astype(int)

# 6. Create leakage-aware churn modelling dataset.
# Features are from the 180 days before snapshot date.
# Label = no completed purchase in the following 90 days.
feature_tx = completed[
    (completed["InvoiceDate"] >= FEATURE_START)
    & (completed["InvoiceDate"] < SNAPSHOT_DATE)
].copy()
future_tx = completed[
    (completed["InvoiceDate"] >= SNAPSHOT_DATE)
    & (completed["InvoiceDate"] <= END_DATE)
].copy()

model_data = feature_tx.groupby("CustomerID").agg(
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalSales", "sum"),
    AverageOrderValue=("TotalSales", "mean"),
    CategoriesPurchased=("Category", "nunique"),
    LastPurchaseDate=("InvoiceDate", "max"),
)
model_data["Recency"] = (SNAPSHOT_DATE - model_data["LastPurchaseDate"]).dt.days
model_data = model_data.drop(columns=["LastPurchaseDate"])
model_data = model_data.reindex(customers_df["CustomerID"])
model_data.index.name = "CustomerID"

for column in ["Frequency", "Monetary", "AverageOrderValue", "CategoriesPurchased"]:
    model_data[column] = model_data[column].fillna(0)
model_data["Recency"] = model_data["Recency"].fillna(181)

future_customer_ids = set(future_tx["CustomerID"].unique())
model_data["Churn"] = (~model_data.index.isin(future_customer_ids)).astype(int)
model_data = model_data.reset_index()
model_data["SnapshotDate"] = SNAPSHOT_DATE.strftime("%Y-%m-%d")
model_data["LabelWindowEnd"] = END_DATE.strftime("%Y-%m-%d")

# 7. Save datasets.
customers_df.to_csv(OUTPUT_DIR / "customers.csv", index=False)
products_df.to_csv(OUTPUT_DIR / "products.csv", index=False)

# --------------------------------------------------
# INTENTIONALLY ADD DATA QUALITY ISSUES FOR PRACTICE
# --------------------------------------------------


# --------------------------------------------------
# INTENTIONALLY ADD DATA QUALITY ISSUES FOR PRACTICE
# --------------------------------------------------

# 1. Add missing values to DiscountPercent
missing_rows = transactions_df.sample(
    frac=0.01,
    random_state=42
).index

transactions_df.loc[
    missing_rows, "DiscountPercent"
] = float("nan")

# 2. Add missing values to PaymentMethod
missing_payment_rows = transactions_df.sample(
    frac=0.01,
    random_state=43
).index

transactions_df.loc[
    missing_payment_rows, "PaymentMethod"
] = float("nan")

# 3. Add exact duplicate transaction rows (2%)
duplicate_rows = transactions_df.sample(
    frac=0.02,
    random_state=44
)

transactions_df = pd.concat(
    [transactions_df, duplicate_rows],
    ignore_index=True
)

# 4. Shuffle the data
transactions_df = transactions_df.sample(
    frac=1,
    random_state=45
).reset_index(drop=True)

# 5. Verify data quality issues
print("\nDATA QUALITY ISSUES ADDED")
print("Rows including duplicates:", len(transactions_df))
print("Exact duplicate rows:", transactions_df.duplicated().sum())
print("\nMissing values by column:")
print(transactions_df.isnull().sum())
transactions_df.to_csv(
    OUTPUT_DIR / "transactions.csv",
    index=False
)

print("Updated transactions.csv saved successfully!")

inventory_df.to_csv(OUTPUT_DIR / "inventory.csv", index=False)
model_data.to_csv(OUTPUT_DIR / "churn_model_dataset.csv", index=False)

print("\n" + "=" * 55)
print("RETAILPULSE DATA CREATED")
print("=" * 55)
print(f"Customers: {len(customers_df):,}")
print(f"Products: {len(products_df):,}")
print(f"Transactions: {len(transactions_df):,}")
print("\nTransactions by status:")
print(transactions_df["OrderStatus"].value_counts().to_string())
print("\nCustomer churn status (last 90 days):")
print(customers_df["ChurnStatus"].value_counts().to_string())
print("\nChurn modelling labels (future 90 days):")
print(model_data["Churn"].value_counts().rename({0: "Not churned", 1: "Churned"}).to_string())
print("\nFiles saved in:", OUTPUT_DIR.resolve())
print("Created: customers.csv, products.csv, transactions.csv, inventory.csv, churn_model_dataset.csv")
print("NOTE: Data is synthetic and imitates retail behaviour; it is not real business data.")
