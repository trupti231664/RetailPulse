
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


BASE_DIR = Path(__file__).resolve().parent
INVENTORY_FILE = BASE_DIR / "data" / "inventory.csv"

TRANSACTIONS_FILE = BASE_DIR / "data" / "transactions.csv"

def load_inventory():
    inventory = pd.read_csv(INVENTORY_FILE)

    inventory["StockQuantity"] = pd.to_numeric(
        inventory["StockQuantity"], errors="coerce"
    )
    inventory["ReorderLevel"] = pd.to_numeric(
        inventory["ReorderLevel"], errors="coerce"
    )

    inventory = inventory.dropna(
        subset=["ProductID", "StockQuantity", "ReorderLevel"]
    )
    inventory["SuggestedReorderQty"] = (
        2 * inventory["ReorderLevel"]
        - inventory["StockQuantity"]
    ).clip(lower=0).astype(int)

    inventory["StockStatus"] = inventory.apply(
        lambda row: (
            "Reorder Required"
            if row["StockQuantity"] <= row["ReorderLevel"]
            else "In Stock"
        ),
        axis=1,
    )

    return inventory


def calculate_product_demand():
    transactions = pd.read_csv(TRANSACTIONS_FILE)

    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"],
        dayfirst=True,
        errors="coerce"
    )

    transactions["Quantity"] = pd.to_numeric(
        transactions["Quantity"],
        errors="coerce"
    )

    transactions = transactions[
        (transactions["OrderStatus"] == "Completed")
        & transactions["InvoiceDate"].notna()
        & transactions["Quantity"].notna()
    ].copy()

    product_demand = (
        transactions.groupby("ProductID")
        .agg(
            TotalUnitsSold=("Quantity", "sum"),
            LastSaleDate=("InvoiceDate", "max"),
            SalesDays=("InvoiceDate", "nunique")
        )
        .reset_index()
    )

    return product_demand

def main():
    st.title("📦 Inventory Optimization")
    st.write("Monitor stock levels and identify products that need restocking.")

    try:
        inventory = load_inventory()
        
        demand = calculate_product_demand()

        inventory = inventory.merge(
            demand,
             on="ProductID",
            how="left"
        )

        inventory["TotalUnitsSold"] = (
            inventory["TotalUnitsSold"].fillna(0).astype(int)
        )

        inventory["SalesDays"] = (
            inventory["SalesDays"].fillna(0).astype(int)
        )

        inventory["LastSaleDate"] = pd.to_datetime(
            inventory["LastSaleDate"],
            errors="coerce"
        )

        inventory["AverageDailyDemand"] = (
            inventory["TotalUnitsSold"]
            / inventory["SalesDays"].replace(0, pd.NA)
            ).fillna(0).round(2)
        # Estimate demand over the next 7 days
        inventory["Estimated7DayDemand"] = (
            inventory["AverageDailyDemand"] * 7
        ).round().astype(int)

        # Suggested quantity to cover the next 7 days
        inventory["DemandBasedReorderQty"] = (
            inventory["Estimated7DayDemand"]
             - inventory["StockQuantity"]
        ).clip(lower=0).astype(int)

    except Exception as error:
        st.error(f"Unable to load inventory data: {error}")
        return

    total_products = len(inventory)
    low_stock = int((inventory["StockStatus"] == "Reorder Required").sum())
    in_stock = total_products - low_stock

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products", total_products)
    col2.metric("Products to Reorder", low_stock)
    col3.metric("Products In Stock", in_stock)

    st.subheader("📊 Overall Stock Status")

    stock_summary = pd.DataFrame({
         "Stock Status": ["Products to Reorder", "Products In Stock"],
         "Count": [low_stock, in_stock]
    })

    fig_stock = px.pie(
        stock_summary,
        names="Stock Status",
        values="Count",
        title="Products Requiring Restock vs In Stock",
        hole=0.45
    )

    st.plotly_chart(
        fig_stock,
         width="stretch"
    )

    st.subheader("Stock Status by Category")

    category_summary = (
        inventory.groupby("Category")["StockStatus"]
        .apply(lambda values: (values == "Reorder Required").sum())
        .reset_index(name="Products to Reorder")
    )

    fig = px.bar(
        category_summary,
        x="Category",
        y="Products to Reorder",
        title="Products Requiring Reorder by Category",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("🚨 Products to Restock")

    restock_products = inventory[
         inventory["StockStatus"] == "Reorder Required"
    ].copy()

    restock_products = restock_products.sort_values(
         by="StockQuantity",
        ascending=True
    )

    st.dataframe(
         restock_products[
             [
                 "ProductID",
                "ProductName",
                "Category",
                "StockQuantity",
                "ReorderLevel",
                "SuggestedReorderQty",
                "TotalUnitsSold",
                "LastSaleDate",
                "AverageDailyDemand",
                "Estimated7DayDemand",
                "DemandBasedReorderQty",
            ]
        ],
        width="stretch",
        hide_index=True
    )

    restock_csv = restock_products.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Restock Report",
        data=restock_csv,
        file_name="products_to_restock.csv",
        mime="text/csv",
    )
    st.subheader("🔥 Top 10 Products by Demand")
    
    top_demand = (
         inventory.sort_values(
            by="TotalUnitsSold",
            ascending=False
        )
        .head(10)
        .sort_values(
             by="TotalUnitsSold",
            ascending=True
        )
    )
    fig_demand = px.bar(
        top_demand,
        x="TotalUnitsSold",
         y="ProductName",
         orientation="h",
         text="TotalUnitsSold",
        color_discrete_sequence=["#2878B5"],
        hover_data=["ProductID", "Category", "StockQuantity"],
        title="Top 10 Best-Selling Products",
        labels={
            "TotalUnitsSold": "Total Units Sold",
            "ProductName": "Product"
        }
    )
    
    top_demand = (
        inventory.sort_values(
            by="TotalUnitsSold",
            ascending=False
        )
        .head(10)
        .sort_values(
             by="TotalUnitsSold",
            ascending=True
        )
    )

    fig_demand.update_traces(
        textposition="outside"
    )
    
    fig_demand.update_layout(
        showlegend=False,
        yaxis_title="Product",
        xaxis_title="Total Units Sold"
    )
    st.plotly_chart(
        fig_demand,
         width="stretch"
    )

    st.subheader("Inventory Details")

    search_text = st.text_input(
        "🔍 Search Product",
        placeholder="Enter product name or Product ID"
    )

    category_options = ["All Categories"] + sorted(
        inventory["Category"].dropna().unique().tolist()
    )

    selected_category = st.selectbox(
        "Filter by Category",
        category_options
    )

    status_options = ["All Products", "In Stock", "Reorder Required"]
  
    selected_status = st.selectbox(
        "Filter by Stock Status",
        status_options
    )
    
    filtered_inventory = inventory.copy()

    if search_text:
        filtered_inventory = filtered_inventory[
            filtered_inventory["ProductName"].astype(str).str.contains(
                search_text, case=False, na=False
            )
            | filtered_inventory["ProductID"].astype(str).str.contains(
                search_text, case=False, na=False
            )
        ]

    if selected_category != "All Categories":
        filtered_inventory = filtered_inventory[
            filtered_inventory["Category"] == selected_category
        ]

    if selected_status == "In Stock":
        filtered_inventory = filtered_inventory[
            filtered_inventory["StockStatus"] == "In Stock"
        ]
    elif selected_status == "Reorder Required":
        filtered_inventory = filtered_inventory[
            filtered_inventory["StockStatus"] == "Reorder Required"
        ]

    status_filter = st.selectbox(
        "Filter by Stock Status",
        ["All", "Reorder Required", "In Stock"],
    )

    if status_filter != "All":
        inventory = inventory[
            inventory["StockStatus"] == status_filter
        ]

    st.dataframe(filtered_inventory, width="stretch", hide_index=True)

    csv_data = filtered_inventory.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Inventory CSV",
        data=csv_data,
        file_name="inventory_optimization.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()
