
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from io import BytesIO

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RetailPulse | Sales Analytics",
    page_icon="📊",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent
CLEAN_DIR = PROJECT_DIR / "cleaned"

TRANSACTIONS_FILE = CLEAN_DIR / "transactions_cleaned.csv"
SEGMENTS_FILE = CLEAN_DIR / "customer_segments.csv"
CHURN_FILE = CLEAN_DIR / "customer_churn_predictions.csv"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    transactions = pd.read_csv(TRANSACTIONS_FILE)
    segments = pd.read_csv(SEGMENTS_FILE)

    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"], errors="coerce"
    )

    for col in ["TotalSales", "Quantity", "UnitPrice"]:
        if col in transactions.columns:
            transactions[col] = pd.to_numeric(
                transactions[col], errors="coerce"
            )

    transactions = transactions.dropna(
        subset=["InvoiceDate", "TotalSales"]
    )

    churn = None
    if CHURN_FILE.exists():
        churn = pd.read_csv(CHURN_FILE)

    return transactions, segments, churn


try:
    transactions, segments, churn = load_data()
except FileNotFoundError as error:
    st.error(f"Required project file not found: {error}")
    st.info(
        "Check that your CSV files are saved in the project's "
        "'cleaned' folder."
    )
    st.stop()


# --------------------------------------------------
# 3. CUSTOMER SEGMENT NAMES
# --------------------------------------------------

CLUSTER_NAMES = {
    0: "Loyal Frequent Buyers",
    1: "Inactive / Win-Back Customers",
    2: "Regular Customers",
    3: "High-Value VIP Customers",
}

CLUSTER_DESCRIPTIONS = {
    0: (
        "These customers purchase frequently and have purchased recently. "
        "Reward their loyalty with points, exclusive offers, and "
        "personalised recommendations."
    ),
    1: (
        "These customers have a very high recency value, meaning their "
        "last recorded purchase was a long time ago. Try win-back campaigns, "
        "customer feedback surveys, and relevant reactivation offers."
    ),
    2: (
        "These customers have moderate purchase recency, frequency, "
        "and spending. Encourage repeat purchases through product "
        "recommendations, bundles, and loyalty rewards."
    ),
    3: (
        "These customers have the highest average spending and also "
        "purchase frequently. Focus on VIP benefits, personalised service, "
        "and retaining their long-term loyalty."
    ),
}

segments["Cluster"] = pd.to_numeric(
    segments["Cluster"], errors="coerce"
)

segments = segments.dropna(subset=["Cluster"]).copy()

segments["Cluster"] = segments["Cluster"].astype(int)

segments["Segment"] = segments["Cluster"].map(
    CLUSTER_NAMES
).fillna(
    segments["Cluster"].map(
        lambda cluster: f"Customer Cluster {cluster}"
    )
)


# --------------------------------------------------
# 4. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.title("Dashboard Filters")

min_date = transactions["InvoiceDate"].min().date()
max_date = transactions["InvoiceDate"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

categories = sorted(
    transactions["Category"].dropna().unique().tolist()
)
selected_categories = st.sidebar.multiselect(
    "Select categories",
    categories,
    default=categories
)

statuses = sorted(
    transactions["OrderStatus"].dropna().unique().tolist()
)
selected_statuses = st.sidebar.multiselect(
    "Select order status",
    statuses,
    default=statuses
)


# --------------------------------------------------
# 5. APPLY FILTERS
# --------------------------------------------------

filtered = transactions.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range

    filtered = filtered[
        filtered["InvoiceDate"].dt.date.between(
            start_date, end_date
        )
    ]

if selected_categories:
    filtered = filtered[
        filtered["Category"].isin(selected_categories)
    ]
else:
    filtered = filtered.iloc[0:0]

if selected_statuses:
    filtered = filtered[
        filtered["OrderStatus"].isin(selected_statuses)
    ]
else:
    filtered = filtered.iloc[0:0]


# --------------------------------------------------
# 6. DASHBOARD HEADER AND KPIs
# --------------------------------------------------

st.title("📊 RetailPulse | Sales Analytics")
st.caption(
    "Retail sales performance, customer segmentation, "
    "and customer behaviour analysis"
)

total_sales = filtered["TotalSales"].sum()
total_orders = filtered["InvoiceNo"].nunique()
total_customers = filtered["CustomerID"].nunique()

average_order_value = (
    total_sales / total_orders if total_orders else 0
)

return_rate = (
    (filtered["OrderStatus"] == "Returned").sum()
    / len(filtered) * 100
    if len(filtered) else 0
)


k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Net Sales", f"₹{total_sales / 10000000:.2f} Cr")
k2.metric("Total Orders", f"{total_orders:,}")
k3.metric("Customers", f"{total_customers:,}")
k4.metric("Average Order Value", f"₹{average_order_value:,.0f}")
k5.metric("Return Rate", f"{return_rate:.1f}%")

# --------------------------------------------------
# 7. SALES ANALYSIS
# --------------------------------------------------

st.header("Sales Performance")

left, right = st.columns(2)

with left:
    st.subheader("Monthly Sales Trend")

    monthly = (
        filtered.dropna(subset=["InvoiceDate"])
        .set_index("InvoiceDate")["TotalSales"]
        .resample("MS")
        .sum()
        .reset_index()
    )

    if not monthly.empty:
        fig = px.line(
            monthly,
            x="InvoiceDate",
            y="TotalSales",
            markers=True,
            title="Monthly Sales"
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Sales (₹)"
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No sales data for the selected filters.")

with right:
    st.subheader("Sales by Category")

    category_sales = (
        filtered.groupby("Category", as_index=False)["TotalSales"]
        .sum()
        .sort_values("TotalSales", ascending=False)
    )

    if not category_sales.empty:
        fig = px.bar(
            category_sales,
            x="Category",
            y="TotalSales",
            title="Category-wise Sales"
        )
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Sales (₹)",
            xaxis_tickangle=-40
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No category data for the selected filters.")

left, right = st.columns(2)

with left:
    st.subheader("Transactions by Order Status")
    status_counts = (
        filtered["OrderStatus"].value_counts()
        .rename_axis("OrderStatus")
        .reset_index(name="Transactions")
    )

    if not status_counts.empty:
        fig = px.pie(
            status_counts,
            names="OrderStatus",
            values="Transactions",
            title="Order Status Distribution"
        )
        st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Top Products by Sales")

    if "ProductName" in filtered.columns:
        top_products = (
            filtered.groupby("ProductName", as_index=False)["TotalSales"]
            .sum()
            .nlargest(10, "TotalSales")
            .sort_values("TotalSales")
        )

        if not top_products.empty:
            fig = px.bar(
                top_products,
                x="TotalSales",
                y="ProductName",
                orientation="h",
                title="Top 10 Products"
            )
            fig.update_layout(
                xaxis_title="Sales (₹)",
                yaxis_title="Product"
            )
            st.plotly_chart(fig, width="stretch")


# --------------------------------------------------
# 8. CUSTOMER SEGMENTATION
# --------------------------------------------------

st.header("Customer Segmentation")
st.write(
    "Customers are grouped using Recency, Frequency, and Monetary (RFM) "
    "metrics. The summary below recalculates using the current dashboard "
    "filters and completed transactions."
)

# Recalculate RFM metrics using the currently filtered transactions.
rfm_transactions = filtered.copy()

if "OrderStatus" in rfm_transactions.columns:
    rfm_transactions = rfm_transactions[
        rfm_transactions["OrderStatus"] == "Completed"
    ].copy()

if not rfm_transactions.empty:
    reference_date = rfm_transactions["InvoiceDate"].max().normalize()

    rfm_summary = (
        rfm_transactions.groupby("CustomerID")
        .agg(
            LastPurchase=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalSales", "sum"),
        )
        .reset_index()
    )

    rfm_summary["Recency"] = (
        reference_date - rfm_summary["LastPurchase"].dt.normalize()
    ).dt.days

    # Simple, rule-based segments based on the selected data.
    recency_median = rfm_summary["Recency"].median()
    frequency_median = rfm_summary["Frequency"].median()
    monetary_median = rfm_summary["Monetary"].median()

    rfm_summary["Cluster"] = 2

    rfm_summary.loc[
        (rfm_summary["Recency"] <= recency_median)
        & (rfm_summary["Frequency"] >= frequency_median)
        & (rfm_summary["Monetary"] >= monetary_median),
        "Cluster",
    ] = 3

    rfm_summary.loc[
        (rfm_summary["Recency"] > recency_median)
        & (rfm_summary["Frequency"] < frequency_median),
        "Cluster",
    ] = 1

    rfm_summary.loc[
        (rfm_summary["Recency"] <= recency_median)
        & (rfm_summary["Frequency"] >= frequency_median)
        & (rfm_summary["Monetary"] < monetary_median),
        "Cluster",
    ] = 0

    rfm_summary["Cluster"] = rfm_summary["Cluster"].astype(int)
    rfm_summary["Segment"] = rfm_summary["Cluster"].map(CLUSTER_NAMES)

    segment_summary = (
        rfm_summary.groupby(["Cluster", "Segment"], as_index=False)
        .agg(
            Customers=("CustomerID", "nunique"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            **{"Total Revenue": ("Monetary", "sum")},
        )
        .sort_values("Cluster")
    )

    customer_count = rfm_summary["CustomerID"].nunique()
    total_revenue = segment_summary["Total Revenue"].sum()

    segment_summary["Customer Share (%)"] = (
        segment_summary["Customers"] / customer_count * 100
        if customer_count else 0
    )
    segment_summary["Revenue Contribution (%)"] = (
        segment_summary["Total Revenue"] / total_revenue * 100
        if total_revenue > 0 else 0
    )
else:
    st.info(
        "No completed transactions are available for the selected filters. "
        "Choose a different date range, category, or order status."
    )
    segment_summary = pd.DataFrame(
        columns=[
            "Cluster", "Segment", "Customers", "Avg_Recency",
            "Avg_Frequency", "Avg_Monetary", "Total Revenue",
            "Customer Share (%)", "Revenue Contribution (%)",
        ]
    )

# Charts respond to the current dashboard filters.
left, right = st.columns(2)

with left:
    st.subheader("Customers by Segment")
    if not segment_summary.empty:
        fig = px.bar(
            segment_summary,
            x="Segment",
            y="Customers",
            color="Segment",
            text="Customers",
            title="Customer Segment Distribution",
            hover_data=["Customer Share (%)"],
        )
        fig.update_layout(
            xaxis_title="Customer Segment",
            yaxis_title="Number of Customers",
            showlegend=False,
            xaxis_tickangle=-25,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No customer segments available for the selected filters.")

with right:
    st.subheader("Average Spending by Segment")
    if not segment_summary.empty:
        fig = px.bar(
            segment_summary,
            x="Segment",
            y="Avg_Monetary",
            color="Segment",
            text_auto=".2s",
            title="Average Monetary Value by Segment",
        )
        fig.update_layout(
            xaxis_title="Customer Segment",
            yaxis_title="Average Spending (₹)",
            showlegend=False,
            xaxis_tickangle=-25,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No spending data available for the selected filters.")

# RFM summary table
st.subheader("RFM Summary by Customer Segment")

display_summary = segment_summary.rename(
    columns={
        "Cluster": "Cluster ID",
        "Avg_Recency": "Average Recency (days)",
        "Avg_Frequency": "Average Frequency",
        "Avg_Monetary": "Average Monetary (₹)",
    }
).copy()

if not display_summary.empty:
    display_summary["Average Recency (days)"] = display_summary["Average Recency (days)"].round(1)
    display_summary["Average Frequency"] = display_summary["Average Frequency"].round(1)
    display_summary["Average Monetary (₹)"] = display_summary["Average Monetary (₹)"].round(0)
    display_summary["Customer Share (%)"] = display_summary["Customer Share (%)"].round(2)
    display_summary["Revenue Contribution (%)"] = display_summary["Revenue Contribution (%)"].round(2)
    display_summary["Total Revenue"] = display_summary["Total Revenue"].round(2)

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Customer Share (%)": st.column_config.NumberColumn("Customer Share (%)", format="%.2f"),
        "Revenue Contribution (%)": st.column_config.NumberColumn("Revenue Contribution (%)", format="%.2f"),
        "Average Recency (days)": st.column_config.NumberColumn("Average Recency (days)", format="%.1f"),
        "Average Frequency": st.column_config.NumberColumn("Average Frequency", format="%.1f"),
        "Average Monetary (₹)": st.column_config.NumberColumn("Average Monetary (₹)", format="₹%,.0f"),
        "Total Revenue": st.column_config.NumberColumn("Total Revenue (₹)", format="₹%,.2f"),
    },
)

# Explain each segment when data is available.
st.subheader("What Each Segment Means")
if not segment_summary.empty:
    for _, row in segment_summary.iterrows():
        cluster_id = int(row["Cluster"])
        segment_name = row["Segment"]
        description = CLUSTER_DESCRIPTIONS.get(
            cluster_id,
            "Review this group's RFM values to decide on an appropriate strategy.",
        )
        with st.expander(f"{segment_name} — {int(row['Customers']):,} customers"):
            st.write(description)
            a, b, c = st.columns(3)
            a.metric("Avg. Recency", f"{row['Avg_Recency']:.1f} days")
            b.metric("Avg. Frequency", f"{row['Avg_Frequency']:.1f}")
            c.metric("Avg. Spending", f"₹{row['Avg_Monetary']:,.2f}")

# Individual customer records (pre-generated segmentation file).
st.subheader("Customer Segmentation Records")
segment_options = ["All segments"] + sorted(
    segments["Segment"].dropna().unique().tolist()
)
selected_segment = st.selectbox("Filter customers by segment", segment_options)
customer_view = segments.copy()
if selected_segment != "All segments":
    customer_view = customer_view[customer_view["Segment"] == selected_segment]

columns_to_show = [
    col for col in ["CustomerID", "Segment", "Recency", "Frequency", "Monetary"]
    if col in customer_view.columns
]
if "Monetary" in customer_view.columns:
    customer_view = customer_view.sort_values("Monetary", ascending=False)
st.dataframe(customer_view[columns_to_show], use_container_width=True, hide_index=True)
st.download_button(
    "Download customer segments CSV",
    data=customer_view.to_csv(index=False).encode("utf-8"),
    file_name="customer_segments_named.csv",
    mime="text/csv",
)


# --------------------------------------------------
# 9. BUSINESS RECOMMENDATIONS
# --------------------------------------------------

st.header("Business Recommendations")

st.write(
    "Recommended business actions based on customer "
    "segmentation and RFM analysis."
)

recommendations = {
    "Loyal Frequent Buyers": (
        "Strategy: Reward and retain loyal customers.",
        "Actions: Offer loyalty points, exclusive discounts, "
        "personalised recommendations, and early access to products."
    ),
    "Inactive / Win-Back Customers": (
        "Strategy: Reactivate inactive customers.",
        "Actions: Send win-back campaigns, request customer feedback, "
        "and test personalised return offers."
    ),
    "Regular Customers": (
        "Strategy: Increase repeat purchases.",
        "Actions: Recommend related products, create product bundles, "
        "and encourage loyalty-program enrolment."
    ),
    "High-Value VIP Customers": (
        "Strategy: Protect high-value customer relationships.",
        "Actions: Provide VIP benefits, personalised service, "
        "exclusive offers, and early access to products."
    )
}

for segment_name, details in recommendations.items():
    with st.expander(segment_name):
        st.write(details[0])
        st.write(details[1])


# --------------------------------------------------
# DOWNLOAD BUSINESS REPORT
# --------------------------------------------------

st.subheader("Download Business Report")

st.write(
    "Export your customer segmentation results "
    "and recommendations for portfolio presentations."
)

# Prepare the customer segment summary
report_summary = segment_summary.rename(
    columns={
        "Cluster": "Cluster ID",
        "Avg_Recency": "Average Recency (days)",
        "Avg_Frequency": "Average Frequency",
        "Avg_Monetary": "Average Monetary (₹)"
    }
)


# Create an Excel report in memory
report_buffer = BytesIO()

# Round values before exporting
report_summary["Average Recency (days)"] = report_summary["Average Recency (days)"].round(1)
report_summary["Average Frequency"] = report_summary["Average Frequency"].round(1)
report_summary["Average Monetary (₹)"] = report_summary["Average Monetary (₹)"].round(0)
report_summary["Customer Share (%)"] = report_summary["Customer Share (%)"].round(2)
report_summary["Revenue Contribution (%)"] = report_summary["Revenue Contribution (%)"].round(2)
report_summary["Total Revenue"] = report_summary["Total Revenue"].round(2)

with pd.ExcelWriter(report_buffer, engine="openpyxl") as writer:

    # Export Segment Summary
    report_summary.to_excel(
        writer,
        sheet_name="Segment Summary",
        index=False
    )

    # Export Customer Records
    customer_view.to_excel(
        writer,
        sheet_name="Customer Records",
        index=False
    )

    # Format the Total Revenue column in the Segment Summary sheet.
    worksheet = writer.sheets["Segment Summary"]
    total_revenue_column = None
    for column_cells in worksheet.iter_cols(1, worksheet.max_column):
        if column_cells[0].value == "Total Revenue":
            total_revenue_column = column_cells[0].column
            break

    if total_revenue_column is not None:
        for row in worksheet.iter_rows(
            min_row=2,
            min_col=total_revenue_column,
            max_col=total_revenue_column,
        ):
            for cell in row:
                cell.number_format = '₹#,##,##0.00'
        worksheet.column_dimensions[
            worksheet.cell(row=1, column=total_revenue_column).column_letter
        ].width = 24

# Download button
st.download_button(
    label="📥 Download Business Report (Excel)",
    data=report_buffer.getvalue(),
    file_name="RetailPulse_Business_Report.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    )
)


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "RetailPulse | Educational portfolio project. "
    "The underlying dataset is synthetic and does not represent real customers."
)
