
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "cleaned" / "transactions_cleaned.csv"
OUTPUT_DIR = PROJECT_DIR / "cleaned"

# --------------------------------------------------
# 2. LOAD AND PREPARE DATA
# --------------------------------------------------


@st.cache_data
def prepare_monthly_sales():
    df = pd.read_csv(DATA_FILE)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df["TotalSales"] = pd.to_numeric(
        df["TotalSales"], errors="coerce"
    )

    df = df.dropna(
        subset=["InvoiceDate", "TotalSales", "OrderStatus"]
    )

    # Use completed transactions only
    df = df[df["OrderStatus"] == "Completed"].copy()

    # Aggregate sales by month
    monthly_sales = (
        df.set_index("InvoiceDate")["TotalSales"]
        .resample("MS")
        .sum()
        .rename("Sales")
        .reset_index()
    )

    monthly_sales = monthly_sales.rename(
        columns={"InvoiceDate": "Month"}
    )

    # Exclude the current incomplete month
    today = pd.Timestamp.today().normalize()
    current_month = today.replace(day=1)

    monthly_sales = monthly_sales[
        monthly_sales["Month"] < current_month
    ].copy()

    return monthly_sales

# --------------------------------------------------
# 3. CREATE FORECAST FEATURES
# --------------------------------------------------

def add_features(data):
    result = data.copy()

    result["MonthNumber"] = (
        result["Month"].dt.year * 12
        + result["Month"].dt.month
    )

    result["MonthOfYear"] = result["Month"].dt.month

    result["MonthSin"] = np.sin(
        2 * np.pi * result["MonthOfYear"] / 12
    )

    result["MonthCos"] = np.cos(
        2 * np.pi * result["MonthOfYear"] / 12
    )

    return result


# --------------------------------------------------
# 4. TRAIN AND EVALUATE MODEL
# --------------------------------------------------

def train_forecast(monthly_sales, forecast_months=3):

    # Close any leftover MLflow run from a previous execution
    if mlflow.active_run() is not None:
        mlflow.end_run()

    data = add_features(monthly_sales)

    if len(data) < 18:
        raise ValueError(
            "At least 18 months of sales data are recommended."
        )

    feature_columns = [
        "MonthNumber", "MonthSin", "MonthCos"
    ]

    # Keep the latest 6 months for testing
    test_size = 6

    train_data = data.iloc[:-test_size].copy()
    test_data = data.iloc[-test_size:].copy()

    test_model = LinearRegression()

    test_model.fit(
        train_data[feature_columns],
        train_data["Sales"]
    )

    test_predictions = test_model.predict(
        test_data[feature_columns]
    )
    # Random Forest test model
    rf_test_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    rf_test_model.fit(
        train_data[feature_columns],
        train_data["Sales"]
    )

    rf_test_predictions = rf_test_model.predict(
        test_data[feature_columns]
    )

    rf_mae = mean_absolute_error(
        test_data["Sales"], rf_test_predictions
    )

    rf_rmse = np.sqrt(
        mean_squared_error(
            test_data["Sales"], rf_test_predictions
        )
    )

    mae = mean_absolute_error(
        test_data["Sales"], test_predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            test_data["Sales"], test_predictions
        )
    )
    baseline_predictions = test_data["Sales"].shift(1)
    baseline_predictions.iloc[0] = train_data["Sales"].iloc[-1]

    baseline_mae = mean_absolute_error(
    test_data["Sales"], baseline_predictions
    )

    baseline_rmse = np.sqrt(
    mean_squared_error(test_data["Sales"], baseline_predictions)
    )

    
    # MLflow experiment tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("RetailPulse-Demand-Forecasting")

    # Log Linear Regression evaluation
    with mlflow.start_run(run_name="Linear-Regression"):
        mlflow.log_param("model", "Linear Regression")
        mlflow.log_param("features", ",".join(feature_columns))
        mlflow.log_param("test_months", test_size)
        mlflow.log_metric("MAE", float(mae))
        mlflow.log_metric("RMSE", float(rmse))

    # Log Random Forest evaluation
    with mlflow.start_run(run_name="Random-Forest"):
        mlflow.log_param("model", "Random Forest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("features", ",".join(feature_columns))
        mlflow.log_param("test_months", test_size)
        mlflow.log_metric("MAE", float(rf_mae))
        mlflow.log_metric("RMSE", float(rf_rmse))

    # Log baseline evaluation
    with mlflow.start_run(run_name="Baseline"):
        mlflow.log_param("model", "Previous Month Sales")
        mlflow.log_param("test_months", test_size)
        mlflow.log_metric("MAE", float(baseline_mae))
        mlflow.log_metric("RMSE", float(baseline_rmse))

    # Train Linear Regression model
    final_model = LinearRegression()
    final_model.fit(
        data[feature_columns],
        data["Sales"]
    )

    # Train Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    rf_model.fit(
        data[feature_columns],
        data["Sales"]
    )
    
    # Save trained models in MLflow
    with mlflow.start_run(run_name="Final-Linear-Regression"):
        mlflow.log_param("model", "Linear Regression")
        mlflow.log_param("features", ",".join(feature_columns))
        mlflow.sklearn.log_model(
            sk_model=final_model,
            name="linear_regression_model"
        )

        # Close any previously active MLflow run
        if mlflow.active_run() is not None:
            mlflow.end_run()

        with mlflow.start_run(run_name="Final-Random-Forest"):
            mlflow.log_param("model", "Random Forest")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 5)
            mlflow.log_param("features", ",".join(feature_columns))

        rf_model_info = mlflow.sklearn.log_model(
            sk_model=rf_model,
            name="random_forest_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        mlflow.register_model(
            model_uri=rf_model_info.model_uri,
            name="RetailPulse-Random-Forest"
        )

    last_month = data["Month"].max()

    future_months = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        periods=forecast_months,
        freq="MS"
    )

    future_data = pd.DataFrame({"Month": future_months})
    future_data = add_features(future_data)

    # Linear Regression forecast
    predictions = final_model.predict(
        future_data[feature_columns]
    )

    # Load registered Random Forest model from MLflow
    registered_rf_model = mlflow.pyfunc.load_model(
        "models:/RetailPulse-Random-Forest@champion"
    )

    # Random Forest forecast using registered model
    rf_predictions = registered_rf_model.predict(
        future_data[feature_columns]
    )

    future_data["ForecastSales"] = np.maximum(
        predictions, 0
    )
    
    future_data["RandomForestForecast"] = np.maximum(
        rf_predictions, 0
    )
    # Sales cannot be negative
    future_data["ForecastSales"] = np.maximum(
        predictions, 0
    )

    return (
        future_data,
        mae,
        rmse,
        rf_mae,
        rf_rmse,
        baseline_mae,
        baseline_rmse,
        test_data,
        test_predictions,
        rf_test_predictions
    )


# --------------------------------------------------
# 5. STREAMLIT PAGE
# --------------------------------------------------

def main():
    st.set_page_config(
        page_title="RetailPulse Demand Forecasting",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 RetailPulse Demand Forecasting")

    st.write(
        "Forecast future monthly sales using historical "
        "completed transactions."
    )

    if not DATA_FILE.exists():
        st.error(f"Dataset not found: {DATA_FILE}")
        st.stop()

    try:
        monthly_sales = prepare_monthly_sales()

        if monthly_sales.empty:
            st.warning("No completed sales data is available.")
            st.stop()

        forecast_months = st.slider(
            "Forecast horizon (months)",
            min_value=1,
            max_value=6,
            value=3
        )

        (
            forecast,
            mae,
            rmse,
            rf_mae,
            rf_rmse,
            baseline_mae,
            baseline_rmse,
            test_data,
            test_predictions,
            rf_test_predictions
        ) = train_forecast(monthly_sales, forecast_months)
    

        # KPI cards
        
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Historical Months",
             f"{len(monthly_sales)}"
        )

        col2.metric(
            "Linear Regression MAE",
             f"₹{mae:,.0f}"
        )

        col3.metric(
            "Linear Regression RMSE",
             f"₹{rmse:,.0f}"
        )

        st.subheader("Model vs Baseline Comparison")

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Baseline MAE",
            f"₹{baseline_mae:,.0f}",
        )

        col5.metric(
            "Random Forest MAE",
             f"₹{rf_mae:,.0f}"
        ) 
        col6.metric(
            "Random Forest RMSE",
             f"₹{rf_rmse:,.0f}"
        )

        # Model MAE Comparison Chart
        st.subheader("📊 Model MAE Comparison")

        mae_comparison = pd.DataFrame({
            "Model": [
                "Baseline",
                "Linear Regression",
                "Random Forest"
            ],
            "MAE (₹)": [
                baseline_mae,
                mae,
                rf_mae
            ]
        })

        fig_mae = px.bar(
            mae_comparison,
            x="Model",
            y="MAE (₹)",
            color="Model",
            text="MAE (₹)",
            title="Mean Absolute Error (Lower is Better)"
        )

        fig_mae.update_traces(
            texttemplate="₹%{y:,.0f}",
            textposition="outside"
        )

        fig_mae.update_layout(
            yaxis_title="Mean Absolute Error (₹)",
            xaxis_title="Model",
            showlegend=False
        )

        st.plotly_chart(
            fig_mae,
            width="stretch"
        )

        st.subheader("RMSE Comparison")

        col7, col8, col9 = st.columns(3)

        col7.metric(
            "Baseline RMSE",
            f"₹{baseline_rmse:,.0f}"
        )

        col8.metric(
            "Linear Regression RMSE",
            f"₹{rmse:,.0f}"
        )

        col9.metric(
            "Random Forest RMSE",
            f"₹{rf_rmse:,.0f}"
        )

        
        st.subheader("📊 RMSE Model Comparison")

        rmse_comparison = pd.DataFrame({
            "Model": [
                "Baseline",
                "Linear Regression",
                "Random Forest"
            ],
            "RMSE": [
                baseline_rmse,
                rmse,
                rf_rmse
            ]
        })

        fig_rmse = px.bar(
            rmse_comparison,
            x="Model",
            y="RMSE",
            color="Model",
            text="RMSE",
            title="RMSE Comparison (Lower Is Better)",
            labels={
                "Model": "Model",
                "RMSE": "RMSE (₹)"
            }
        )

        fig_rmse.update_traces(
            texttemplate="₹%{y:,.0f}",
            textposition="outside"
        )

        fig_rmse.update_layout(
            showlegend=False,
            yaxis_title="RMSE (₹)",
            xaxis_title="Model"
        )

        st.plotly_chart(fig_rmse, width="stretch")


        # Historical sales and forecast chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=monthly_sales["Month"],
                y=monthly_sales["Sales"],
                mode="lines+markers",
                name="Historical Sales"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast["Month"],
                y=forecast["ForecastSales"],
                mode="lines+markers",
                name="Forecast Sales",
                line=dict(dash="dash")
            )
        )

        fig.update_layout(
            title="Historical and Forecasted Monthly Sales",
            xaxis_title="Month",
            yaxis_title="Sales (₹)",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        # Forecast results table
        st.subheader("Monthly Sales Forecast")

        forecast_display = forecast[
            ["Month", "ForecastSales", "RandomForestForecast"]
        ].copy()

        forecast_display["Month"] = (
            forecast_display["Month"].dt.strftime("%B %Y")
        )

        forecast_display = forecast_display.rename(
            columns={
                "Month": "Forecast Month",
                "ForecastSales": "Linear Regression (₹)",
                "RandomForestForecast": "Random Forest (₹)"
            }
        )

        forecast_display["Linear Regression (₹)"] = (
            forecast_display["Linear Regression (₹)"].round(2)
        )
        forecast_display["Random Forest (₹)"] = (
            forecast_display["Random Forest (₹)"].round(2)
        )

        st.dataframe(
            forecast_display,
            width="stretch",
            hide_index=True
        )

        # Download forecast
        csv_data = forecast_display.to_csv(index=False).encode(
            "utf-8"
        )

        st.download_button(
            "Download Sales Forecast CSV",
            data=csv_data,
            file_name="retailpulse_sales_forecast.csv",
            mime="text/csv"
        )

        # Model evaluation
        st.subheader("Model Evaluation")

        evaluation = pd.DataFrame({
            "Month": test_data["Month"].dt.strftime("%B %Y"),
            "Actual Sales (₹)": test_data["Sales"].values,
            "Linear Regression Prediction (₹)": np.maximum(
                test_predictions, 0
        ),
        "Random Forest Prediction (₹)": np.maximum(
             rf_test_predictions, 0
        )
    })
           

        st.dataframe(
            evaluation.round(2),
            width="stretch",
            hide_index=True
        )

        st.caption(
            "Educational forecast based on synthetic retail data. "
            "Predictions are estimates, not guaranteed sales."
        )

        # Save forecast results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output_file = OUTPUT_DIR / "sales_forecast.csv"

        forecast.to_csv(output_file, index=False)

        st.success("Forecast generated successfully.")

    except (ValueError, KeyError, OSError) as error:
        st.error(f"Could not generate forecast: {error}")


if __name__ == "__main__":
    main()
