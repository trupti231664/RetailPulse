
import mlflow

# Use SQLite as the MLflow tracking database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Create or select the RetailPulse experiment
mlflow.set_experiment("RetailPulse-Demand-Forecasting")

# Record a sample experiment
with mlflow.start_run(run_name="MLflow-Setup-Test"):
    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("test_type", "setup_check")
    mlflow.log_metric("sample_mae", 100.0)

print("MLflow experiment tracking is working!")
print("Experiment: RetailPulse-Demand-Forecasting")
print("Database: mlflow.db")
