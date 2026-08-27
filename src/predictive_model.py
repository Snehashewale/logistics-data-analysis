import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# WEEK 4 - PREDICTIVE MODELING AND OPTIMIZATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "logistics_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("WEEK 4 - PREDICTIVE MODELING AND LOGISTICS OPTIMIZATION")
print("=" * 65)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. DATA PREPARATION
# ============================================================

date_columns = [
    "Shipment_Date",
    "Delivery_Date",
    "Expected_Delivery_Date"
]

for column in date_columns:
    df[column] = pd.to_datetime(df[column])


# Create target variable
df["Delivery_Time"] = (
    df["Delivery_Date"] -
    df["Shipment_Date"]
).dt.days


# Additional features
df["Distance"] = pd.to_numeric(
    df["Distance"],
    errors="coerce"
)

df["Shipping_Cost"] = pd.to_numeric(
    df["Shipping_Cost"],
    errors="coerce"
)

df["Order_Quantity"] = pd.to_numeric(
    df["Order_Quantity"],
    errors="coerce"
)


# Remove incomplete records
df = df.dropna(
    subset=[
        "Distance",
        "Shipping_Cost",
        "Order_Quantity",
        "Delivery_Time"
    ]
)


# ============================================================
# 3. DEFINE FEATURES AND TARGET
# ============================================================

features = [
    "Distance",
    "Shipping_Cost",
    "Order_Quantity"
]

X = df[features]
y = df["Delivery_Time"]

print("\nFeatures:")
print(features)

print("\nTarget:")
print("Delivery_Time")


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))


# ============================================================
# 5. LINEAR REGRESSION
# ============================================================

print("\n" + "-" * 65)
print("MODEL 1: LINEAR REGRESSION")
print("-" * 65)

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(
    X_test
)


linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print(f"MAE  : {linear_mae:.4f} days")
print(f"RMSE : {linear_rmse:.4f} days")
print(f"R2   : {linear_r2:.4f}")


# ============================================================
# 6. RANDOM FOREST REGRESSION
# ============================================================

print("\n" + "-" * 65)
print("MODEL 2: RANDOM FOREST REGRESSION")
print("-" * 65)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = rf_model.predict(
    X_test
)


rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print(f"MAE  : {rf_mae:.4f} days")
print(f"RMSE : {rf_rmse:.4f} days")
print(f"R2   : {rf_r2:.4f}")


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        linear_mae,
        rf_mae
    ],
    "RMSE": [
        linear_rmse,
        rf_rmse
    ],
    "R2": [
        linear_r2,
        rf_r2
    ]
})

print("\n" + "-" * 65)
print("MODEL COMPARISON")
print("-" * 65)

print(comparison.round(4))


comparison.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)


# ============================================================
# 8. SELECT BEST MODEL
# ============================================================

if rf_mae < linear_mae:
    best_model = rf_model
    best_predictions = rf_predictions
    best_model_name = "Random Forest"
    best_mae = rf_mae
    best_rmse = rf_rmse
    best_r2 = rf_r2
else:
    best_model = linear_model
    best_predictions = linear_predictions
    best_model_name = "Linear Regression"
    best_mae = linear_mae
    best_rmse = linear_rmse
    best_r2 = linear_r2


print("\nBest Model:", best_model_name)
print(f"Best MAE : {best_mae:.4f}")
print(f"Best RMSE: {best_rmse:.4f}")
print(f"Best R2  : {best_r2:.4f}")


# ============================================================
# 9. CROSS VALIDATION
# ============================================================

print("\n" + "-" * 65)
print("CROSS-VALIDATION")
print("-" * 65)

cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("Cross-validation R2 scores:")
print(np.round(cv_scores, 4))

print(
    f"Average CV R2: "
    f"{cv_scores.mean():.4f}"
)


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

if best_model_name == "Random Forest":

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": best_model.feature_importances_
    })

else:

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": np.abs(
            best_model.coef_
        )
    })

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n" + "-" * 65)
print("FEATURE IMPORTANCE")
print("-" * 65)

print(importance)


importance.to_csv(
    OUTPUT_DIR / "week4_feature_importance.csv",
    index=False
)


# ============================================================
# 11. ACTUAL VS PREDICTED
# ============================================================

results = pd.DataFrame({
    "Actual_Delivery_Time": y_test.values,
    "Predicted_Delivery_Time": best_predictions
})

results["Prediction_Error"] = (
    results["Actual_Delivery_Time"] -
    results["Predicted_Delivery_Time"]
)

results.to_csv(
    OUTPUT_DIR / "delivery_time_predictions.csv",
    index=False
)


plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    best_predictions,
    s=80
)

min_value = min(
    y_test.min(),
    best_predictions.min()
)

max_value = max(
    y_test.max(),
    best_predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Delivery Time (Days)")
plt.ylabel("Predicted Delivery Time (Days)")

plt.title(
    f"Actual vs Predicted Delivery Time - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted_delivery_time.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. PREDICTION ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 6))

plt.hist(
    results["Prediction_Error"],
    bins=8
)

plt.xlabel("Prediction Error (Days)")
plt.ylabel("Frequency")

plt.title("Delivery Time Prediction Error Distribution")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "prediction_error_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. LOGISTICS OPTIMIZATION ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("LOGISTICS OPTIMIZATION ANALYSIS")
print("=" * 65)


# Predict delivery time for all shipments
df["Predicted_Delivery_Time"] = (
    best_model.predict(X)
)


# Calculate expected delay risk
df["Delay_Risk"] = np.where(
    df["Predicted_Delivery_Time"] >
    (
        df["Expected_Delivery_Date"] -
        df["Shipment_Date"]
    ).dt.days,
    "High",
    "Normal"
)


# Cost efficiency
df["Cost_Per_KM"] = (
    df["Shipping_Cost"] /
    df["Distance"]
)


# Optimization recommendations
high_risk_shipments = df[
    df["Delay_Risk"] == "High"
]

high_cost_shipments = df[
    df["Cost_Per_KM"] >
    df["Cost_Per_KM"].median()
]


print(
    "\nHigh Delay Risk Shipments:",
    len(high_risk_shipments)
)

print(
    "High Cost Per KM Shipments:",
    len(high_cost_shipments)
)


# ============================================================
# 14. OPTIMIZATION SUMMARY
# ============================================================

optimization_summary = pd.DataFrame({
    "Optimization_Area": [
        "Delay Risk Monitoring",
        "Route Planning",
        "Vehicle Allocation",
        "Transportation Cost"
    ],
    "Recommendation": [
        "Prioritize shipments with high predicted delivery time.",
        "Review long-distance routes with high delivery duration.",
        "Prefer faster vehicle types when delivery deadlines are strict.",
        "Monitor shipments with above-median cost per kilometre."
    ]
})

optimization_summary.to_csv(
    OUTPUT_DIR / "optimization_recommendations.csv",
    index=False
)


# ============================================================
# 15. SAVE FINAL ANALYTICAL DATA
# ============================================================

df.to_csv(
    OUTPUT_DIR / "week4_prediction_analysis.csv",
    index=False
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("WEEK 4 COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nBest Model:", best_model_name)
print(f"MAE : {best_mae:.4f} days")
print(f"RMSE: {best_rmse:.4f} days")
print(f"R2  : {best_r2:.4f}")

print("\nGenerated files:")
print("- model_comparison.csv")
print("- week4_feature_importance.csv")
print("- delivery_time_predictions.csv")
print("- actual_vs_predicted_delivery_time.png")
print("- prediction_error_distribution.png")
print("- optimization_recommendations.csv")
print("- week4_prediction_analysis.csv")

print("\nOutput folder:")
print(OUTPUT_DIR)