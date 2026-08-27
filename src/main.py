import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------------------------
# 1. SET PATHS
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "logistics_data.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------
# 2. LOAD DATA
# -------------------------------------------------

print("\n" + "=" * 60)
print("LOGISTICS DATA ANALYSIS AND DELIVERY TIME PREDICTION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print(f"Dataset shape: {df.shape}")

print("\nFirst 5 records:")
print(df.head())


# -------------------------------------------------
# 3. DATA CLEANING
# -------------------------------------------------

print("\n" + "-" * 60)
print("DATA CLEANING")
print("-" * 60)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Convert date columns
date_columns = [
    "Shipment_Date",
    "Delivery_Date",
    "Expected_Delivery_Date"
]

for column in date_columns:
    df[column] = pd.to_datetime(df[column])


# -------------------------------------------------
# 4. FEATURE ENGINEERING
# -------------------------------------------------

# Calculate delivery time in days
df["Delivery_Time"] = (
    df["Delivery_Date"] - df["Shipment_Date"]
).dt.days

# Identify delayed deliveries
df["Delayed"] = (
    df["Delivery_Date"] >
    df["Expected_Delivery_Date"]
).astype(int)

# Calculate cost per kilometer
df["Cost_Per_KM"] = (
    df["Shipping_Cost"] / df["Distance"]
)


# -------------------------------------------------
# 5. KPI CALCULATION
# -------------------------------------------------

print("\n" + "-" * 60)
print("KEY PERFORMANCE INDICATORS")
print("-" * 60)

total_shipments = len(df)

on_time_deliveries = (df["Delayed"] == 0).sum()

on_time_delivery_rate = (
    on_time_deliveries / total_shipments
) * 100

average_delivery_time = df["Delivery_Time"].mean()

average_shipping_cost = df["Shipping_Cost"].mean()

delay_rate = df["Delayed"].mean() * 100

print(f"\nTotal Shipments: {total_shipments}")
print(f"On-Time Delivery Rate: {on_time_delivery_rate:.2f}%")
print(f"Average Delivery Time: {average_delivery_time:.2f} days")
print(f"Average Shipping Cost: Rs. {average_shipping_cost:.2f}")
print(f"Delivery Delay Rate: {delay_rate:.2f}%")


# -------------------------------------------------
# 6. EXPLORATORY DATA ANALYSIS
# -------------------------------------------------

print("\nGenerating visualizations...")


# Chart 1: Delivery Time Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    df["Delivery_Time"],
    bins=6,
    kde=True
)

plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Shipments")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "delivery_time_distribution.png"
    )
)

plt.close()


# Chart 2: Distance vs Delivery Time
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Distance",
    y="Delivery_Time",
    hue="Delayed",
    s=80
)

plt.title("Distance vs Delivery Time")
plt.xlabel("Distance (KM)")
plt.ylabel("Delivery Time (Days)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "distance_vs_delivery_time.png"
    )
)

plt.close()


# Chart 3: Shipping Cost vs Distance
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Distance",
    y="Shipping_Cost",
    hue="Vehicle_Type",
    s=80
)

plt.title("Shipping Cost vs Distance")
plt.xlabel("Distance (KM)")
plt.ylabel("Shipping Cost (Rs.)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "shipping_cost_vs_distance.png"
    )
)

plt.close()


# Chart 4: Average Shipping Cost by Vehicle Type
vehicle_cost = (
    df.groupby("Vehicle_Type")["Shipping_Cost"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(7, 5))

sns.barplot(
    data=vehicle_cost,
    x="Vehicle_Type",
    y="Shipping_Cost"
)

plt.title("Average Shipping Cost by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Average Shipping Cost (Rs.)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "average_cost_by_vehicle.png"
    )
)

plt.close()


# -------------------------------------------------
# 7. PREPARE DATA FOR MACHINE LEARNING
# -------------------------------------------------

print("\n" + "-" * 60)
print("MACHINE LEARNING: DELIVERY TIME PREDICTION")
print("-" * 60)

features = [
    "Distance",
    "Shipping_Cost",
    "Order_Quantity"
]

X = df[features]

y = df["Delivery_Time"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# -------------------------------------------------
# 8. TRAIN RANDOM FOREST MODEL
# -------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(X_test)


# -------------------------------------------------
# 9. MODEL EVALUATION
# -------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print(f"\nMean Absolute Error: {mae:.2f} days")
print(f"Root Mean Squared Error: {rmse:.2f} days")
print(f"R2 Score: {r2:.2f}")


# -------------------------------------------------
# 10. FEATURE IMPORTANCE
# -------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)


plt.figure(figsize=(7, 5))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance for Delivery Time Prediction")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "feature_importance.png"
    )
)

plt.close()


# -------------------------------------------------
# 11. SAVE RESULTS
# -------------------------------------------------

results_path = os.path.join(
    OUTPUT_DIR,
    "model_results.txt"
)

with open(results_path, "w") as file:

    file.write(
        "LOGISTICS DATA ANALYSIS RESULTS\n"
    )

    file.write(
        "=" * 40 + "\n\n"
    )

    file.write(
        f"Total Shipments: {total_shipments}\n"
    )

    file.write(
        f"On-Time Delivery Rate: "
        f"{on_time_delivery_rate:.2f}%\n"
    )

    file.write(
        f"Average Delivery Time: "
        f"{average_delivery_time:.2f} days\n"
    )

    file.write(
        f"Average Shipping Cost: "
        f"Rs. {average_shipping_cost:.2f}\n"
    )

    file.write(
        f"Delivery Delay Rate: "
        f"{delay_rate:.2f}%\n\n"
    )

    file.write(
        "MODEL PERFORMANCE\n"
    )

    file.write(
        "-" * 25 + "\n"
    )

    file.write(
        f"Mean Absolute Error: {mae:.2f} days\n"
    )

    file.write(
        f"Root Mean Squared Error: {rmse:.2f} days\n"
    )

    file.write(
        f"R2 Score: {r2:.2f}\n"
    )


print("\nResults saved successfully!")
print(f"Output folder: {OUTPUT_DIR}")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)