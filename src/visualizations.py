import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(
    os.path.join(OUTPUT_FOLDER, "cleaned_logistics_data.csv")
)

# ------------------------------------------------------------
# 1. Missing Values Visualization
# ------------------------------------------------------------

missing = df.isnull().sum()

plt.figure(figsize=(10, 5))
missing.plot(kind="bar")
plt.title("Missing Values After Data Cleaning")
plt.xlabel("Features")
plt.ylabel("Number of Missing Values")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "missing_values_after_cleaning.png"
    )
)

plt.close()


# ------------------------------------------------------------
# 2. Distance Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Distance"],
    kde=True
)

plt.title("Distribution of Shipment Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Number of Shipments")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "distance_distribution.png"
    )
)

plt.close()


# ------------------------------------------------------------
# 3. Shipping Cost Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Shipping_Cost"],
    kde=True
)

plt.title("Distribution of Shipping Cost")
plt.xlabel("Shipping Cost (Rs.)")
plt.ylabel("Number of Shipments")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "shipping_cost_distribution.png"
    )
)

plt.close()


# ------------------------------------------------------------
# 4. Boxplot for Outlier Detection
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    y=df["Shipping_Cost"]
)

plt.title("Shipping Cost Outlier Detection")
plt.ylabel("Shipping Cost (Rs.)")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "shipping_cost_outliers.png"
    )
)

plt.close()


# ------------------------------------------------------------
# 5. Delivery Time Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Delivery_Time"],
    kde=True
)

plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Shipments")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "delivery_time_distribution_week2.png"
    )
)

plt.close()


# ------------------------------------------------------------
# 6. Correlation Heatmap
# ------------------------------------------------------------

numeric_columns = [
    "Distance",
    "Shipping_Cost",
    "Order_Quantity",
    "Delivery_Time",
    "Delayed",
    "Cost_Per_KM"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Matrix of Logistics Features")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "correlation_heatmap.png"
    )
)

plt.close()


print("=" * 60)
print("WEEK 2 VISUALIZATIONS GENERATED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated files:")

print("1. missing_values_after_cleaning.png")
print("2. distance_distribution.png")
print("3. shipping_cost_distribution.png")
print("4. shipping_cost_outliers.png")
print("5. delivery_time_distribution_week2.png")
print("6. correlation_heatmap.png")

print(f"\nOutput folder: {OUTPUT_FOLDER}")