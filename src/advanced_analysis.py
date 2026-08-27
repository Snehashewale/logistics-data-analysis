import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================
# WEEK 3 - ADVANCED DATA ANALYSIS AND VISUALIZATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "logistics_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 65)
print("WEEK 3 - ADVANCED LOGISTICS DATA ANALYSIS")
print("=" * 65)

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


# Create analytical features
df["Delivery_Time"] = (
    df["Delivery_Date"] - df["Shipment_Date"]
).dt.days

df["Delayed"] = (
    df["Delivery_Date"] >
    df["Expected_Delivery_Date"]
).astype(int)

df["Delay_Days"] = (
    df["Delivery_Date"] -
    df["Expected_Delivery_Date"]
).dt.days

df["Cost_Per_KM"] = (
    df["Shipping_Cost"] /
    df["Distance"]
)

df["Shipment_Month"] = (
    df["Shipment_Date"].dt.to_period("M").astype(str)
)

df["Shipment_Day"] = (
    df["Shipment_Date"].dt.day_name()
)


# ============================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "-" * 65)
print("DESCRIPTIVE STATISTICS")
print("-" * 65)

numeric_columns = [
    "Distance",
    "Shipping_Cost",
    "Order_Quantity",
    "Delivery_Time",
    "Delay_Days",
    "Cost_Per_KM"
]

print(df[numeric_columns].describe())


# ============================================================
# 4. KEY LOGISTICS METRICS
# ============================================================

print("\n" + "-" * 65)
print("KEY LOGISTICS METRICS")
print("-" * 65)

total_shipments = len(df)

average_delivery_time = df["Delivery_Time"].mean()

average_shipping_cost = df["Shipping_Cost"].mean()

average_distance = df["Distance"].mean()

average_order_quantity = df["Order_Quantity"].mean()

delay_rate = df["Delayed"].mean() * 100

on_time_rate = 100 - delay_rate

average_cost_per_km = df["Cost_Per_KM"].mean()

print(f"Total Shipments        : {total_shipments}")
print(f"Average Delivery Time : {average_delivery_time:.2f} days")
print(f"Average Shipping Cost  : Rs. {average_shipping_cost:.2f}")
print(f"Average Distance       : {average_distance:.2f} km")
print(f"Average Order Quantity : {average_order_quantity:.2f}")
print(f"On-Time Delivery Rate  : {on_time_rate:.2f}%")
print(f"Delay Rate             : {delay_rate:.2f}%")
print(f"Average Cost/KM        : Rs. {average_cost_per_km:.2f}")


# ============================================================
# 5. CORRELATION ANALYSIS
# ============================================================

print("\n" + "-" * 65)
print("CORRELATION ANALYSIS")
print("-" * 65)

correlation = df[numeric_columns].corr()

print(correlation.round(2))

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Advanced Correlation Analysis of Logistics Variables")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "advanced_correlation_heatmap.png",
    dpi=300
)

plt.close()


# ============================================================
# 6. MONTHLY SHIPMENT TREND
# ============================================================

monthly_shipments = (
    df.groupby("Shipment_Month")
    .size()
)

plt.figure(figsize=(10, 6))

monthly_shipments.plot(
    kind="line",
    marker="o"
)

plt.title("Monthly Shipment Volume Trend")
plt.xlabel("Month")
plt.ylabel("Number of Shipments")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "monthly_shipment_trend.png",
    dpi=300
)

plt.close()


# ============================================================
# 7. MONTHLY SHIPPING COST TREND
# ============================================================

monthly_cost = (
    df.groupby("Shipment_Month")["Shipping_Cost"]
    .mean()
)

plt.figure(figsize=(10, 6))

monthly_cost.plot(
    kind="line",
    marker="o"
)

plt.title("Monthly Average Shipping Cost")
plt.xlabel("Month")
plt.ylabel("Average Shipping Cost (Rs.)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "monthly_cost_trend.png",
    dpi=300
)

plt.close()


# ============================================================
# 8. TRANSPORT MODE ANALYSIS
# ============================================================

transport_analysis = (
    df.groupby("Transport_Mode")
    .agg(
        Shipments=("Order_ID", "count"),
        Avg_Delivery_Time=("Delivery_Time", "mean"),
        Avg_Shipping_Cost=("Shipping_Cost", "mean"),
        Delay_Rate=("Delayed", "mean")
    )
)

transport_analysis["Delay_Rate"] *= 100

print("\nTransport Mode Analysis:")
print(transport_analysis.round(2))

plt.figure(figsize=(9, 6))

sns.barplot(
    data=transport_analysis.reset_index(),
    x="Transport_Mode",
    y="Avg_Delivery_Time"
)

plt.title("Average Delivery Time by Transport Mode")
plt.xlabel("Transport Mode")
plt.ylabel("Average Delivery Time (Days)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "transport_mode_analysis.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. VEHICLE DELIVERY ANALYSIS
# ============================================================

vehicle_analysis = (
    df.groupby("Vehicle_Type")
    .agg(
        Shipments=("Order_ID", "count"),
        Avg_Delivery_Time=("Delivery_Time", "mean"),
        Avg_Cost=("Shipping_Cost", "mean"),
        Delay_Rate=("Delayed", "mean")
    )
)

vehicle_analysis["Delay_Rate"] *= 100

print("\nVehicle Type Analysis:")
print(vehicle_analysis.round(2))

plt.figure(figsize=(9, 6))

sns.barplot(
    data=vehicle_analysis.reset_index(),
    x="Vehicle_Type",
    y="Avg_Delivery_Time"
)

plt.title("Average Delivery Time by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Average Delivery Time (Days)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "vehicle_delivery_analysis.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. DISTANCE VS DELIVERY TIME
# ============================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="Distance",
    y="Delivery_Time",
    hue="Transport_Mode",
    s=80
)

plt.title("Distance vs Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (Days)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "distance_vs_delivery_time.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. DISTANCE VS SHIPPING COST
# ============================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="Distance",
    y="Shipping_Cost",
    hue="Vehicle_Type",
    s=80
)

plt.title("Distance vs Shipping Cost")
plt.xlabel("Distance (km)")
plt.ylabel("Shipping Cost (Rs.)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "distance_vs_shipping_cost.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. ORDER QUANTITY VS SHIPPING COST
# ============================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="Order_Quantity",
    y="Shipping_Cost",
    hue="Transport_Mode",
    s=80
)

plt.title("Order Quantity vs Shipping Cost")
plt.xlabel("Order Quantity")
plt.ylabel("Shipping Cost (Rs.)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "quantity_vs_shipping_cost.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. DELAY ANALYSIS
# ============================================================

delay_counts = (
    df["Delayed"]
    .map({
        0: "On Time",
        1: "Delayed"
    })
    .value_counts()
)

plt.figure(figsize=(8, 6))

plt.pie(
    delay_counts.values,
    labels=delay_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("On-Time vs Delayed Shipments")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "delivery_delay_analysis.png",
    dpi=300
)

plt.close()


# ============================================================
# 14. ORIGIN-DESTINATION ANALYSIS
# ============================================================

route_analysis = (
    df.groupby(
        ["Origin", "Destination"]
    )
    .agg(
        Shipments=("Order_ID", "count"),
        Avg_Distance=("Distance", "mean"),
        Avg_Delivery_Time=("Delivery_Time", "mean"),
        Avg_Cost=("Shipping_Cost", "mean")
    )
    .reset_index()
)

route_analysis = route_analysis.sort_values(
    "Avg_Delivery_Time",
    ascending=False
)

print("\nTop Routes by Average Delivery Time:")
print(route_analysis.head(10).round(2))

route_plot = route_analysis.head(10).copy()

route_plot["Route"] = (
    route_plot["Origin"] +
    " → " +
    route_plot["Destination"]
)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=route_plot,
    y="Route",
    x="Avg_Delivery_Time"
)

plt.title("Top Routes by Average Delivery Time")
plt.xlabel("Average Delivery Time (Days)")
plt.ylabel("Route")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "origin_destination_analysis.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. SUMMARY TABLE
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "Total Shipments",
        "Average Delivery Time",
        "Average Shipping Cost",
        "Average Distance",
        "Average Order Quantity",
        "On-Time Delivery Rate",
        "Delay Rate",
        "Average Cost Per KM"
    ],
    "Value": [
        total_shipments,
        round(average_delivery_time, 2),
        round(average_shipping_cost, 2),
        round(average_distance, 2),
        round(average_order_quantity, 2),
        round(on_time_rate, 2),
        round(delay_rate, 2),
        round(average_cost_per_km, 2)
    ]
})

summary.to_csv(
    OUTPUT_DIR / "logistics_analysis_summary.csv",
    index=False
)


# ============================================================
# 16. SAVE ANALYTICAL DATA
# ============================================================

df.to_csv(
    OUTPUT_DIR / "week3_analyzed_logistics_data.csv",
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("WEEK 3 ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nGenerated files:")
print("- monthly_shipment_trend.png")
print("- monthly_cost_trend.png")
print("- transport_mode_analysis.png")
print("- vehicle_delivery_analysis.png")
print("- distance_vs_delivery_time.png")
print("- distance_vs_shipping_cost.png")
print("- quantity_vs_shipping_cost.png")
print("- origin_destination_analysis.png")
print("- delivery_delay_analysis.png")
print("- advanced_correlation_heatmap.png")
print("- logistics_analysis_summary.csv")
print("- week3_analyzed_logistics_data.csv")

print("\nOutput folder:")
print(OUTPUT_DIR)