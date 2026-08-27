import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# WEEK 1 - CLUSTERING ANALYSIS
# LOGISTICS SHIPMENT SEGMENTATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "logistics_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


print("=" * 65)
print("WEEK 1 - LOGISTICS SHIPMENT CLUSTERING ANALYSIS")
print("=" * 65)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. DATA PREPARATION
# ============================================================

required_columns = [
    "Distance",
    "Shipping_Cost",
    "Order_Quantity"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# Convert numerical columns
for column in required_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Remove rows with missing values
df = df.dropna(
    subset=required_columns
).copy()


print("\nRecords available for clustering:", len(df))


# ============================================================
# 3. SELECT CLUSTERING FEATURES
# ============================================================

features = [
    "Distance",
    "Shipping_Cost",
    "Order_Quantity"
]

X = df[features].copy()

print("\nClustering Features:")
for feature in features:
    print("-", feature)


# ============================================================
# 4. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeatures standardized successfully.")


# ============================================================
# 5. FIND OPTIMAL NUMBER OF CLUSTERS
# ============================================================

print("\n" + "-" * 65)
print("TESTING DIFFERENT NUMBERS OF CLUSTERS")
print("-" * 65)

inertia_values = []
silhouette_values = []

cluster_range = range(2, 6)

for k in cluster_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertia_values.append(
        model.inertia_
    )

    silhouette_values.append(
        silhouette_score(
            X_scaled,
            labels
        )
    )

    print(
        f"K = {k} | "
        f"Inertia = {model.inertia_:.2f} | "
        f"Silhouette Score = "
        f"{silhouette_values[-1]:.4f}"
    )


# ============================================================
# 6. SELECT BEST K
# ============================================================

best_index = np.argmax(
    silhouette_values
)

best_k = list(cluster_range)[best_index]

print(
    f"\nBest number of clusters: {best_k}"
)

print(
    f"Best Silhouette Score: "
    f"{silhouette_values[best_index]:.4f}"
)


# ============================================================
# 7. ELBOW / INERTIA PLOT
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    list(cluster_range),
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.title(
    "Elbow Method for Logistics Shipment Clustering"
)

plt.xticks(list(cluster_range))

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "clustering_elbow_method.png",
    dpi=300
)

plt.close()


# ============================================================
# 8. SILHOUETTE SCORE PLOT
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    list(cluster_range),
    silhouette_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")

plt.title(
    "Silhouette Score for Logistics Shipment Clustering"
)

plt.xticks(list(cluster_range))

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "clustering_silhouette_scores.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. FINAL K-MEANS MODEL
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(
    X_scaled
)


# ============================================================
# 10. CLUSTER SUMMARY
# ============================================================

cluster_summary = (
    df.groupby("Cluster")
    .agg(
        Shipments=("Order_ID", "count"),
        Avg_Distance=("Distance", "mean"),
        Avg_Shipping_Cost=(
            "Shipping_Cost",
            "mean"
        ),
        Avg_Order_Quantity=(
            "Order_Quantity",
            "mean"
        )
    )
    .reset_index()
)


print("\n" + "=" * 65)
print("CLUSTER SUMMARY")
print("=" * 65)

print(
    cluster_summary.round(2)
)


# ============================================================
# 11. ASSIGN CLUSTER DESCRIPTIONS
# ============================================================

overall_distance = df["Distance"].median()
overall_cost = df["Shipping_Cost"].median()
overall_quantity = df["Order_Quantity"].median()


def classify_cluster(row):

    if (
        row["Avg_Distance"] <= overall_distance
        and
        row["Avg_Shipping_Cost"] <= overall_cost
        and
        row["Avg_Order_Quantity"] <= overall_quantity
    ):
        return "Short-Distance / Low-Volume"

    elif (
        row["Avg_Distance"] > overall_distance
        and
        row["Avg_Shipping_Cost"] > overall_cost
    ):
        return "Long-Distance / High-Cost"

    else:
        return "Medium / Mixed Shipments"


cluster_summary["Cluster_Type"] = (
    cluster_summary.apply(
        classify_cluster,
        axis=1
    )
)


print("\nCluster Interpretation:")

for _, row in cluster_summary.iterrows():

    print(
        f"Cluster {int(row['Cluster'])}: "
        f"{row['Cluster_Type']}"
    )


# ============================================================
# 12. SAVE CLUSTER SUMMARY
# ============================================================

cluster_summary.to_csv(
    OUTPUT_DIR / "logistics_cluster_summary.csv",
    index=False
)


# ============================================================
# 13. SAVE SHIPMENT-LEVEL CLUSTERS
# ============================================================

df.to_csv(
    OUTPUT_DIR / "logistics_clustered_data.csv",
    index=False
)


# ============================================================
# 14. CLUSTER VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 7))

for cluster in sorted(
    df["Cluster"].unique()
):

    cluster_data = df[
        df["Cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["Distance"],
        cluster_data["Shipping_Cost"],
        s=80,
        label=f"Cluster {cluster}"
    )


plt.xlabel("Distance (km)")
plt.ylabel("Shipping Cost (Rs.)")

plt.title(
    "Logistics Shipment Clusters"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "logistics_shipment_clusters.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. CLUSTER SIZE VISUALIZATION
# ============================================================

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)


plt.figure(figsize=(9, 6))

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.xlabel("Cluster")
plt.ylabel("Number of Shipments")

plt.title(
    "Number of Shipments in Each Logistics Cluster"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "cluster_size_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. BUSINESS RECOMMENDATIONS
# ============================================================

recommendations = []

for _, row in cluster_summary.iterrows():

    cluster = int(row["Cluster"])
    cluster_type = row["Cluster_Type"]

    if "Long-Distance" in cluster_type:

        recommendation = (
            "Prioritize route optimization, "
            "vehicle capacity planning, and "
            "delivery-time monitoring."
        )

    elif "Short-Distance" in cluster_type:

        recommendation = (
            "Use efficient local delivery vehicles "
            "and consider shipment consolidation."
        )

    else:

        recommendation = (
            "Monitor shipment characteristics and "
            "optimize vehicle assignment based on "
            "distance, quantity, and cost."
        )

    recommendations.append({
        "Cluster": cluster,
        "Cluster_Type": cluster_type,
        "Recommendation": recommendation
    })


recommendation_df = pd.DataFrame(
    recommendations
)

recommendation_df.to_csv(
    OUTPUT_DIR / "clustering_recommendations.csv",
    index=False
)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("CLUSTERING ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nGenerated files:")

print("- clustering_elbow_method.png")
print("- clustering_silhouette_scores.png")
print("- logistics_shipment_clusters.png")
print("- cluster_size_distribution.png")
print("- logistics_cluster_summary.csv")
print("- logistics_clustered_data.csv")
print("- clustering_recommendations.csv")

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\n" + "=" * 65)