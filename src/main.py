import os
from preprocessing import (
    load_data,
    inspect_data,
    clean_data,
    feature_engineering,
    detect_outliers,
    normalize_data,
    standardize_data,
    save_data
)


# ============================================================
# WEEK 2 - DATA COLLECTION, CLEANING AND PREPROCESSING
# ============================================================

print("=" * 60)
print("WEEK 2 - LOGISTICS DATA PREPROCESSING")
print("=" * 60)


# File paths
DATA_PATH = "data/logistics_data.csv"
OUTPUT_FOLDER = "outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 1. DATA COLLECTION
# ============================================================

print("\n" + "-" * 60)
print("1. DATA COLLECTION")
print("-" * 60)

df = load_data(DATA_PATH)

print("Dataset loaded successfully!")
print(f"Number of records: {len(df)}")
print(f"Number of columns: {len(df.columns)}")


# ============================================================
# 2. INITIAL DATA INSPECTION
# ============================================================

inspect_data(df)


# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\n" + "-" * 60)
print("2. DATA CLEANING")
print("-" * 60)

original_rows = len(df)

df = clean_data(df)

print(f"Original records: {original_rows}")
print(f"Records after cleaning: {len(df)}")

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate records after cleaning:")
print(df.duplicated().sum())


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

print("\n" + "-" * 60)
print("3. FEATURE ENGINEERING")
print("-" * 60)

df = feature_engineering(df)

print("Created features:")
print("- Delivery_Time")
print("- Delayed")
print("- Cost_Per_KM")

print("\nSample processed data:")
print(
    df[
        [
            "Order_ID",
            "Delivery_Time",
            "Delayed",
            "Cost_Per_KM"
        ]
    ].head()
)


# ============================================================
# 5. OUTLIER DETECTION
# ============================================================

print("\n" + "-" * 60)
print("4. OUTLIER DETECTION")
print("-" * 60)

numeric_columns = [
    "Distance",
    "Shipping_Cost",
    "Order_Quantity",
    "Delivery_Time"
]

outlier_results = []

for column in numeric_columns:

    outliers, lower, upper = detect_outliers(
        df,
        column
    )

    print(f"\n{column}:")
    print(f"Lower Bound: {lower:.2f}")
    print(f"Upper Bound: {upper:.2f}")
    print(f"Number of Outliers: {len(outliers)}")

    outlier_results.append(
        {
            "Feature": column,
            "Lower_Bound": lower,
            "Upper_Bound": upper,
            "Outlier_Count": len(outliers)
        }
    )


# Save outlier report
import pandas as pd

outlier_df = pd.DataFrame(outlier_results)

outlier_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "outlier_report.csv"
    ),
    index=False
)


# ============================================================
# 6. NORMALIZATION
# ============================================================

print("\n" + "-" * 60)
print("5. MIN-MAX NORMALIZATION")
print("-" * 60)

normalized_df = normalize_data(df)

print(
    normalized_df[
        [
            "Distance",
            "Shipping_Cost",
            "Order_Quantity"
        ]
    ].head()
)

save_data(
    normalized_df,
    os.path.join(
        OUTPUT_FOLDER,
        "normalized_logistics_data.csv"
    )
)

print(
    "\nNormalized dataset saved successfully!"
)


# ============================================================
# 7. STANDARDIZATION
# ============================================================

print("\n" + "-" * 60)
print("6. STANDARDIZATION")
print("-" * 60)

standardized_df = standardize_data(df)

print(
    standardized_df[
        [
            "Distance",
            "Shipping_Cost",
            "Order_Quantity"
        ]
    ].head()
)

save_data(
    standardized_df,
    os.path.join(
        OUTPUT_FOLDER,
        "standardized_logistics_data.csv"
    )
)

print(
    "\nStandardized dataset saved successfully!"
)


# ============================================================
# 8. SAVE CLEAN DATA
# ============================================================

print("\n" + "-" * 60)
print("7. SAVE CLEAN DATASET")
print("-" * 60)

save_data(
    df,
    os.path.join(
        OUTPUT_FOLDER,
        "cleaned_logistics_data.csv"
    )
)

print(
    "Cleaned dataset saved successfully!"
)


# ============================================================
# 9. FINAL VALIDATION
# ============================================================

print("\n" + "-" * 60)
print("8. FINAL DATA VALIDATION")
print("-" * 60)

print(f"Final dataset shape: {df.shape}")

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

print("\nFinal columns:")
print(list(df.columns))


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("WEEK 2 PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated files:")
print("1. cleaned_logistics_data.csv")
print("2. normalized_logistics_data.csv")
print("3. standardized_logistics_data.csv")
print("4. outlier_report.csv")

print(f"\nOutput folder: {OUTPUT_FOLDER}")