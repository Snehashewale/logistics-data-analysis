import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_data(file_path):
    """Load logistics dataset."""
    df = pd.read_csv(file_path)
    return df


def inspect_data(df):
    """Display basic information about the dataset."""
    print("\n" + "=" * 60)
    print("INITIAL DATA INSPECTION")
    print("=" * 60)

    print(f"Dataset Shape: {df.shape}")

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Records:")
    print(df.duplicated().sum())


def clean_data(df):
    """Clean and preprocess the logistics data."""

    # Remove duplicate records
    df = df.drop_duplicates().copy()

    # Convert date columns
    date_columns = [
        "Shipment_Date",
        "Delivery_Date",
        "Expected_Delivery_Date"
    ]

    for column in date_columns:
        df[column] = pd.to_datetime(df[column])

    # Handle numerical missing values
    numerical_columns = [
        "Distance",
        "Shipping_Cost",
        "Order_Quantity"
    ]

    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].median())

    # Handle categorical missing values
    categorical_columns = [
        "Transport_Mode",
        "Vehicle_Type",
        "Origin",
        "Destination"
    ]

    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    return df


def feature_engineering(df):
    """Create new logistics features."""

    # Delivery time in days
    df["Delivery_Time"] = (
        df["Delivery_Date"] -
        df["Shipment_Date"]
    ).dt.days

    # Delivery delay indicator
    df["Delayed"] = (
        df["Delivery_Date"] >
        df["Expected_Delivery_Date"]
    ).astype(int)

    # Cost per kilometre
    df["Cost_Per_KM"] = (
        df["Shipping_Cost"] /
        df["Distance"]
    )

    return df


def detect_outliers(df, column):
    """Detect outliers using IQR method."""

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    return outliers, lower_bound, upper_bound


def normalize_data(df):
    """Apply Min-Max normalization."""

    columns = [
        "Distance",
        "Shipping_Cost",
        "Order_Quantity"
    ]

    scaler = MinMaxScaler()

    normalized = df.copy()

    normalized[
        columns
    ] = scaler.fit_transform(
        normalized[columns]
    )

    return normalized


def standardize_data(df):
    """Apply standardization."""

    columns = [
        "Distance",
        "Shipping_Cost",
        "Order_Quantity"
    ]

    scaler = StandardScaler()

    standardized = df.copy()

    standardized[
        columns
    ] = scaler.fit_transform(
        standardized[columns]
    )

    return standardized


def save_data(df, file_path):
    """Save processed dataset."""
    df.to_csv(file_path, index=False)