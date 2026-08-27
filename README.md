# Logistics Delivery Performance Analysis and Prediction Using Python

## Project Overview

This project performs strategic logistics data analysis using Python to study delivery performance, transportation costs, and resource utilization. It applies data cleaning, exploratory data analysis, KPI calculation, data visualization, and machine learning to predict delivery time.

The project was developed as part of an internship task on Strategic Planning and Data Exploration in Logistics.

## Objectives

- Analyze logistics and delivery data.
- Calculate important logistics KPIs.
- Identify delivery delays.
- Study the relationship between distance, shipping cost, and delivery time.
- Visualize logistics performance using charts.
- Build a machine learning model to predict delivery time.

## Key Performance Indicators

1. **On-Time Delivery Rate** – Percentage of shipments delivered on or before the expected date.
2. **Average Delivery Time** – Average number of days required to complete a delivery.
3. **Average Shipping Cost** – Average transportation cost per shipment.
4. **Delivery Delay Rate** – Percentage of shipments delivered after the expected delivery date.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. KPI Calculation
5. Exploratory Data Analysis
6. Data Visualization
7. Machine Learning Model Training
8. Model Evaluation
9. Result Generation

## Machine Learning Model

A Random Forest Regressor is used to predict delivery time based on:

- Distance
- Shipping Cost
- Order Quantity

The model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Project Structure

```text
logistics-data-analysis/
│
├── data/
│   └── logistics_data.csv
│
├── outputs/
│   ├── delivery_time_distribution.png
│   ├── distance_vs_delivery_time.png
│   ├── shipping_cost_vs_distance.png
│   ├── average_cost_by_vehicle.png
│   ├── feature_importance.png
│   └── model_results.txt
│
├── src/
│   └── main.py
│
├── README.md
├── requirements.txt
└── .gitignore