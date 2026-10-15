
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Page Configuration
st.set_page_config(
    page_title="Electricity Bill Prediction",
    page_icon="⚡",
    layout="centered"
)

# App Title
st.title("⚡ Electricity Bill Prediction")
st.write("Predict your electricity bill using Machine Learning.")

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("electricity_bill_dataset.csv")

df = load_data()

# Select Features and Target
X = df[["MonthlyHours", "TariffRate"]]
y = df["ElectricityBill"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# User Input Section
st.subheader("Enter Your Details")

hours = st.number_input(
    "Monthly Usage Hours",
    min_value=0.0,
    value=300.0,
    step=10.0
)

tariff = st.number_input(
    "Tariff Rate (₹ per unit)",
    min_value=0.0,
    value=8.5,
    step=0.5
)

# Prediction
if st.button("Predict Electricity Bill", type="primary"):
    prediction = model.predict(
        pd.DataFrame({
            "MonthlyHours": [hours],
            "TariffRate": [tariff]
        })
    )[0]

    st.success(f"Predicted Bill: ₹{prediction:.2f}")

# Model Information
st.subheader("Model Information")

st.write("Dataset Rows:", len(df))

