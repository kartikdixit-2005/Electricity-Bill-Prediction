
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Electricity Bill Prediction")

st.title("Electricity Bill Prediction")
st.write("Predict your electricity bill using Machine Learning.")

@st.cache_data
def load_data():
    return pd.read_csv("electricity_bill_dataset.csv")

df = load_data()

X = df[["MonthlyHours", "TariffRate"]]
y = df["ElectricityBill"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

st.subheader("Enter Your Details")

hours = st.number_input(
    "Monthly Usage Hours",
    min_value=0.0,
    value=300.0
)

tariff = st.number_input(
    "Tariff Rate (₹)",
    min_value=0.0,
    value=8.5
)

if st.button("Predict Electricity Bill"):
    prediction = model.predict(
        pd.DataFrame({
            "MonthlyHours": [hours],
            "TariffRate": [tariff]
        })
    )[0]

    st.success(f"Predicted Bill: ₹{prediction:.2f}")

st.subheader("Model Information")
st.write("Dataset rows:", len(df))
st.write("R² Score:", round(
    r2_score(y_test, model.predict(X_test)), 4
))

st.subheader("Dataset Preview")
st.dataframe(df.head())
