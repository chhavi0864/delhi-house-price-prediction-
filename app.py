import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="Delhi House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("house_price_pipeline.pkl")

@st.cache_data
def load_categories():
    with open("category_options.json") as file:
        return json.load(file)

model = load_model()
categories = load_categories()

st.title("🏠 Delhi House Price Predictor")

st.write(
    "Estimate the price of a residential property in Delhi "
    "using a Machine Learning model trained on historical housing listings."
)

st.divider()

st.subheader("Enter Property Details")

area = st.number_input(
    "Property Area (sq. ft.)",
    min_value=1.0,
    max_value=25000.0,
    value=1200.0,
    step=50.0
)

bhk = st.number_input(
    "Bedrooms (BHK)",
    min_value=1,
    max_value=20,
    value=3
)

bathroom = st.number_input(
    "Bathrooms",
    min_value=0,
    max_value=20,
    value=2
)

parking = st.number_input(
    "Parking Spaces",
    min_value=0,
    max_value=20,
    value=1
)

locality = st.selectbox(
    "Locality",
    categories["Locality"]
)

furnishing = st.selectbox(
    "Furnishing",
    categories["Furnishing"]
)

status = st.selectbox(
    "Property Status",
    categories["Status"]
)

transaction = st.selectbox(
    "Transaction Type",
    categories["Transaction"]
)

property_type = st.selectbox(
    "Property Type",
    categories["Type"]
)

st.divider()

if st.button("Predict House Price", use_container_width=True):

    input_data = pd.DataFrame([{
        "Area": area,
        "BHK": bhk,
        "Bathroom": bathroom,
        "Furnishing": furnishing,
        "Locality": locality,
        "Parking": parking,
        "Status": status,
        "Transaction": transaction,
        "Type": property_type
    }])

    prediction = float(model.predict(input_data)[0])

    st.success("Prediction completed!")

    if prediction <= 0:
        st.warning(
            "The model returned an invalid price estimate."
        )
    else:
        st.metric(
            "Estimated Property Price",
            f"₹{prediction:,.0f}"
        )

        if prediction >= 10000000:
            st.info(
                f"Approximately ₹{prediction / 10000000:.2f} Crore"
            )
        else:
            st.info(
                f"Approximately ₹{prediction / 100000:.2f} Lakh"
            )

st.divider()

st.caption(
    "Academic ML project | Dataset: Delhi MagicBricks | "
    "Model: Random Forest Regressor | "
    "Predictions are historical estimates, not current market valuations."
)
