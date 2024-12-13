import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import category_encoders as ce
import joblib
from babel.numbers import format_currency

# Load augmented data function with caching
@st.cache_data
def load_data():
    data = pd.read_csv("final_corrected_augmented_dataset.csv")
    return data

# Load standard dataset function with caching
@st.cache_data
def load_standard_data():
    standard_data = pd.read_csv("standardized_locations_dataset.csv")
    return standard_data

# Add weighted features based on proximity categories
def add_weighted_features(data):
    data['Weighted_Beachfront'] = (data['beach_proximity'] == 'Beachfront').astype(int) * 2.5
    data['Weighted_Seaview'] = (data['beach_proximity'] == 'Sea view').astype(int) * 2.0
    data['Weighted_Lakefront'] = (data['lake_proximity'] == 'Lakefront').astype(int) * 1.8
    data['Weighted_Lakeview'] = (data['lake_proximity'] == 'Lake view').astype(int) * 1.5
    return data

# Add Mean_Price_per_Cent based on training data
def add_location_mean_price(data):
    data['Price_per_cent'] = data['Price'] / data['Area']
    mean_price_per_location = (
        data.groupby("Location")['Price_per_cent']
        .mean()
        .rename("Mean_Price_per_Cent")
        .reset_index()
    )
    data = pd.merge(data, mean_price_per_location, on="Location", how="left")
    return data

# Predict function
def predict_price(model, training_data, area, location, beach_proximity, lake_proximity, density):
    # Map proximity inputs to weights
    beach_weights = {'Inland': 0, 'Sea view': 2.0, 'Beachfront': 2.5}
    lake_weights = {'Inland': 0, 'Lake view': 1.5, 'Lakefront': 1.8}

    # Calculate Mean_Price_per_Cent dynamically
    price_per_cent_mean = training_data.loc[training_data['Location'] == location, 'Price'].sum() / \
                          training_data.loc[training_data['Location'] == location, 'Area'].sum()
    if np.isnan(price_per_cent_mean):
        price_per_cent_mean = training_data['Price'].sum() / training_data['Area'].sum()

    # Calculate weights
    weighted_beachfront = beach_weights.get(beach_proximity, 0)
    weighted_seaview = beach_weights.get(beach_proximity, 0)
    weighted_lakefront = lake_weights.get(lake_proximity, 0)
    weighted_lakeview = lake_weights.get(lake_proximity, 0) * 0.75
    area_density = area * (1 if density == 'High' else 0)

    # Create input DataFrame
    input_data = pd.DataFrame([{
        'Area': area,
        'Mean_Price_per_Cent': price_per_cent_mean,
        'Weighted_Beachfront': weighted_beachfront,
        'Weighted_Seaview': weighted_seaview,
        'Weighted_Lakefront': weighted_lakefront,
        'Weighted_Lakeview': weighted_lakeview,
        'Area_Density': area_density,
        'density': density,
        'Location': location
    }])

    # Target encode location using category_encoders
    target_encoder = ce.TargetEncoder(cols=['Location'])
    target_encoder.fit(training_data['Location'], training_data['Price'])
    input_data['Location'] = target_encoder.transform(input_data[['Location']])

    # Predict
    predicted_price = model.predict(input_data)
    predicted_price = max(0, predicted_price[0])  # Ensure no negative prices
    return predicted_price

# Format in Indian numbering system
def format_inr(value):
    return format_currency(value, 'INR', locale='en_IN')  # Ensures proper Indian numbering system

# Streamlit UI
st.title("Real Estate Price Predictor (Augmented Dataset)")
st.write("Predict the price of plots based on features like location, proximity to amenities, and area.")

# Load and preprocess the data
data = load_data()
data = add_weighted_features(data)
data = add_location_mean_price(data)

# Load the standard dataset
standard_data = load_standard_data()

# Load the trained model
model = joblib.load('xgb_real_estate_model.pkl')

# User Inputs
area = st.number_input("Enter the area in cents:", min_value=1.0, step=0.1)
location = st.selectbox("Select the location:", options=sorted(data['Location'].unique()))
beach_proximity = st.selectbox("Select beach proximity:", options=['Inland', 'Sea view', 'Beachfront'])
lake_proximity = st.selectbox("Select lake proximity:", options=['Inland', 'Lake view', 'Lakefront'])
density = st.selectbox("Select density:", options=['Low', 'High'])

# Display Mean Price per Cent and Plot Count for the Location
if location:
    mean_price = standard_data.loc[standard_data['Location'] == location, 'Price'].sum() / \
                 standard_data.loc[standard_data['Location'] == location, 'Area'].sum()
    plot_count = standard_data.loc[standard_data['Location'] == location].shape[0]
    st.write(f"Mean Price per Cent for {location}: {format_inr(mean_price)}")
    st.write(f"Number of plots for {location}: {plot_count}")

# Predict Button
if st.button("Predict Price"):
    try:
        predicted_price = predict_price(model, data, area, location, beach_proximity, lake_proximity, density)
        st.success(f"Predicted Price for the plot: {format_inr(predicted_price)}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
