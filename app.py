import os

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from src.regression_model import train_regression
from src.classification_model import train_classification
from src.datacleaning import merge_data, encode_features

st.title("Tourism Experience Analytics")

if not os.path.exists("models"):
    os.makedirs("models")

if not os.path.exists("models/regression.pkl"):
    train_regression()

if not os.path.exists("models/classification.pkl"):
    train_classification()

reg_model = joblib.load("models/regression.pkl")
clf_model = joblib.load("models/classification.pkl")

continent = st.number_input("Continent (Encoded)")
region = st.number_input("Region (Encoded)")
country = st.number_input("Country (Encoded)")
city = st.number_input("City (Encoded)")
year = st.number_input("Visit Year")
month = st.number_input("Visit Month")
atype = st.number_input("Attraction Type (Encoded)")

if st.button("Predict"):
    input_data = np.array([[continent, region, country, city, year, month, atype]])

    rating_pred = reg_model.predict(input_data)
    mode_pred = clf_model.predict(input_data)

    st.write("Predicted Rating:", rating_pred[0])
    st.write("Predicted Visit Mode (Encoded):", mode_pred[0])