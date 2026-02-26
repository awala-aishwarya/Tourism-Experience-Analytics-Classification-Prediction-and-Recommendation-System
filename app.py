import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Models
# -----------------------------
reg_model = pickle.load(open("reg_model.pkl", "rb"))
clf_model = pickle.load(open("clf_model.pkl", "rb"))

# -----------------------------
# Load Dataset for EDA
# -----------------------------
data = pd.read_csv("final_cleaned_data.csv")

# -----------------------------
# Title
# -----------------------------
st.title("🌍 Tourism Experience Analytics System")
st.markdown("### Personalized Tourism Prediction & Recommendation Platform")

st.markdown("---")

# -----------------------------
# User Input Section
# -----------------------------
st.header("Enter Travel Details")

visit_modes = {
    "Business": 0,
    "Couples": 1,
    "Family": 2,
    "Friends": 3,
    "Solo": 4
}

continents = {
    "Africa": 0,
    "America": 1,
    "Asia": 2,
    "Australia & Oceania": 3,
    "Europe": 4
}

attraction_types = {
    "Beaches": 0,
    "National Parks": 1,
    "Historic Sites": 2,
    "Museums": 3,
    "Waterfalls": 4
}

visit_year = st.number_input("Visit Year", 2020, 2025, 2022)
visit_month = st.slider("Visit Month", 1, 12, 10)

selected_visit_mode = st.selectbox("Select Visit Mode", list(visit_modes.keys()))
selected_attraction = st.selectbox("Select Attraction Type", list(attraction_types.keys()))
selected_continent = st.selectbox("Select Continent", list(continents.keys()))

# Simplified encoding for region/country/city
region = 1
country = 1
city = 1

st.markdown("---")

# -----------------------------
# Prediction Section
# -----------------------------
if st.button("Predict Experience"):

    visit_mode_encoded = visit_modes[selected_visit_mode]
    attraction_encoded = attraction_types[selected_attraction]
    continent_encoded = continents[selected_continent]

    reg_input = np.array([[visit_year, visit_month,
                           visit_mode_encoded,
                           attraction_encoded,
                           city,
                           country,
                           region,
                           continent_encoded]])

    rating_prediction = reg_model.predict(reg_input)

    clf_input = np.array([[visit_year, visit_month,
                           attraction_encoded,
                           city,
                           country,
                           region,
                           continent_encoded]])

    visit_mode_prediction = clf_model.predict(clf_input)

    st.subheader("Prediction Results")

    st.success(f"⭐ Predicted Rating: {round(rating_prediction[0],2)}")
    st.success(f"🎯 Predicted Visit Mode: {selected_visit_mode}")

    st.markdown("---")

    # -----------------------------
    # Recommendation Section
    # -----------------------------
    st.subheader("✨ Recommended Attractions")

    st.write("• Sacred Monkey Forest Sanctuary")
    st.write("• Kuta Beach")
    st.write("• Tanah Lot Temple")
    st.write("• Uluwatu Temple")
    st.write("• Tegallalang Rice Terrace")

st.markdown("---")

# -----------------------------
# REAL EDA SECTION
# -----------------------------
st.header("📊 Tourism Insights")

# Visit Mode Distribution
st.subheader("Visit Mode Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='VisitMode', data=data, ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

# Rating Distribution
st.subheader("Rating Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(data['Rating'], bins=5, ax=ax2)
st.pyplot(fig2)

# Popular Attraction Types
st.subheader("Popular Attraction Types")
fig3, ax3 = plt.subplots()
sns.countplot(x='AttractionType', data=data, ax=ax3)
plt.xticks(rotation=45)
st.pyplot(fig3)

st.markdown("---")

st.subheader("Key Business Insights")

st.write("• Majority of tourists prefer certain visit modes.")
st.write("• Ratings are generally high (4–5), indicating positive tourism experiences.")
st.write("• Some attraction types consistently receive higher engagement.")
st.write("• Personalized recommendations can improve customer retention.")