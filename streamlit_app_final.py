# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:00:59 2025

@author: assay
"""

# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import joblib
import pandas as pd

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# Load the trained model
ML_model = joblib.load("random_forest_model.pkl")

# App title
st.title("Cough Presence Prediction")

# Input Features
maternal_age = st.selectbox("Maternal Age", ["15-24 years old", "25-35 years old", ">35 years old"])
region = st.selectbox("Region", ['tigray', 'afar', 'amhara', 'oromia', 'snnpr', 'benishangul', 'somali', 'harari', 'gambela', 'addis ababa', 'dire dawa'])
rural_residence = st.selectbox("Residence", ['rural', 'urban'])
uneducated_mother = st.selectbox("Mothers Education", ['no education', 'primary', 'secondary', 'higher'])
religion = st.selectbox("Religion", ['Orthodox', 'Muslim', 'Protestant', 'Others'])
more_than_5_family_size = st.selectbox("Family Size", ['<=5', '>5'])
male_househead = st.selectbox("Sex of Household Head", ['male', 'female'])
wealth_index = st.selectbox("Wealth Index", ['Poor', 'Middle', 'Rich'])
currently_pregnant = st.selectbox("Currently Pregnant", ['no or unsure', 'yes'])
more_than_3_number_of_children = st.selectbox("Number of Children", ['<4', '>=4'])
desire_more_children = st.selectbox("Desire More Children", ['Wants no more', 'Undecided', 'Wants'])
uneducated_father = st.selectbox("Fathers Education", ['no education', 'primary', 'secondary', 'higher'])
unemployed_father = st.selectbox("Fathers Occupation", ['Not working', 'Working'])
unemployed_mother = st.selectbox("Mothers Occupation", ['Not working', 'Working'])
male_child_sex = st.selectbox("Sex of Child", ['male', 'female'])
child_age = st.selectbox("Child Age", ['6-11', '12-23'])
anc_visit = st.selectbox("ANC Visit", ['No', 'Yes'])
delivered_at_home = st.selectbox("Place of Delivery", ['Home', 'Health facility'])
cesarean_section_delivery = st.selectbox("Mode of Delivery", ['Spontaneous vaginal delivery', 'Cesarean section'])
pnc_visit = st.selectbox("PNC Visit", ['no', 'yes'])
had_diarrhea = st.selectbox("Had Diarrhea", ['no', 'yes'])
media_exposure = st.selectbox("Media Exposure", ['no', 'yes'])
micronutrient_intake_status = st.selectbox("Micronutrient Intake Status", ['no', 'yes'])

# **Feature Mappings for Numerical Encoding**
feature_mappings = {
    "Maternal Age": {"15-24 years old": 0, "25-35 years old": 1, ">35 years old": 2},
    "Residence": {'rural': 1, 'urban': 0},
    "Mothers Education": {'no education': 1, 'primary': 0, 'secondary': 0, 'higher': 0},
    "Family Size": {'<=5': 0, '>5': 1},
    "Sex of Household Head": {'male': 1, 'female': 0},
    "Wealth Index": {'Poor': 0, 'Middle': 1, 'Rich': 2},
    "Currently Pregnant": {'no or unsure': 0, 'yes': 1},
    "Number of Children": {'<4': 0, '>=4': 1},
    "Desire More Children": {'Wants no more': 0, 'Undecided': 1, 'Wants': 2},
    "Fathers Education": {'no education': 1, 'primary': 0, 'secondary': 0, 'higher': 0},
    "Fathers Occupation": {'Not working': 1, 'Working': 0},
    "Mothers Occupation": {'Not working': 1, 'Working': 0},
    "Sex of Child": {'male': 1, 'female': 0},
    "Child Age": {'6-11': 0, '12-23': 1},
    "ANC Visit": {'No': 0, 'Yes': 1},
    "Place of Delivery": {'Home': 1, 'Health facility': 0},
    "Mode of Delivery": {'Spontaneous vaginal delivery': 0, 'Cesarean section': 1},
    "PNC Visit": {'no': 0, 'yes': 1},
    "Had Diarrhea": {'no': 0, 'yes': 1},
    "Media Exposure": {'no': 0, 'yes': 1},
    "Micronutrient Intake Status": {'no': 0, 'yes': 1}
}

# **Convert categorical inputs to numerical values**
#input_data = {feature: feature_mappings[feature][eval(feature.lower().replace(" ", "_"))] for feature in feature_mappings}

# Convert to DataFrame
input_df = pd.DataFrame(feature_mappings)
input_df = input_df.apply(pd.to_numeric, errors='coerce')

# Convert to DataFrame
#input_df = pd.DataFrame([input_data])

# **One-Hot Encoding for "Region" & "Religion"**
input_df["Region"] = region
input_df["Religion"] = religion
input_df = pd.get_dummies(input_df, columns=["Region", "Religion"], drop_first=False)

# **Ensure All Model Features Exist**
expected_features = ML_model.feature_names_in_  # Features expected by the model
missing_cols = [col for col in expected_features if col not in input_df.columns]



# **Predict & Display Results**
if st.button("Submit"):
    prediction = ML_model.predict_proba(input_df)
    st.write(f"### Probability of having cough: {prediction[0][1]:.2%}")
