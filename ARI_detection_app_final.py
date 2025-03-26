# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:00:59 2025

@author: assay
"""

# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import pandas as pd

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# Load the trained model
ML_model = joblib.load("random_forest_model.pkl",mmap_mode=None)
expected_columns = joblib.load("column_order.pkl")

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
desire_morechildren = st.selectbox("Desire More Children", ['Wants no more', 'Undecided', 'Wants'])
uneducated_father = st.selectbox("Fathers Education", ['no education', 'primary', 'secondary', 'higher'])
unemployed_father = st.selectbox("Fathers Occupation", ['Not working', 'Working'])
unemployed_mother = st.selectbox("Mothers Occupation", ['Not working', 'Working'])
male_child_sex = st.selectbox("Sex of Child", ['male', 'female'])
chid_age = st.selectbox("Child Age", ['6-11', '12-23'])
ANC_visit = st.selectbox("ANC Visit", ['No', 'Yes'])
delivered_at_home = st.selectbox("Place of Delivery", ['Home', 'Health facility'])
cesarean_section_delivery = st.selectbox("Mode of Delivery", ['Spontaneous vaginal delivery', 'Cesarean section'])
PNC_visit = st.selectbox("PNC Visit", ['no', 'yes'])
Had_diarrhea = st.selectbox("Had Diarrhea", ['no', 'yes'])
media_exposure = st.selectbox("Media Exposure", ['no', 'yes'])
Micronutrient_intakestatus = st.selectbox("Micronutrient Intake Status", ['no', 'yes'])

# **Feature Mappings for Numerical Encoding**
feature_mappings = {
    "maternal_age": {"15-24 years old": 0, "25-35 years old": 1, ">35 years old": 2},
    "rural_residence": {'rural': 1, 'urban': 0},
    "uneducated_mother": {'no education': 1, 'primary': 0, 'secondary': 0, 'higher': 0},
    "more_than_5_family_size": {'<=5': 0, '>5': 1},
    "male_househead": {'male': 1, 'female': 0},
    "wealth_index": {'Poor': 0, 'Middle': 1, 'Rich': 2},
    "currently_pregnant": {'no or unsure': 0, 'yes': 1},
    "more_than_3_number_of_children": {'<4': 0, '>=4': 1},
    "desire_morechildren": {'Wants no more': 0, 'Undecided': 1, 'Wants': 2},
    "uneducated_father": {'no education': 1, 'primary': 0, 'secondary': 0, 'higher': 0},
    "unemployed_father": {'Not working': 1, 'Working': 0},
    "unemployed_mother": {'Not working': 1, 'Working': 0},
    "male_child_sex": {'male': 1, 'female': 0},
    "chid_age": {'6-11': 0, '12-23': 1},
    "ANC_visit": {'No': 0, 'Yes': 1},
    "delivered_at_home": {'Home': 1, 'Health facility': 0},
    "cesarean_section_delivery": {'Spontaneous vaginal delivery': 0, 'Cesarean section': 1},
    "PNC_visit": {'no': 0, 'yes': 1},
    "Had_diarrhea": {'no': 0, 'yes': 1},
    "media_exposure": {'no': 0, 'yes': 1},
    "Micronutrient_intakestatus": {'no': 0, 'yes': 1}
}

# **Convert categorical inputs to numerical values**
#input_data = {feature: feature_mappings[feature][eval(feature.lower().replace(" ", "_"))] for feature in feature_mappings}

# Convert to DataFrame
input_df = pd.DataFrame(feature_mappings)
input_df = input_df.apply(pd.to_numeric, errors='coerce')

# Convert to DataFrame
#input_df = pd.DataFrame([input_data])

# **One-Hot Encoding for "Region" & "Religion"**
input_df["region"] = region
input_df["religion"] = religion
input_df = pd.get_dummies(input_df, columns=["region", "religion"], drop_first=False)
input_df.columns = input_df.columns.str.lower().str.replace(' ','_')


# Reorder columns to match training data
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns with 0

input_df = input_df[expected_columns]  # Reorder columns

# Assigning o if there are any missing features 
missing_features = set(ML_model.feature_names_in_) - set(input_df.columns)
print(missing_features)
for feature in missing_features:
    input_df[feature] = 0  # or some default value
    
# for consistency convert all the data elements in to float
input_df = input_df.astype(float)

# **Predict & Display Results**
if st.button("Submit"):
    prediction = ML_model.predict_proba(input_df)
    st.write(f"### Probability of having cough: {prediction[0][1]:.2%}")



#print("Model expected features:", ML_model.feature_names_in_)
#print("Input dataframe features:", input_df.columns)
#print("Input dataframe types:\n", input_df.dtypes)