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

# Add custom CSS for centering the title, submit button, and changing the button style
st.markdown("""
    <style>
        /* Custom Title Color */
        h1 {
            text-align: center;
            color: #FF6347;  /* Tomato Red */
        }
        /* Center the submit button */
        div.stButton {
            display: flex;
            justify-content: center;
        }
        /* Make the submit button bold, increase its font size, and change color */
        .css-1emrehy.edgvbvh3 {
            font-weight: bold;
            font-size: 20px;
            background-color: #4CAF50;  /* Green Background */
            color: white;  /* White text */
            border-radius: 5px;  /* Rounded corners */
        }
        /* Hover effect for the submit button */
        .css-1emrehy.edgvbvh3:hover {
            background-color: #45a049;  /* Darker green */
        }
        /* Center the tabs */
        div.stTabs {
            display: flex;
            justify-content: center;
        }
        div.stTabs > div {
            flex-grow: 0;
        }
        /* Add margin after the title */
        .title-margin {
            margin-top: 30px;
        }
        /* Add space before the submit button */
        .submit-margin {
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# Title for the app
# st.title("Welcome, let's predict the cough presence")
st.markdown("<div style='font-size:20px; font-weight:bold; text-align:center;'>Welcome, let's predict the cough presence</div>", unsafe_allow_html=True)
# Organize content in tabs
tab1, tab2, tab3, tab4 = st.tabs(["Feature Set 1", "Feature Set 2", "Feature Set 3", 
                                  "Feature Set 4"])


# Collapsible sections for input features
with tab1:
    st.write("### Feature Set 1")
    col1, col2, col3 = st.columns(3)
    with col1:
        maternal_age = st.selectbox("Maternal Age", ["15-24 years old", "25-35 years old", ">35 years old"])
        region = st.selectbox("Region", ['tigray', 'afar', 'amhara', 'oromia', 'snnpr', 'benishangul', 
                                         'somali', 'harari', 'gambela', 'addis ababa', 'dire dawa'])
    with col2:
        rural_residence = st.selectbox("Residence", ['rural', 'urban'])
        uneducated_mother = st.selectbox("Mothers Education", ['no education','primary','secondary','higher'])
    with col3:
        religion = st.selectbox("Religion", ['Orthodox', 'Muslim', 'Protestant', 'Others'])
        more_than_5_family_size = st.selectbox("Family Size", ['<=5', '>5'])

with tab2:
    st.write("### Feature Set 2")
    col1, col2, col3 = st.columns(3)
    with col1:
        male_househead = st.selectbox("Sex of Household Head", ['male', 'female'])
        wealth_index = st.selectbox("Wealth Index", ['Poor', 'Middle', 'Rich'])
    with col2:
        currently_pregnant = st.selectbox("Currently Pregnant", ['no or unsure', 'yes'])
        more_than_3_number_of_children = st.selectbox("Number of Children", ['<4', '>=4'])
    with col3:
        desire_morechildren = st.selectbox("Desire More Children", ['Wants no more', 'Undecided', 'Wants'])
        uneducated_father = st.selectbox("Fathers Education", ['no education', 'primary', 'secondary', 'higher'])

with tab3:
    st.write("### Feature Set 3")
    col1, col2, col3 = st.columns(3)
    with col1:
        unemployed_father = st.selectbox("Fathers Occupation", ['Not working', 'Working'])
        unemployed_mother = st.selectbox("Mothers Occupation", ['Not working', 'Working'])
    with col2:
        male_child_sex = st.selectbox("Sex of Child", ['male', 'female'])
        chid_age = st.selectbox("Child Age", ['6-11', '12-23'])
    with col3:
        ANC_visit = st.selectbox("ANC Visit", ['No', 'Yes'])
        delivered_at_home = st.selectbox("Place of Delivery", ['Home', 'Health facility'])

with tab4:
    st.write("### Feature Set 4")
    col1, col2, col3 = st.columns(3)
    with col1:
        cesarean_section_delivery = st.selectbox("Mode of Delivery", ['Spontaneous vaginal delivery', 'Cesarean section'])
        PNC_visit = st.selectbox("PNC Visit", ['no', 'yes'])
    with col2:
        Had_diarrhea = st.selectbox("Had Diarrhea", ['no', 'yes'])
        media_exposure = st.selectbox("Media Exposure", ['no', 'yes'])
    with col3:
        Micronutrient_intakestatus = st.selectbox("Micronutrient Intake Status", ['no', 'yes'])

# Load the trained model
ML_model = joblib.load("random_forest_model.pkl",mmap_mode=None)
expected_columns = joblib.load("column_order.pkl")


# Input Features
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
   st.markdown(
    f"<div style='font-size:16px; text-align:center; font-weight:500;'>Probability of having cough: {prob:.2%}</div>",
    unsafe_allow_html=True
)



#print("Model expected features:", ML_model.feature_names_in_)
#print("Input dataframe features:", input_df.columns)
#print("Input dataframe types:\n", input_df.dtypes)
