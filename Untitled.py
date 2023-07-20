import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import joblib


# Load the trained model
with open('best_model.pkl', 'rb') as f:
    best_model = joblib.load(f)
    
# Import columns_stats.xlsx file and get data from the name, min, and max columns
columns_stats = pd.read_excel('columns_stats.xlsx')



st.write("# Prediction information")
st.write(" Please manually enter the information for this member according to the data dictionary, or leave it in the initial position if no information is available.")
st.write(" ")
# Convert specified columns to selectbox or slider widgets
selected_columns = []
for index, row in columns_stats.iterrows():
    column_name = row['name']
    if index == 1 or (3 <= index <= 21) or index == 29:
        # For the 2nd row, 4th to 22nd rows, and 30th row, use st.selectbox widget with options [0, 1]
        selected_value = st.selectbox(f"Select an option for {column_name}:", [0, 1])
    elif index == 22:
        # For the 23rd column, use st.selectbox widget with options [1, 2, 3, 4, 5]
        selected_value = st.selectbox(f"Select an option for {column_name}:", [1, 2, 3, 4, 5])
    elif index == 25:
        # For the 26th column, use st.selectbox widget with options [1, 2, 3, 4]
        selected_value = st.selectbox(f"Select an option for {column_name}:", [1, 2, 3, 4])
    elif index == 26:
        # For the 27th column, use st.selectbox widget with options [1, 2, ..., 26]
        selected_value = st.selectbox(f"Select an option for {column_name}:", list(range(1, 27)))
    elif index == 27:
        # For the 28th column, use st.selectbox widget with options [1, 2, ..., 15]
        selected_value = st.selectbox(f"Select an option for {column_name}:", list(range(1, 16)))
    elif index == 30:
        # For the 31st column, use st.selectbox widget with options [1, 2, 3, 4, 5]
        selected_value = st.selectbox(f"Select an option for {column_name}:", [1, 2, 3, 4, 5])
    else:
        # For other columns, use st.slider widget
        min_value = float(row['min'])
        max_value = float(row['max'])
        selected_value = st.slider(f"{column_name}", min_value, max_value, step=0.1)

    selected_columns.append((column_name, selected_value))

# Convert the user input to a DataFrame
test_data = pd.DataFrame(columns=[col[0] for col in selected_columns])
test_data.loc[0] = [col[1] for col in selected_columns]


# Use the trained model for prediction
prediction = best_model.predict(test_data)

# Display the prediction result with larger font size
st.write("# Prediction Result")

# Set default values for color and probability_message
color = "black"
probability_message = "Error: Unable to determine the prediction."

# Determine the color and probability message based on the prediction
if prediction[0] == 0:
    color = "blue"
    probability_message = "More attraction needed before considering membership."
elif prediction[0] == 1:
    color = "red"
    probability_message = "Highly likely (greater than 50%) to purchase a membership."

# Apply the custom styling using Markdown
st.markdown(f"<p style='font-size: 20px; color: {color};'> Prediction result: {prediction[0]}</p>", unsafe_allow_html=True)
st.markdown(f"<p style='font-size: 20px; color: {color};'>{probability_message}</p>", unsafe_allow_html=True)
