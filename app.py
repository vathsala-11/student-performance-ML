import streamlit as st
import pickle
import pandas as pd

# Load model and columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("🎓 Student Performance Predictor")

# Inputs
G1 = st.number_input("G1 (0-20)", 0, 20, 10)
G2 = st.number_input("G2 (0-20)", 0, 20, 10)
studytime = st.slider("Study Time (1-4)", 1, 4, 2)
absences = st.number_input("Absences", 0, 100, 5)

if st.button("Predict"):
    
    # Create full input with all columns
    input_dict = {col: 0 for col in columns}
    
    # Fill only required values
    if "G1" in input_dict:
        input_dict["G1"] = G1
    if "G2" in input_dict:
        input_dict["G2"] = G2
    if "studytime" in input_dict:
        input_dict["studytime"] = studytime
    if "absences" in input_dict:
        input_dict["absences"] = absences

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Predict
    prediction = model.predict(input_df)[0]

    # Output
    if prediction == 0:
        st.error("Low Performance")
    elif prediction == 1:
        st.warning("Medium Performance")
    else:
        st.success("High Performance")