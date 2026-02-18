import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Diabetes Prediction", layout="wide")

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Prediction App")

st.sidebar.header("Medical Measurements")

preg = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glu = st.sidebar.number_input("Glucose", 0, 200, 120)
bp = st.sidebar.number_input("Blood Pressure", 0, 150, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
ins = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 60.0, 30.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.number_input("Age", 1, 120, 35)

if st.sidebar.button("Predict"):

    input_data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.subheader("Prediction Results")

    if prediction[0] == 1:
        st.error("🔴 HIGH RISK - Diabetic")
    else:
        st.success("🟢 LOW RISK - Not Diabetic")
    diabetic_prob = probability[0][1] * 100

    st.write(f"### Risk Probability: {diabetic_prob:.2f}%")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=diabetic_prob,
        title={'text': "Risk Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(fig)