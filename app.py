import streamlit as st
import pickle
import pandas as pd

# Page Config
st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load Model
with open("models/student_model.pkl", "rb") as file:
    model = pickle.load(file)

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
.prediction-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📌 Project Information")

st.sidebar.info("""
### Student Marks Prediction

**Machine Learning Model:** Linear Regression

**Input:** Study Hours

**Output:** Predicted Marks

Built using:
- Python
- Pandas
- Scikit-Learn
- Streamlit
""")

# Main Title
st.title("🎓 Student Marks Predictor")

st.markdown("---")

st.write(
    "Predict student marks based on study hours using a Machine Learning model."
)

# Layout
col1, col2 = st.columns(2)

with col1:
    hours = st.number_input(
        "⏰ Enter Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=1.0,
        step=0.5
    )

with col2:
    st.metric(
        label="Current Study Hours",
        value=hours
    )

st.markdown("")

# Predict Button
if st.button("🚀 Predict Marks"):

    new_data = pd.DataFrame({
        "Hours_Studied": [hours]
    })

    prediction = model.predict(new_data)

    st.success(
        f"🎯 Predicted Marks: {prediction[0]:.2f}"
    )

st.markdown("---")

# Project Statistics
st.subheader("📊 Model Information")

col3, col4 = st.columns(2)

with col3:
    st.metric("MAE", "3.60")

with col4:
    st.metric("R² Score", "0.885")

st.markdown("---")


st.caption("Made with ❤️ using Streamlit and Machine Learning")