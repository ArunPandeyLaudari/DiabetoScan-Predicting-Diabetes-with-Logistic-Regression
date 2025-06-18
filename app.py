import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏥",
    layout="centered"
)

# Load model and scaler with error handling
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'notebook', 'diabetes_model.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'notebook', 'scaler.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model and scaler
model, scaler = load_model_and_scaler()

# Header
st.title("🏥 Diabetes Prediction")
st.write("Enter patient information to assess diabetes risk")

# Check if model is loaded
if model is None or scaler is None:
    st.error("Unable to load the prediction model. Please check model files.")
    st.stop()

# Sidebar information
with st.sidebar:
    st.header("Information")
    st.info("This tool provides diabetes risk assessment based on medical parameters.")
    st.warning("⚠️ This is for screening purposes only. Consult a healthcare professional for proper diagnosis.")

# Input form
st.header("Patient Information")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies", 
        min_value=0, max_value=20, value=0,
        help="Number of pregnancies"
    )
    
    glucose = st.number_input(
        "Glucose (mg/dL)", 
        min_value=0, max_value=300, value=100,
        help="Plasma glucose concentration"
    )
    
    blood_pressure = st.number_input(
        "Blood Pressure (mmHg)", 
        min_value=0, max_value=200, value=70,
        help="Diastolic blood pressure"
    )
    
    skin_thickness = st.number_input(
        "Skin Thickness (mm)", 
        min_value=0, max_value=100, value=20,
        help="Triceps skin fold thickness"
    )

with col2:
    insulin = st.number_input(
        "Insulin (μU/mL)", 
        min_value=0, max_value=500, value=100,
        help="2-hour serum insulin"
    )
    
    bmi = st.number_input(
        "BMI", 
        min_value=0.0, max_value=60.0, value=25.0,
        help="Body Mass Index"
    )
    
    dpf = st.number_input(
        "Diabetes Pedigree Function", 
        min_value=0.0, max_value=3.0, value=0.5, step=0.01,
        help="Genetic diabetes likelihood"
    )
    
    age = st.number_input(
        "Age", 
        min_value=1, max_value=120, value=30,
        help="Age in years"
    )

# Prediction section
st.header("Risk Assessment")

# Create button row
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn1:
    predict_button = st.button("Predict Risk", type="primary", use_container_width=True)

with col_btn2:
    if st.button("Reset", use_container_width=True):
        st.rerun()

with col_btn3:
    show_summary = st.button("Show Summary", use_container_width=True)

# Show prediction results
if predict_button:
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })
    
    try:
        # Scale input and predict
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)
        
        # Display results
        st.subheader("Prediction Result")
        
        if prediction[0] == 1:
            st.error("⚠️ **Higher Risk of Diabetes Detected**")
            st.write(f"Confidence: {prediction_proba[0][1]:.1%}")
            
            st.subheader("Recommendations")
            st.write("• Consult with a healthcare provider")
            st.write("• Consider comprehensive diabetes screening")
            st.write("• Monitor blood glucose levels")
            st.write("• Adopt healthy lifestyle changes")
            
        else:
            st.success("✅ **Lower Risk of Diabetes**")
            st.write(f"Confidence: {prediction_proba[0][0]:.1%}")
            
            st.subheader("Recommendations")
            st.write("• Maintain regular health check-ups")
            st.write("• Continue healthy lifestyle habits")
            st.write("• Monitor weight and exercise regularly")
        
        # Risk factors
        st.subheader("Risk Factor Analysis")
        risk_factors = []
        
        if glucose > 140:
            risk_factors.append("High glucose level")
        if bmi > 30:
            risk_factors.append("High BMI (Obesity)")
        if blood_pressure > 90:
            risk_factors.append("High blood pressure")
        if age > 45:
            risk_factors.append("Age over 45")
        if dpf > 1.0:
            risk_factors.append("High genetic predisposition")
        
        if risk_factors:
            st.write("**Identified Risk Factors:**")
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.write("✅ No major risk factors identified")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Show summary if requested
if show_summary:
    st.subheader("Input Summary")
    
    summary_data = {
        "Parameter": [
            "Age", "BMI", "Glucose", "Blood Pressure", 
            "Pregnancies", "Insulin", "Skin Thickness", "Diabetes Pedigree Function"
        ],
        "Value": [
            f"{age} years", f"{bmi:.1f}", f"{glucose} mg/dL", f"{blood_pressure} mmHg",
            str(pregnancies), f"{insulin} μU/mL", f"{skin_thickness} mm", f"{dpf:.3f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("**Diabetes Prediction App** • Made by Arun Pandey Laudari")