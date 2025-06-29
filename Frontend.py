import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_model():
    with open("trained_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model
st.set_page_config(
    page_title="Medical Insurance Premium Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("💡 AI-Driven Medical Insurance Premium Prediction")
st.write("""
    Welcome to the **Medical Insurance Premium Predictor**!  
    Fill in your details below.
    """)
st.sidebar.header("Input Your Details")
age = st.sidebar.slider("Age", min_value=18, max_value=68, value=25, step=1)
gender = st.sidebar.radio("Gender", options=["Male", "Female"])
bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0, step=0.1, format="%.1f")
children = st.sidebar.slider("Number of Children", min_value=0, max_value=5, value=0, step=1)
smoker = st.sidebar.checkbox("Are you a smoker?")
region = st.sidebar.radio(
    "Region", 
    options=["Southwest", "Northwest", "Southeast", "Northeast"], 
    index=0
)
medical_history = st.sidebar.selectbox(
    "Medical History", 
    options=["None", "High blood pressure", "Heart disease", "Diabetes"],
    index=1
)
family_history = st.sidebar.selectbox(
    "Family Medical History", 
    options=["None", "High blood pressure", "Heart disease", "Diabetes"],
    index=1 
)
exercise_frequency = st.sidebar.selectbox(
    "Exercise Frequency", 
    options=["Never", "Rarely", "Occasionally", "Frequently"], 
    index=2
)
occupation = st.sidebar.selectbox(
    "Occupation", 
    options=["Blue collar", "White collar", "Student", "Unemployed"], 
    index=1
)
coverage_level = st.sidebar.radio(
    "Coverage Level", 
    options=["Basic", "Standard", "Premium"], 
    index=1
)

st.markdown("---")

if st.button("Predict Premium 💵"):
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender.lower()],  
        "bmi": [bmi],
        "children": [children],
        "smoker": ["yes" if smoker else "no"],
        "region": [region.lower()],
        "medical_history": [medical_history],
        "family_medical_history": [family_history],
        "exercise_frequency": [exercise_frequency],
        "occupation": [occupation],
        "coverage_level": [coverage_level],
    })
    try:
        model = load_model()
        predicted_charge = model.predict(input_data)[0]
        st.success(f"💵 Predicted Medical Insurance Premium: ${predicted_charge:,.2f}")
    except FileNotFoundError:
        st.error("❌ Prediction model not found! Please upload the model to make predictions.")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
st.markdown("---")
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Prediction Summary")
    st.write("""
        Fill in your details and click the button below to predict your medical insurance premium.
    """)
with col2:
    st.subheader("Quick Details")
    st.write(f"**Age:** {age} years")
    st.write(f"Sex:{gender}")
    st.write(f"**BMI:** {bmi}")
    st.write(f"**Number of Children:** {children}")
    st.write(f"**Smoker:** {'Yes' if smoker else 'No'}")
    st.write(f"**Region:** {region}")
    st.write(f"**Medical History:** {(medical_history)}")
    st.write(f"**Family History:** {(family_history)}")
    st.write(f"**Exercise Frequency:** {exercise_frequency}")
    st.write(f"**Occupation:** {occupation}")
    st.write(f"**Coverage Level:** {coverage_level}")
st.markdown("---")
