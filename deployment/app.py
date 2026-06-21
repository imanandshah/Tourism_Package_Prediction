"""
Streamlit App for Tourism Package Prediction
"""
import os
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="🏖️",
    layout="wide",
)

# ----------------------------------------------------------------------
# Load model from Hugging Face Model Hub
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="imanandshah/Tourism_Package_Prediction_MODEL",
            filename="best_tourism_model_v1.joblib",
            repo_type="model",
            token=os.getenv("HF_TOKEN"),  # harmless if public, required if private
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(
            f"❌ Could not load the trained model. This is expected if the CI/CD pipeline "
            f"has not yet run to train and upload the model file.\n\nError: {e}"
        )
        st.info(
            "**Next steps:** Push code to GitHub `main` to trigger CI/CD. Once `train.py` "
            "completes, the model will be uploaded and this page will become functional."
        )
        st.stop()

model = load_model()

# ----------------------------------------------------------------------
# Title & description
# ----------------------------------------------------------------------
st.title("🌴 Wellness Tourism Package Prediction App")
st.write("This app predicts whether a customer is likely to purchase the **Wellness Tourism Package**.")
st.write(
    "**For Sales Team Use:** Enter customer details to identify potential buyers "
    "and optimize your outreach strategy."
)
st.markdown("---")

# ----------------------------------------------------------------------
# Customer demographics & travel preferences
# ----------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader(" Customer Demographics")

    Age = st.number_input(
        "Age (years)",
        min_value=18, max_value=100, value=35,
        help="Customer's age in years",
    )
    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"],
        help="Customer's gender",
    )
    CityTier = st.selectbox(
        "City Tier",
        [1, 2, 3],
        help="1: Metro cities, 2: Tier-2 cities, 3: Tier-3 cities",
    )
    Occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"],
        help="Customer's occupation type",
    )
    Designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
        help="Customer's designation in their organization",
    )
    MaritalStatus = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced", "Unmarried"],
        help="Customer's marital status",
    )
    MonthlyIncome = st.number_input(
        "Monthly Income (₹)",
        min_value=1000, max_value=100000, value=22000, step=1000,
        help="Customer's gross monthly income",
    )

with col2:
    st.subheader(" Travel Preferences & Interaction")

    TypeofContact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"],
        help="How the customer initiated contact",
    )
    NumberOfPersonVisiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1, max_value=5, value=3,
        help="Total number of people traveling",
    )
    NumberOfChildrenVisiting = st.number_input(
        "Number of Children Visiting (under 5 years)",
        min_value=0, max_value=3, value=1,
        help="Number of children below age 5",
    )
    PreferredPropertyStar = st.selectbox(
        "Preferred Property Star Rating",
        [3.0, 4.0, 5.0],
        help="Preferred hotel star rating",
    )
    NumberOfTrips = st.number_input(
        "Number of Trips per Year",
        min_value=1, max_value=25, value=3,
        help="Average annual trips taken",
    )
    Passport = st.selectbox(
        "Has Valid Passport?",
        ["Yes", "No"],
        help="Does the customer hold a valid passport?",
    )
    OwnCar = st.selectbox(
        "Owns a Car?",
        ["Yes", "No"],
        help="Does the customer own a car?",
    )

# ----------------------------------------------------------------------
# Sales interaction details
# ----------------------------------------------------------------------
st.markdown("---")
st.subheader(" Sales Interaction Details")

col3, col4 = st.columns(2)

with col3:
    ProductPitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
        help="Type of package pitched to the customer",
    )
    DurationOfPitch = st.number_input(
        "Duration of Pitch (minutes)",
        min_value=5, max_value=130, value=15,
        help="Duration of the sales pitch",
    )

with col4:
    NumberOfFollowups = st.number_input(
        "Number of Follow-ups",
        min_value=1, max_value=6, value=3,
        help="Number of follow-up contacts made",
    )
    PitchSatisfactionScore = st.slider(
        "Pitch Satisfaction Score",
        min_value=1, max_value=5, value=3,
        help="Customer's satisfaction with the pitch (1=Low, 5=High)",
    )

st.markdown("---")

# ----------------------------------------------------------------------
# Prepare input data (column order must match training)
# ----------------------------------------------------------------------
input_data = pd.DataFrame([{
    "Age": Age,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "ProductPitched": ProductPitched,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,
}])

# Classification threshold (tune as per training/validation)
CLASSIFICATION_THRESHOLD = 0.45

# ----------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------
if st.button(" Predict Purchase Likelihood", type="primary"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = int(prediction_proba >= CLASSIFICATION_THRESHOLD)

    st.markdown("---")
    st.subheader(" Prediction Results")

    if prediction == 1:
        st.success(" **HIGH LIKELIHOOD**: This customer is likely to purchase the package!")
        st.metric("Purchase Probability", f"{prediction_proba:.1%}")
        st.info("**Recommendation:** Prioritize this customer for immediate follow-up.")
    else:
        st.warning(" **LOW LIKELIHOOD**: This customer is unlikely to purchase the package.")
        st.metric("Purchase Probability", f"{prediction_proba:.1%}")
        st.info("**Recommendation:** Nurture this lead or focus on higher-priority customers.")

    st.markdown("---")
    st.caption(
        f"Model Confidence: {prediction_proba:.1%} | "
        f"Threshold: {CLASSIFICATION_THRESHOLD:.1%}"
    )

# Footer
st.markdown("---")
st.caption(" Powered by XGBoost ML Pipeline | Visit With Us Tourism")
