
import streamlit as st
import pandas as pd
import joblib

# Load trained model
artifact = joblib.load("predictive_maintenance_model.joblib")

model = artifact["model"]
threshold = artifact["threshold"]
feature_columns = artifact["features"]

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Intelligence",
    page_icon="⚙️",
    layout="wide"
)

# Application title
st.title("⚙️ Predictive Maintenance Intelligence")

st.markdown(
    """
    Predict machine failure risk using a trained LightGBM model.

    Enter the machine's operating parameters to estimate its
    failure probability.
    """
)

st.info(
    "This application uses the UCI AI4I 2020 synthetic dataset. "
    "Predictions require validation before real industrial use."
)

# Sidebar
st.sidebar.header("About the Model")
st.sidebar.write("**Algorithm:** LightGBM")
st.sidebar.write(f"**Alert threshold:** {threshold:.0%}")
st.sidebar.write("**Dataset:** UCI AI4I 2020")

# Input section
st.header("Enter Machine Parameters")

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        options=["L", "M", "H"],
        index=0
    )

    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=250.0,
        max_value=350.0,
        value=299.2,
        step=0.1
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=250.0,
        max_value=400.0,
        value=310.3,
        step=0.1
    )

with col2:
    rotational_speed = st.number_input(
        "Rotational Speed (rpm)",
        min_value=0,
        max_value=10000,
        value=1352,
        step=1
    )

    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=200.0,
        value=48.9,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        min_value=0,
        max_value=500,
        value=238,
        step=1
    )

# Prediction
if st.button("Predict Failure Risk", type="primary"):

    input_data = pd.DataFrame(
        [[
            machine_type,
            air_temp,
            process_temp,
            rotational_speed,
            torque,
            tool_wear
        ]],
        columns=feature_columns
    )

    probability = model.predict_proba(input_data)[0][1]

    st.header("Prediction Results")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            "Estimated Failure Probability",
            f"{probability:.2%}"
        )

    with metric_col2:
        st.metric(
            "Alert Threshold",
            f"{threshold:.0%}"
        )

    st.progress(float(probability))

    if probability >= threshold:
        st.error(
            "🚨 FAILURE ALERT: The model predicts elevated "
            "machine failure risk."
        )

        st.warning(
            "Consider inspecting the machine and reviewing its "
            "operating conditions. This prediction is not a diagnosis."
        )

    else:
        st.success(
            "✅ NORMAL STATUS: The estimated failure probability "
            "is below the selected alert threshold."
        )

    st.caption(
        "This result is a model-generated estimate, not a guarantee "
        "of future machine behavior."
    )
