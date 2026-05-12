# ============================================================
# SMART GRID STABILITY PREDICTION APP
# ============================================================
# This Streamlit app loads a trained LightGBM model bundle and
# predicts whether a smart-grid configuration is stable or unstable.
#
# Required file in the same folder:
#   smart_grid_lightgbm_bundle.pkl
#
# The bundle must contain:
#   model
#   scaler
#   feature_names
# ============================================================

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Grid Stability Prediction",
    page_icon="⚡",
    layout="centered"
)


# ============================================================
# LOAD MODEL BUNDLE
# ============================================================

BUNDLE_PATH = Path("smart_grid_lightgbm_bundle.pkl")


@st.cache_resource
def load_model_bundle():
    """
    Loads the saved LightGBM model bundle.

    The bundle should contain:
    - model: trained LightGBM model
    - scaler: fitted StandardScaler
    - feature_names: list of six engineered features
    """
    if not BUNDLE_PATH.exists():
        return None

    return joblib.load(BUNDLE_PATH)


bundle = load_model_bundle()

if bundle is None:
    st.error(
        "Model bundle file not found. Please make sure "
        "`smart_grid_lightgbm_bundle.pkl` is in the same folder as `app.py`."
    )
    st.stop()


model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]


# ============================================================
# APP HEADER
# ============================================================

st.title("Smart Grid Stability Prediction System")

st.write(
    "This application predicts whether a decentralised smart-grid configuration "
    "is stable or unstable using a trained LightGBM machine-learning model."
)

st.info(
    "The model uses six engineered features derived from reaction-time and "
    "price-elasticity inputs."
)


# ============================================================
# USER INPUTS
# ============================================================

st.subheader("Reaction-Time Inputs")

st.caption(
    "Reaction time represents how quickly each participant responds to changes "
    "in grid conditions."
)

col1, col2 = st.columns(2)

with col1:
    tau1 = st.number_input(
        "tau1 — Producer reaction time",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

    tau2 = st.number_input(
        "tau2 — Consumer 1 reaction time",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

with col2:
    tau3 = st.number_input(
        "tau3 — Consumer 2 reaction time",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

    tau4 = st.number_input(
        "tau4 — Consumer 3 reaction time",
        min_value=0.0,
        value=5.0,
        step=0.1
    )


st.subheader("Price-Elasticity Inputs")

st.caption(
    "Price elasticity represents how strongly each participant responds to "
    "price-frequency signals."
)

col3, col4 = st.columns(2)

with col3:
    g1 = st.number_input(
        "g1 — Producer price elasticity",
        min_value=0.0,
        value=0.5,
        step=0.01
    )

    g2 = st.number_input(
        "g2 — Consumer 1 price elasticity",
        min_value=0.0,
        value=0.5,
        step=0.01
    )

with col4:
    g3 = st.number_input(
        "g3 — Consumer 2 price elasticity",
        min_value=0.0,
        value=0.5,
        step=0.01
    )

    g4 = st.number_input(
        "g4 — Consumer 3 price elasticity",
        min_value=0.0,
        value=0.5,
        step=0.01
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_engineered_features(tau1, tau2, tau3, tau4, g1, g2, g3, g4):
    """
    Converts raw user inputs into the six engineered features used by the model.

    Engineered features:
    - tau_mean
    - tau_max
    - tau_range
    - g_mean
    - g_range
    - tau_g_interaction
    """

    tau_values = [tau1, tau2, tau3, tau4]
    g_values = [g1, g2, g3, g4]

    tau_mean = sum(tau_values) / 4
    tau_max = max(tau_values)
    tau_range = max(tau_values) - min(tau_values)

    g_mean = sum(g_values) / 4
    g_range = max(g_values) - min(g_values)

    tau_g_interaction = tau_mean * g_mean

    input_data = pd.DataFrame([{
        "tau_mean": tau_mean,
        "tau_max": tau_max,
        "tau_range": tau_range,
        "g_mean": g_mean,
        "g_range": g_range,
        "tau_g_interaction": tau_g_interaction
    }])

    return input_data


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict Stability", type="primary"):

    # Create engineered features from user inputs.
    input_df = build_engineered_features(
        tau1=tau1,
        tau2=tau2,
        tau3=tau3,
        tau4=tau4,
        g1=g1,
        g2=g2,
        g3=g3,
        g4=g4
    )

    # Ensure the feature order matches the training order.
    input_df = input_df[feature_names]

    # Scale input using the scaler fitted during training.
    input_scaled = pd.DataFrame(
        scaler.transform(input_df),
        columns=feature_names
    )

    # Make prediction.
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")

    prediction_text = str(prediction).lower()

    if prediction_text == "stable":
        st.success("Predicted grid state: Stable")
    elif prediction_text == "unstable":
        st.error("Predicted grid state: Unstable")
    else:
        st.info(f"Predicted class: {prediction}")

    # Show engineered features used for transparency.
    with st.expander("View engineered features used by the model"):
        st.dataframe(input_df, use_container_width=True)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Final model: LightGBM trained on six engineered features — "
    "tau_mean, tau_max, tau_range, g_mean, g_range, and tau_g_interaction."
)
