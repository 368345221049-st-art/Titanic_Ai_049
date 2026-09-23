import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

FEATURE_NAMES = ["Pclass", "Sex_female", "Age", "Fare", "FamilySize"]

# ----------------------------------------------------------------------------
# ตั้งค่าหน้าเว็บ
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="ระบบทำนายการรอดชีวิต Titanic",
    page_icon="🚢",
    layout="centered",
)

BASE_DIR = Path(__file__).parent
AGE_MAX_DEFAULT = 80.0
FARE_MAX_DEFAULT = 88.0


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(BASE_DIR / "titanic_scaler.joblib")
    metrics_path = BASE_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {"accuracy": None, "age_max": AGE_MAX_DEFAULT, "fare_max": FARE_MAX_DEFAULT}
    return scaler, metrics


scaler, metrics = load_artifacts()
AGE_MAX = metrics.get("age_max", AGE_MAX_DEFAULT)
FARE_MAX = metrics.get("fare_max", FARE_MAX_DEFAULT)

# ----------------------------------------------------------------------------
# สไตล์ (มินิมอล โทนน้ำเงิน-ขาว โทนเดียวกับท้องทะเล)
# ----------------------------------------------------------------------------
st.markdown(
    """
    
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# หัวข้อระบบ
# ----------------------------------------------------------------------------
st.markdown('X_raw = pd.DataFrame(
    [[pclass_enc, sex_female, age_scaled_raw, fare_scaled_raw, family_size]],
    columns=FEATURE_NAMES,
)
X_scaled = scaler.transform(X_raw)

score = (sex_female * 2.5) + ((3 - pclass) * 1.0) + (fare_scaled_raw * 1.2) - (age_scaled_raw * 0.8) - 1.5
proba = 1 / (1 + np.exp(-score))
pred = 1 if proba >= 0.5 else 0

if pred == 1:
    st.markdown(
        f"""""",
    unsafe_allow_html=True,
)""",
        unsafe_allow_html=True,
    )

st.progress(min(max(float(proba), 0.0), 1.0))""",
unsafe_allow_html=True,
