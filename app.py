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
    model = joblib.load(BASE_DIR / "titanic_scaler.joblib")
    metrics_path = BASE_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {"accuracy": None, "age_max": AGE_MAX_DEFAULT, "fare_max": FARE_MAX_DEFAULT}
    return scaler, model, metrics


scaler, model, metrics = load_artifacts()
AGE_MAX = metrics.get("age_max", AGE_MAX_DEFAULT)
FARE_MAX = metrics.get("fare_max", FARE_MAX_DEFAULT)

# ----------------------------------------------------------------------------
# สไตล์ (มินิมอล โทนน้ำเงิน-ขาว โทนเดียวกับท้องทะเล)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F7F9FB;
    }
    .main-title {
        text-align: center;
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F3057;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1rem;
        color: #5A7184;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 1.6rem 1.6rem 1.2rem 1.6rem;
        box-shadow: 0 2px 14px rgba(15, 48, 87, 0.07);
        margin-bottom: 1.2rem;
        border: 1px solid #EAF0F6;
    }
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F3057;
        margin-bottom: 0.6rem;
    }
    .result-survive {
        background: linear-gradient(135deg, #E4F7EE, #F3FFF9);
        border: 1px solid #B7E4C7;
        border-radius: 16px;
        padding: 1.4rem;
        text-align: center;
    }
    .result-notsurvive {
        background: linear-gradient(135deg, #FDECEC, #FFF6F6);
        border: 1px solid #F3B9B9;
        border-radius: 16px;
        padding: 1.4rem;
        text-align: center;
    }
    .result-emoji {
        font-size: 2.6rem;
    }
    .result-text {
        font-size: 1.4rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }
    .footer {
        text-align: center;
        color: #94A3B3;
        font-size: 0.85rem;
        margin-top: 2.2rem;
        padding-top: 1rem;
        border-top: 1px solid #E6EBF0;
    }
    div[data-testid="stMetricValue"] {
        color: #0F3057;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# หัวข้อระบบ
# ----------------------------------------------------------------------------
st.markdown('<div class="main-title">🚢 ระบบทำนายการรอดชีวิตผู้โดยสารเรือไททานิก</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">กรอกข้อมูลผู้โดยสารเพื่อทำนายโอกาสรอดชีวิตด้วย Machine Learning</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# แถบแสดงความแม่นยำของระบบ (เพิ่มความเชื่อมั่น)
# ----------------------------------------------------------------------------
acc_col1, acc_col2, acc_col3 = st.columns(3)
if metrics.get("accuracy") is not None:
    acc_col1.metric("🎯 ความแม่นยำของระบบ", f"{metrics['accuracy']*100:.1f}%")
if metrics.get("precision") is not None:
    acc_col2.metric("✅ Precision", f"{metrics['precision']*100:.1f}%")
if metrics.get("f1") is not None:
    acc_col3.metric("📊 F1-score", f"{metrics['f1']*100:.1f}%")

st.write("")

# ----------------------------------------------------------------------------
# ฟอร์มกรอกข้อมูล
# ----------------------------------------------------------------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎟️ ข้อมูลผู้โดยสาร</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pclass_label = st.selectbox(
            "🛳️ ชั้นโดยสาร",
            ["ชั้น 1 (Pclass 1)", "ชั้น 2 (Pclass 2)", "ชั้น 3 (Pclass 3)"],
            index=2,
        )
        pclass = int(pclass_label.split("Pclass ")[1].rstrip(")"))

        sex_label = st.radio("👤 เพศ", ["หญิง", "ชาย"], horizontal=True)
        sex_female = 1 if sex_label == "หญิง" else 0

    with col2:
        age = st.slider("🎂 อายุ (ปี)", min_value=0, max_value=80, value=28)
        fare = st.slider("💰 ค่าโดยสาร (Fare, ปอนด์)", min_value=0.0, max_value=300.0, value=32.0, step=0.5)

    fam_col1, fam_col2 = st.columns(2)
    with fam_col1:
        sibsp = st.number_input("👫 จำนวนพี่น้อง/คู่สมรสที่ร่วมเดินทาง (SibSp)", min_value=0, max_value=10, value=0)
    with fam_col2:
        parch = st.number_input("👨‍👩‍👧 จำนวนพ่อแม่/ลูกที่ร่วมเดินทาง (Parch)", min_value=0, max_value=10, value=0)

    family_size = sibsp + parch + 1
    st.caption(f"👨‍👩‍👧‍👦 ขนาดครอบครัวทั้งหมด (FamilySize) = {family_size} คน")

    st.markdown("</div>", unsafe_allow_html=True)

predict_clicked = st.button("🔮 ทำนายผล", use_container_width=True, type="primary")

# ----------------------------------------------------------------------------
# ทำนายผล
# ----------------------------------------------------------------------------
if predict_clicked:
    pclass_enc = pclass - 1
    age_scaled_raw = age / AGE_MAX
    fare_scaled_raw = fare / FARE_MAX

    X_raw = pd.DataFrame(
        [[pclass_enc, sex_female, age_scaled_raw, fare_scaled_raw, family_size]],
        columns=FEATURE_NAMES,
    )
    X_scaled = scaler.transform(X_raw)

    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]

    if pred == 1:
        st.markdown(
            f"""
            <div class="result-survive">
                <div class="result-emoji">🟢🛟</div>
                <div class="result-text" style="color:#1E7A46;">รอดชีวิต</div>
                <div style="color:#3B7A57; margin-top:0.3rem;">ความน่าจะเป็นในการรอดชีวิต {proba*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-notsurvive">
                <div class="result-emoji">⚓️💔</div>
                <div class="result-text" style="color:#B33A3A;">ไม่รอดชีวิต</div>
                <div style="color:#B36A6A; margin-top:0.3rem;">ความน่าจะเป็นในการรอดชีวิต {proba*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(min(max(proba, 0.0), 1.0))

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        พัฒนาโดย นางสาวชนิดา แก้วเพชร<br>
        ⚓ Titanic Survival Prediction System
    </div>
    """,
    unsafe_allow_html=True,
)
