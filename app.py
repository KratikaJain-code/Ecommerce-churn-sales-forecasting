import streamlit as st
import pandas as pd
import numpy as np
import joblib
import textwrap
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


# ============================================================
# HTML RENDER FUNCTION
# ============================================================

def render_html(content):
    st.html(textwrap.dedent(content).strip())


# ============================================================
# LOAD MODEL + SCALER
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


try:

    model, scaler = load_model()

    model_loaded = True

except Exception as e:

    model_loaded = False
    model = None
    scaler = None

    st.error("⚠️ Model files could not be loaded.")
    st.code(str(e))


# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>

/* GLOBAL */

.stApp {
    background:
        radial-gradient(circle at 8% 8%,
        rgba(99, 102, 241, 0.14),
        transparent 28%),
        radial-gradient(circle at 92% 12%,
        rgba(20, 184, 166, 0.10),
        transparent 25%),
        #080b12;
}

/* MAIN */

.main .block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* HERO */

.hero {
    padding: 20px 0 40px 0;
}

.hero-badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    border: 1px solid rgba(129,140,248,.35);
    background: rgba(99,102,241,.10);
    color: #a5b4fc;
    font-size: .75rem;
    font-weight: 800;
    letter-spacing: 1.7px;
    margin-bottom: 15px;
}

.hero-title {
    font-size: 3.25rem;
    line-height: 1.05;
    font-weight: 850;
    letter-spacing: -2px;
    margin: 0;
    color: #f8fafc;
}

.hero-title span {
    background: linear-gradient(90deg,#a78bfa,#22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #8b95a7;
    font-size: 1.05rem;
    margin-top: 15px;
    max-width: 720px;
    line-height: 1.7;
}

/* SECTIONS */

.section-title {
    font-size: 1.65rem;
    font-weight: 750;
    color: #f8fafc;
    margin-top: 10px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #7f8796;
    font-size: .92rem;
    margin-bottom: 25px;
}

/* CARDS */

.input-card,
.result-card,
.metric-box,
.model-info {
    background: rgba(17,22,32,.82);
    border: 1px solid rgba(148,163,184,.13);
    border-radius: 20px;
    box-shadow: 0 15px 35px rgba(0,0,0,.12);
}

.input-card {
    padding: 20px;
    margin-bottom: 8px;
}

.input-heading {
    font-size: 1rem;
    font-weight: 750;
    color: #e5e7eb;
}

.input-description {
    font-size: .82rem;
    color: #737d90;
    line-height: 1.5;
}

/* RESULT */

.result-card {
    padding: 28px;
    min-height: 235px;
}

.result-card.high {
    border-color: rgba(244,63,94,.42);
    background:
        linear-gradient(
            145deg,
            rgba(127,29,29,.30),
            rgba(15,20,30,.90)
        );
}

.result-card.medium {
    border-color: rgba(245,158,11,.40);
    background:
        linear-gradient(
            145deg,
            rgba(120,53,15,.26),
            rgba(15,20,30,.90)
        );
}

.result-card.low {
    border-color: rgba(34,197,94,.38);
    background:
        linear-gradient(
            145deg,
            rgba(20,83,45,.26),
            rgba(15,20,30,.90)
        );
}

.result-label {
    color: #7f8aa0;
    font-size: .73rem;
    font-weight: 850;
    letter-spacing: 2px;
}

.probability {
    font-size: 3.25rem;
    font-weight: 850;
    color: #f8fafc;
    margin: 12px 0 18px;
}

.progress-bg {
    width: 100%;
    height: 9px;
    background: #202632;
    border-radius: 999px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg,#6366f1,#ec4899);
}

.risk-title {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 1.7rem;
    font-weight: 800;
    color: #f8fafc;
    margin-top: 22px;
}

.risk-dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
}

/* RECOMMENDATION */

.recommendation {
    margin-top: 22px;
    padding: 23px 26px;
    border-radius: 20px;
    background: rgba(99,102,241,.08);
    border: 1px solid rgba(129,140,248,.20);
}

.recommendation-title {
    color: #c4b5fd;
    font-size: .76rem;
    font-weight: 850;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.recommendation-text {
    color: #d6dae3;
    font-size: .96rem;
    line-height: 1.65;
    margin-top: 8px;
}

/* METRICS */

.metric-box {
    padding: 20px;
    text-align: center;
}

.metric-name {
    color: #7f8796;
    font-size: .70rem;
    font-weight: 850;
    letter-spacing: 1.5px;
}

.metric-value {
    color: #f8fafc;
    font-size: 1.35rem;
    font-weight: 800;
    margin-top: 7px;
}

/* INFO */

.model-info {
    padding: 18px 22px;
    color: #7f8796;
    font-size: .82rem;
    line-height: 1.7;
}

.model-info strong {
    color: #cbd5e1;
}

/* BUTTON */

.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: 1px solid rgba(129,140,248,.45);
    background: linear-gradient(135deg,#6366f1,#7c3aed);
    color: white;
    font-weight: 750;
    font-size: 1rem;
    padding: .75rem 1rem;
}

.stButton > button:hover {
    border-color: #a5b4fc;
    transform: translateY(-1px);
}

</style>
""")

# ============================================================
# HERO SECTION
# ============================================================

render_html(
    """
    <div class="hero">

        <div class="hero-badge">
            AI • CUSTOMER ANALYTICS
        </div>

        <h1 class="hero-title">
            Customer <span>Churn Intelligence</span>
        </h1>

        <div class="hero-subtitle">
            Predict customer churn risk using RFM-based behavioural
            analysis and machine learning.
        </div>

    </div>
    """
)


# ============================================================
# CUSTOMER PROFILE
# ============================================================

render_html(
    """
    <div class="section-title">
        Customer Profile
    </div>

    <div class="section-subtitle">
        Enter the customer's recent purchasing behaviour to generate
        a personalized churn-risk prediction.
    </div>
    """
)


# ============================================================
# INPUT COLUMNS
# ============================================================

left_col, right_col = st.columns(
    2,
    gap="large"
)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_col:

    render_html(
        """
        <div class="input-card">

            <div class="input-heading">
                🕒 Recency
            </div>

            <div class="input-description">
                Number of days since the customer's last purchase.
            </div>

        </div>
        """
    )

    recency = st.number_input(
        "Recency (days)",
        min_value=0,
        value=30,
        step=1
    )


    render_html(
        """
        <div class="input-card">

            <div class="input-heading">
                🛒 Frequency
            </div>

            <div class="input-description">
                Number of orders placed by the customer.
            </div>

        </div>
        """
    )

    frequency = st.number_input(
        "Frequency (number of orders)",
        min_value=1,
        value=4,
        step=1
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with right_col:

    render_html(
        """
        <div class="input-card">

            <div class="input-heading">
                💰 Monetary
            </div>

            <div class="input-description">
                Total amount spent by the customer.
            </div>

        </div>
        """
    )

    monetary = st.number_input(
        "Monetary (total spending)",
        min_value=0.0,
        value=1200.0,
        step=50.0
    )


    render_html(
        """
        <div class="input-card">

            <div class="input-heading">
                🧾 Average Order Value
            </div>

            <div class="input-description">
                Average amount spent per order.
            </div>

        </div>
        """
    )

    average_order_value = st.number_input(
        "Average Order Value",
        min_value=0.0,
        value=300.0,
        step=25.0
    )


# ============================================================
# BUTTON
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_clicked = st.button(
    "✨  Predict Churn Risk",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    if not model_loaded:

        st.error(
            "Model could not be loaded. "
            "Please check the models folder."
        )

    else:

        try:

            # =================================================
            # INPUT DATA
            # =================================================

            input_data = pd.DataFrame(
                [[
                    recency,
                    frequency,
                    monetary,
                    average_order_value
                ]],
                columns=[
                    "Recency",
                    "Frequency",
                    "Monetary",
                    "AverageOrderValue"
                ]
            )


            # =================================================
            # SCALE
            # =================================================

            scaled_data = scaler.transform(
                input_data
            )


            # =================================================
            # CHURN PROBABILITY
            # =================================================

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    scaled_data
                )[0]

                churn_probability = float(
                    probabilities[1]
                )

            else:

                prediction = model.predict(
                    scaled_data
                )[0]

                churn_probability = float(
                    prediction
                )


            # Safety limit

            churn_probability = max(
                0.0,
                min(
                    1.0,
                    churn_probability
                )
            )


            # =================================================
            # RISK CLASSIFICATION
            # =================================================

            if churn_probability >= 0.70:

                risk_category = "High Risk"

                risk_class = "high"

                dot_color = "#fb7185"

                action = (
                    "This customer shows a high likelihood of churn. "
                    "Prioritize targeted retention campaigns, "
                    "personalized offers, loyalty incentives, and "
                    "re-engagement strategies."
                )


            elif churn_probability >= 0.40:

                risk_category = "Medium Risk"

                risk_class = "medium"

                dot_color = "#fbbf24"

                action = (
                    "This customer shows moderate churn risk. "
                    "Consider personalized communication, "
                    "product recommendations, and limited-time "
                    "offers to strengthen engagement."
                )


            else:

                risk_category = "Low Risk"

                risk_class = "low"

                dot_color = "#4ade80"

                action = (
                    "This customer currently appears relatively "
                    "stable. Maintain engagement through loyalty "
                    "programs, personalized recommendations, and "
                    "a consistent customer experience."
                )


            # =================================================
            # PREDICTION SECTION
            # =================================================

            render_html(
                """
                <div class="divider"></div>

                <div class="section-title">
                    Prediction
                </div>

                <div class="section-subtitle">
                    Churn probability generated from the customer's
                    RFM profile.
                </div>
                """
            )


            # =================================================
            # RESULT COLUMNS
            # =================================================

            result_left, result_right = st.columns(
                2,
                gap="large"
            )


            # =================================================
            # PROBABILITY CARD
            # =================================================

            with result_left:

                render_html(
                    f"""
                    <div class="result-card">

                        <div class="result-label">
                            CHURN PROBABILITY
                        </div>

                        <div class="probability">
                            {churn_probability:.2%}
                        </div>

                        <div class="progress-bg">

                            <div
                                class="progress-fill"
                                style="width:{churn_probability * 100:.2f}%;">
                            </div>

                        </div>

                        <div style="
                            margin-top:14px;
                            color:#687081;
                            font-size:0.78rem;
                        ">
                            Model confidence indicator
                        </div>

                    </div>
                    """
                )


            # =================================================
            # RISK CARD
            # =================================================

            with result_right:

                render_html(
                    f"""
                    <div class="result-card {risk_class}">

                        <div class="result-label">
                            RISK STATUS
                        </div>

                        <div class="risk-title">

                            <span
                                class="risk-dot"
                                style="
                                    background:{dot_color};
                                    box-shadow:0 0 14px {dot_color};
                                ">
                            </span>

                            {risk_category}

                        </div>

                        <div style="
                            color:#7f8796;
                            margin-top:18px;
                            font-size:0.85rem;
                            line-height:1.6;
                        ">

                            Recommended retention priority

                        </div>

                    </div>
                    """
                )


            # =================================================
            # RECOMMENDATION
            # =================================================

            render_html(
                f"""
                <div class="recommendation">

                    <div class="recommendation-title">
                        ✦ Recommended Action
                    </div>

                    <div class="recommendation-text">
                        {action}
                    </div>

                </div>
                """
            )


            # =================================================
            # CUSTOMER SNAPSHOT
            # =================================================

            render_html(
                """
                <br>

                <div class="section-title">
                    Customer Behaviour Snapshot
                </div>

                <div class="section-subtitle">
                    Key behavioural indicators used by the
                    prediction model.
                </div>
                """
            )


            # =================================================
            # METRIC CARDS
            # =================================================

            metric1, metric2, metric3, metric4 = st.columns(
                4,
                gap="medium"
            )


            with metric1:

                render_html(
                    f"""
                    <div class="metric-box">

                        <div class="metric-name">
                            RECENCY
                        </div>

                        <div class="metric-value">
                            {recency:.0f} days
                        </div>

                    </div>
                    """
                )


            with metric2:

                render_html(
                    f"""
                    <div class="metric-box">

                        <div class="metric-name">
                            FREQUENCY
                        </div>

                        <div class="metric-value">
                            {frequency:.0f}
                        </div>

                    </div>
                    """
                )


            with metric3:

                render_html(
                    f"""
                    <div class="metric-box">

                        <div class="metric-name">
                            MONETARY
                        </div>

                        <div class="metric-value">
                            ₹{monetary:,.2f}
                        </div>

                    </div>
                    """
                )


            with metric4:

                render_html(
                    f"""
                    <div class="metric-box">

                        <div class="metric-name">
                            AVG ORDER
                        </div>

                        <div class="metric-value">
                            ₹{average_order_value:,.2f}
                        </div>

                    </div>
                    """
                )


            # =================================================
            # RISK INTERPRETATION
            # =================================================

            render_html(
                f"""
                <br>

                <div class="model-info">

                    <strong>Risk interpretation:</strong>

                    &nbsp; Low Risk &lt; 40%
                    &nbsp; • &nbsp;

                    Medium Risk 40–69%
                    &nbsp; • &nbsp;

                    High Risk ≥ 70%

                    <br>

                    <strong>Current prediction:</strong>

                    {churn_probability:.2%}

                    → {risk_category}

                </div>
                """
            )


            # =================================================
            # MODEL INFORMATION
            # =================================================

            render_html(
                """
                <br>

                <div class="model-info">

                    <strong>Model:</strong>
                    Machine Learning Churn Classifier

                    &nbsp;&nbsp;•&nbsp;&nbsp;

                    <strong>Features:</strong>
                    Recency, Frequency, Monetary,
                    Average Order Value

                    &nbsp;&nbsp;•&nbsp;&nbsp;

                    <strong>Prediction:</strong>
                    Customer churn probability

                </div>
                """
            )


        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.write(
                "Please check whether the model and scaler "
                "were trained using the same feature order."
            )

            st.code(
                str(e)
            )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <br><br>

    <div style="
        text-align:center;
        color:#525b6b;
        font-size:0.75rem;
        padding-top:25px;
    ">

        Customer Churn Intelligence
        •
        RFM-Based Machine Learning Analytics

    </div>
    """
)