import streamlit as st
import pickle
import pandas as pd

st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #1f3c88;
    margin-bottom: 5px;
    animation: fadeInDown 1s ease-in-out;
}

.sub-text {
    font-size: 18px;
    color: #4b5563;
    margin-bottom: 20px;
    animation: fadeIn 1.5s ease-in-out;
}

/* Cards */
.custom-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
    animation: fadeInUp 0.8s ease-in-out;
}

.result-card {
    padding: 18px;
    border-radius: 16px;
    font-size: 20px;
    font-weight: 700;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.10);
    animation: popIn 0.5s ease-in-out;
}

.success-card {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    color: #065f46;
}

.error-card {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    color: #991b1b;
}

.prob-card {
    background: white;
    padding: 15px;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    border-left: 6px solid #2563eb;
    margin-top: 12px;
    animation: fadeInUp 1s ease-in-out;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 14px;
    border: none;
    padding: 14px 20px;
    font-size: 18px;
    font-weight: 700;
    transition: all 0.3s ease;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 12px 24px rgba(37, 99, 235, 0.35);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a8a, #2563eb);
}

/* Sidebar headings and labels */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: white !important;
}

/* Sidebar input box text */
section[data-testid="stSidebar"] input {
    color: black !important;
    background-color: white !important;
}

/* Sidebar input container */
section[data-testid="stSidebar"] div[data-baseweb="input"] {
    background-color: white !important;
    border-radius: 12px !important;
}

/* Sidebar plus/minus buttons */
section[data-testid="stSidebar"] button {
    color: black !important;
}

/* Animations */
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

@keyframes fadeInDown {
    from {opacity: 0; transform: translateY(-20px);}
    to {opacity: 1; transform: translateY(0);}
}

@keyframes fadeInUp {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

@keyframes popIn {
    from {opacity: 0; transform: scale(0.95);}
    to {opacity: 1; transform: scale(1);}
}
</style>
""", unsafe_allow_html=True)

# ---------- Load model and data ----------
with open("models/best_model.pkl", "rb") as file:
    model = pickle.load(file)

X_test = pd.read_csv("data/X_test.csv")

# ---------- Header ----------
st.markdown('<div class="main-title">🏨 Hotel Booking Cancellation Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">An interactive machine learning application that predicts whether a hotel booking is likely to be <b>Canceled</b> or <b>Not Canceled</b>.</div>',
    unsafe_allow_html=True
)

st.success("Model and test data loaded successfully.")

# ---------- Sidebar ----------
st.sidebar.markdown("## ⚙️ Prediction Settings")
sample_index = st.sidebar.number_input(
    "Select row number",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    step=1
)

sample_input = X_test.iloc[[sample_index]]

# ---------- Top cards ----------
colA, colB, colC = st.columns(3)

with colA:
    st.markdown(
        f'<div class="custom-card"><h4>📌 Selected Row</h4><h2>{sample_index}</h2></div>',
        unsafe_allow_html=True
    )

with colB:
    st.markdown(
        f'<div class="custom-card"><h4>📊 Total Test Rows</h4><h2>{len(X_test)}</h2></div>',
        unsafe_allow_html=True
    )

with colC:
    st.markdown(
        '<div class="custom-card"><h4>🧠 Model</h4><h2>Random Forest</h2></div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------- Main content ----------
left_col, right_col = st.columns([2.2, 1])

with left_col:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📄 Selected Booking Data")
    st.dataframe(sample_input, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📍 Record Summary")
    st.write(f"*Row Number:* {sample_index}")
    st.write(f"*Number of Features:* {sample_input.shape[1]}")
    st.write("*Status:* Ready for prediction")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- Prediction ----------
if st.button("🚀 Predict Booking Status"):
    prediction = model.predict(sample_input)[0]
    probability = model.predict_proba(sample_input)[0]

    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.markdown(
            '<div class="result-card error-card">❌ The booking is likely to be Canceled</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-card success-card">✅ The booking is likely to be Not Canceled</div>',
            unsafe_allow_html=True
        )

    prob_col1, prob_col2 = st.columns(2)

    with prob_col1:
        st.markdown(
            f'<div class="prob-card"><h4>Probability of Not Canceled</h4><h2>{probability[0]*100:.1f}%</h2></div>',
            unsafe_allow_html=True
        )

    with prob_col2:
        st.markdown(
            f'<div class="prob-card"><h4>Probability of Canceled</h4><h2>{probability[1]*100:.1f}%</h2></div>',
            unsafe_allow_html=True
        )

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Developed for the Machine Learning Final Project using Streamlit.")