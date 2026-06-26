import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Aplikasi untuk memprediksi potensi pelanggan berhenti berlangganan.")

@st.cache_resource
def load_saved_model():
    return joblib.load('model_churn_terbaik.pkl')

preprocessor, model = load_saved_model()

st.subheader("Input Data Pelanggan")

# Input form berdasarkan fitur paling berpengaruh
satisfaction_score = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
total_spent = st.number_input("Total Spent", min_value=0.0, value=500.0)
support_tickets = st.number_input("Support Tickets", min_value=0, value=2)
avg_session_time = st.number_input("Avg Session Time (Menit)", min_value=0.0, value=30.0)
pages_per_session = st.number_input("Pages per Session", min_value=0.0, value=5.0)

if st.button("Run Prediction"):
    data = {
        'satisfaction_score': [satisfaction_score],
        'total_spent': [total_spent],
        'support_tickets': [support_tickets],
        'avg_session_time': [avg_session_time],
        'pages_per_session': [pages_per_session]
    }
    
    df_input = pd.DataFrame(data)
    
    # Transformasi data & Prediksi
    X_transformed = preprocessor.transform(df_input)
    prediction = model.predict(X_transformed)
    probability = model.predict_proba(X_transformed)[0][1]
    
    st.subheader("Hasil Prediksi:")
    if prediction[0] == 1:
        st.error(f"⚠️ Pelanggan Berpotensi CHURN! (Probabilitas: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Pelanggan Tetap BERTAHAN (Probabilitas Churn: {probability*100:.2f}%)")