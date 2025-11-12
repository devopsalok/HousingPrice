# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Must be FIRST Streamlit command
st.set_page_config(page_title="🏠 Housing Price Predictor", layout="wide")

# Now you can start writing the app
st.title("🏠 Housing Price Prediction App")

st.write("Upload your dataset to get started!")

@st.cache_data
def load_model():
    data = joblib.load('model.joblib')
    return data

data = load_model()
pipeline = data['pipeline']
FEATURE_NAMES = data['feature_names']
X_test = data['X_test']
y_test = data['y_test']
preds = data['preds']

# Tabs
tab_pred, tab_insight = st.tabs(["🔮 Predict", "📊 Model Insights"])

# --------- Prediction Tab -----------
with tab_pred:
    st.subheader("Single Prediction")
    input_data = {}
    for f in FEATURE_NAMES:
        input_data[f] = st.number_input(f"{f}", value=float(X_test[f].mean()), step=0.1)
    if st.button("Predict"):
        df = pd.DataFrame([input_data])
        pred = pipeline.predict(df)[0]
        st.metric("Predicted Median House Value", f"${pred * 100_000:,.0f}")

    st.markdown("---")
    st.subheader("Batch Prediction (Upload CSV)")
    uploaded = st.file_uploader("Upload a CSV with same feature columns", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        preds = pipeline.predict(df)
        df['PredictedValue(100kUSD)'] = preds
        st.dataframe(df.head(20))
        csv_out = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv_out, "predictions.csv", "text/csv")

# --------- Insights Tab -----------
with tab_insight:
    st.header("Model Insights and Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Importance")
        st.image("feature_importance.png", use_container_width=True)

    with col2:
        st.subheader("Prediction Distribution")
        st.image("prediction_distribution.png", use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Actual vs Predicted")
        st.image("actual_vs_predicted.png", use_container_width=True)

    with col4:
        st.subheader("Feature Correlation Heatmap")
        st.image("correlation_heatmap.png", use_container_width=True)

    st.markdown("---")

    st.subheader("Interactive Feature Importance")
    import numpy as np
    importances = pipeline.named_steps['model'].feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': FEATURE_NAMES, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis', ax=ax)
    st.pyplot(fig)
