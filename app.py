import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("wine_quality.csv")

if not Path("wine_model.pkl").exists():
    st.error("wine_model.pkl not found. Run python train_model.py first.")
    st.stop()
if not Path("wine_quality.csv").exists():
    st.error("wine_quality.csv not found. Run python train_model.py first.")
    st.stop()

model = load_model()
data = load_data()

st.title("🍷 Wine Quality Prediction System")
st.caption("Preliminary wine quality classification")

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average Quality", round(data["quality"].mean(), 2))
c3.metric("Good Quality Records", int((data["quality"] >= 6).sum()))

st.subheader("Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Predict Wine Samples")
st.write("Upload a CSV with the same predictor columns used during training.")
sample = data.drop(columns=["quality"]).head(5)
st.download_button("Download Sample CSV", sample.to_csv(index=False),
                   "sample_wine_input.csv", "text/csv")

uploaded = st.file_uploader("Upload Wine CSV", type=["csv"])
if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        required = list(model.feature_names_in_)
        missing = [c for c in required if c not in new_data.columns]
        if new_data.empty:
            st.warning("CSV file is empty.")
        elif missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            predictions = model.predict(new_data[required])
            results = new_data.copy()
            results["Prediction"] = [
                "GOOD QUALITY" if p == 1 else "NEEDS IMPROVEMENT"
                for p in predictions
            ]
            st.subheader("Prediction Results")
            st.dataframe(results, use_container_width=True)
            st.metric("Predicted Good Quality", int((predictions == 1).sum()))
            st.download_button("Download Prediction Results",
                              results.to_csv(index=False),
                              "wine_quality_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")
