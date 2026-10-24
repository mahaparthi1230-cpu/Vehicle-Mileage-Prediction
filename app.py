"""
Automobile Fuel Efficiency (MPG) Prediction App
Run locally:  streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "auto-mpg.csv"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model year",
]
CATEGORICAL_FEATURES = ["origin"]
ORIGIN_MAP = {1: "USA", 2: "Europe", 3: "Japan"}
TARGET = "mpg"


# ----------------------------------------------------------------------------
# 1. DATA EXTRACTION
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Dataset '{path}' not found. Place it next to app.py.")
        st.stop()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# ----------------------------------------------------------------------------
# 2. CLEANING (missing values: horsepower contains '?')
# ----------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")  # '?' -> NaN
    df["origin"] = df["origin"].map(ORIGIN_MAP)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])  # target can't be imputed
    return df


# ----------------------------------------------------------------------------
# 3. PREPROCESSING + MODEL PIPELINE
#    numeric: median imputation -> standard scaling
#    categorical: most-frequent imputation -> one-hot encoding
# ----------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
    ])


# ----------------------------------------------------------------------------
# 4. TRAINING (cached so it only runs once per server session)
# ----------------------------------------------------------------------------
@st.cache_resource
def train_model():
    df = clean_data(load_data())
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    metrics = {
        "R2": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": float(np.sqrt(np.mean((y_test - preds) ** 2))),
    }
    return pipeline, metrics, df


# ----------------------------------------------------------------------------
# 5. STREAMLIT UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="MPG Predictor", page_icon="🚗", layout="wide")
    st.title("🚗 Automobile Fuel Efficiency (MPG) Predictor")
    st.caption("Enter the car's specifications to estimate its miles per gallon.")

    model, metrics, df = train_model()

    # ---- Sidebar inputs ----
    st.sidebar.header("Car specifications")
    cylinders = st.sidebar.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
    displacement = st.sidebar.slider("Displacement (cu. in.)", 60.0, 460.0, 200.0, 1.0)
    horsepower = st.sidebar.slider("Horsepower", 40.0, 250.0, 100.0, 1.0)
    weight = st.sidebar.slider("Weight (lbs)", 1500, 5200, 3000, 10)
    acceleration = st.sidebar.slider("Acceleration (0-60 mph, sec)", 8.0, 25.0, 15.0, 0.1)
    model_year = st.sidebar.slider("Model year (19xx)", 70, 82, 76)
    origin = st.sidebar.selectbox("Origin", list(ORIGIN_MAP.values()))

    input_df = pd.DataFrame([{
        "cylinders": cylinders,
        "displacement": displacement,
        "horsepower": horsepower,
        "weight": weight,
        "acceleration": acceleration,
        "model year": model_year,
        "origin": origin,
    }])

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Your input")
        st.dataframe(input_df, use_container_width=True, hide_index=True)
        if st.button("Predict MPG", type="primary"):
            pred = float(model.predict(input_df)[0])
            st.metric("Predicted fuel efficiency", f"{pred:.1f} MPG")
            st.metric("≈ km per litre", f"{pred * 0.425144:.1f} km/L")
            st.metric("≈ litres per 100 km", f"{235.215 / pred:.1f} L/100km")

    with col2:
        st.subheader("Model performance (20% hold-out set)")
        m1, m2, m3 = st.columns(3)
        m1.metric("R²", f"{metrics['R2']:.3f}")
        m2.metric("MAE", f"{metrics['MAE']:.2f}")
        m3.metric("RMSE", f"{metrics['RMSE']:.2f}")
        st.caption("Random Forest with median imputation, standard scaling "
                   "and one-hot encoding.")

    with st.expander("Explore the dataset"):
        st.write(f"Rows: {len(df)} | Missing horsepower values imputed: "
                 f"{int(df['horsepower'].isna().sum())}")
        st.dataframe(df.head(50), use_container_width=True)
        st.bar_chart(df.groupby("model year")[TARGET].mean())


if __name__ == "__main__":
    main()
