import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# =========================
# 1. LOAD DATA
# =========================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

st.title("🏠 Boston Housing ML App (Full Pipeline)")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# =========================
# 2. PREPROCESSING
# =========================
target = "medv"
X = df.drop(columns=[target])
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 3. MODEL 1 - LINEAR REGRESSION
# =========================
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_score = r2_score(y_test, lr_pred)

# =========================
# 4. MODEL 2 - RANDOM FOREST
# =========================
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_score = r2_score(y_test, rf_pred)

# =========================
# 5. MODEL COMPARISON
# =========================
st.subheader("📊 Model Comparison")

compare_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2 Score": [lr_score, rf_score]
})

st.dataframe(compare_df)

st.bar_chart(compare_df.set_index("Model"))

# =========================
# 6. FEATURE IMPORTANCE (RF)
# =========================
st.subheader("🌲 Feature Importance (Random Forest)")

feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
st.bar_chart(feat_imp.sort_values(ascending=False))

# =========================
# 7. PREDICTION UI
# =========================
st.subheader("🔮 Predict House Price")

col1, col2, col3 = st.columns(3)

with col1:
    rm = st.number_input("Average rooms (rm)", value=6.0)

with col2:
    lstat = st.number_input("% lower status population (lstat)", value=10.0)

with col3:
    ptratio = st.number_input("Pupil-teacher ratio (ptratio)", value=18.0)

# We need full feature input (use defaults for simplicity)
def make_input(rm, lstat, ptratio):
    sample = np.mean(X.values, axis=0)
    sample = sample.reshape(1, -1)

    # override 3 important features
    sample[0][X.columns.get_loc("rm")] = rm
    sample[0][X.columns.get_loc("lstat")] = lstat
    sample[0][X.columns.get_loc("ptratio")] = ptratio

    return sample

if st.button("Predict Price"):
    input_data = make_input(rm, lstat, ptratio)
    prediction = rf.predict(input_data)[0]

    st.success(f"🏡 Predicted House Price: ${prediction * 1000:.2f}")