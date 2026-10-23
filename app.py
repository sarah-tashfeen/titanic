"""
Titanic Passenger Survival Prediction System  -  Streamlit web app
Student : SARAH TASHFEEN
Reg. No : FA24-BED-102
Course  : Machine Learning Fundamentals (AIC354), CUI Lahore Campus
"""
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "svc_trained_model.pkl")
DATA_FILE = os.path.join(BASE_DIR, "sample-data.csv")

# ---- Same fixed category lists used in the notebook (Step 5.1) ----
CATEGORIES = {
    "PClass": ["First", "Second", "Third"],
    "Gender": ["Male", "Female"],
    "Sibling": ["Zero", "One", "Two", "Three"],
    "Embarked": ["Southampton", "Cherbourg", "Queenstown"],
}
FEATURES = list(CATEGORIES.keys())

encoders = {}
for col, cats in CATEGORIES.items():
    enc = LabelEncoder()
    enc.fit(np.ravel(cats))
    encoders[col] = enc
survived_encoder = LabelEncoder()
survived_encoder.fit(np.ravel(["Yes", "No"]))


def encode(df):
    out = df.copy()
    for col in FEATURES:
        out[col] = encoders[col].transform(df[col])
    return out


def retrain_model():
    """Fallback: re-train the SVC exactly like the notebook (80/20 split, no shuffle)."""
    data = pd.read_csv(DATA_FILE)
    enc = encode(data[FEATURES])
    enc["Survived"] = survived_encoder.transform(data["Survived"])
    n_train = int(np.ceil(len(enc) * 0.8))
    train = enc.iloc[:n_train]
    model = svm.SVC(gamma="auto", random_state=0)
    model.fit(train[FEATURES], np.ravel(train["Survived"]))
    return model


@st.cache_resource
def load_model():
    """Load the saved model; if the file cannot be loaded, re-train from the CSV."""
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        test = pd.DataFrame([[0, 0, 0, 0]], columns=FEATURES)
        model.predict(test)  # make sure it really works in this environment
        return model
    except Exception:
        return retrain_model()


st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Passenger Survival Prediction")
st.write("Enter the passenger details below and the trained **Support Vector Classifier** "
         "will predict whether the passenger would have survived.")

model = load_model()

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger Class (PClass)", CATEGORIES["PClass"])
    sibling = st.selectbox("Siblings / Spouses aboard (Sibling)", CATEGORIES["Sibling"])
with col2:
    gender = st.selectbox("Gender", CATEGORIES["Gender"])
    embarked = st.selectbox("Port of Embarkation (Embarked)", CATEGORIES["Embarked"])

if st.button("Predict Survival"):
    user_input = pd.DataFrame({"PClass": [pclass], "Gender": [gender],
                               "Sibling": [sibling], "Embarked": [embarked]})
    features = encode(user_input)[FEATURES]
    result = int(model.predict(features)[0])
    if result == 1:
        st.success("Prediction: **SURVIVED** ✅")
    else:
        st.error("Prediction: **NOT SURVIVED** ❌")
    st.caption("Input: " + ", ".join(f"{k} = {v}" for k, v in user_input.iloc[0].items()))

st.markdown("---")
st.caption("Developed by SARAH TASHFEEN (FA24-BED-102) | AIC354 Machine Learning Fundamentals | "
           "COMSATS University Islamabad, Lahore Campus")
