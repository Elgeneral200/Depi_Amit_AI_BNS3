import streamlit as st
import pandas as pd
import os
import kagglehub
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -------------------------------
# 1. App Header
# -------------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Prediction App")
st.write("This app uses a Logistic Regression model trained on real heart disease data from Kaggle.")


# -------------------------------
# 2. Cached Data Loading and Training
# -------------------------------
@st.cache_data
def load_and_train():
    # Download dataset (cached by Streamlit)
    path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
    df = pd.read_csv(os.path.join(path, "heart.csv"))

    # Encode categorical features
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'HeartDisease' in categorical_features:
        categorical_features.remove('HeartDisease')
    df_encoded = pd.get_dummies(df[categorical_features])

    # Chi-squared test (optional)
    chi2_stats, p_values = chi2(df_encoded, df['HeartDisease'])

    # Numerical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if "HeartDisease" in numerical_features:
        numerical_features.remove("HeartDisease")

    # Scale numeric data
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_features]), columns=numerical_features)
    df_scaled["HeartDisease"] = df["HeartDisease"]

    # Selected features for training
    num_features = ["Oldpeak", "Age", "FastingBS", "RestingBP", "Cholesterol"]
    X_num = pd.DataFrame(scaler.fit_transform(df[num_features]), columns=num_features)

    cat_features = [
        "ST_Slope_Up",
        "ST_Slope_Flat",
        "ExerciseAngina_Y",
        "ChestPainType_ATA",
        "ChestPainType_ASY",
        "ExerciseAngina_N",
        "Sex_F",
        "ChestPainType_NAP",
        "Sex_M"
    ]

    X_cat = df_encoded[cat_features]
    X = pd.concat([X_num, X_cat], axis=1)
    y = df["HeartDisease"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Logistic Regression model
    model = LogisticRegression(max_iter=3000, solver="liblinear")
    model.fit(X_train, y_train)

    # Accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    #  Return the feature names used during training
    return model, scaler, num_features, X.columns.tolist(), train_acc, test_acc


#  Load model and data (including feature names)
model, scaler, num_features, feature_names, train_acc, test_acc = load_and_train()


# -------------------------------
# 3. Sidebar Inputs
# -------------------------------
st.sidebar.header("🩺 Enter Patient Data")

Age = st.sidebar.number_input("Age", 20, 100, 50)  # min max default 
RestingBP = st.sidebar.number_input("RestingBP", 60, 200, 120)
Cholesterol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
FastingBS = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
Oldpeak = st.sidebar.slider("Oldpeak (ST depression)", 0.0, 6.5, 1.0)

Sex_M = st.sidebar.selectbox("Sex", ["Male", "Female"])
ChestPainType = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY"])
ExerciseAngina = st.sidebar.selectbox("Exercise Angina", ["Y", "N"])
ST_Slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat"])


# -------------------------------
# 4. Input Encoding
# -------------------------------
input_data = {
    "Oldpeak": Oldpeak,
    "Age": Age,
    "FastingBS": FastingBS,
    "RestingBP": RestingBP,
    "Cholesterol": Cholesterol,
    "ST_Slope_Up": 1 if ST_Slope == "Up" else 0,
    "ST_Slope_Flat": 1 if ST_Slope == "Flat" else 0,
    "ExerciseAngina_Y": 1 if ExerciseAngina == "Y" else 0,
    "ExerciseAngina_N": 1 if ExerciseAngina == "N" else 0,
    "ChestPainType_ATA": 1 if ChestPainType == "ATA" else 0,
    "ChestPainType_ASY": 1 if ChestPainType == "ASY" else 0,
    "ChestPainType_NAP": 1 if ChestPainType == "NAP" else 0,
    "Sex_F": 1 if Sex_M == "Female" else 0,
    "Sex_M": 1 if Sex_M == "Male" else 0,
}

input_df = pd.DataFrame([input_data])
input_df[num_features] = scaler.transform(input_df[num_features])

# ✅ FIX: ensure same order and columns as training data
input_df = input_df.reindex(columns=feature_names, fill_value=0)


# -------------------------------
# 5. Prediction
# -------------------------------
st.write("---")
if st.button("🔍 Predict Heart Disease Risk"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result:")
    if pred == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({proba*100:.1f}% probability)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({proba*100:.1f}% probability)")


# -------------------------------
# 6. Model Info
# -------------------------------
st.write("---")
st.metric("Training Accuracy", f"{train_acc*100:.2f}%")
st.metric("Testing Accuracy", f"{test_acc*100:.2f}%")

st.caption("Data Source: Kaggle – Heart Failure Prediction Dataset by fedesoriano")