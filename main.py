import streamlit as st
import pandas as pd
# import plotly.express as px
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns



# Load the trained model
rf_model = joblib.load("occupancy_model_rf.pkl")

# Load and display dataset for insights
df_train = pd.read_csv("data-to-train.csv")

st.title("Occupancy Detection Dashboard")

# Data Overview
st.subheader("📊 Dataset Overview")
st.write(df_train.head())

# Feature Distributions
st.subheader("📈 Feature Distributions")
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
colors = sns.color_palette("coolwarm", len(features))
for i, (feature, color) in enumerate(zip(features, colors)):
    sns.histplot(df_train[feature], kde=True, bins=30, ax=ax[i//3, i%3], color=color)
    ax[i//3, i%3].set_title(f"Distribution of {feature}", fontsize=12, fontweight='bold')
    ax[i//3, i%3].grid(True, linestyle='--', alpha=0.6)
    ax[1, 2].axis("off")
st.pyplot(fig)

# Correlation Heatmap
st.subheader("🌀 Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df_train.corr(), annot=True, cmap="Blues", fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)

# Accuracy Metrics Display
st.subheader("📐 Model Accuracy Scores")
model_scores = {
    "Logistic Regression": 0.81,
    "Random Forest": 0.95,
    "SVM": 0.89
}


# Accuracy Dropdown
with st.expander("📌 Accuracy & Tuning Results", expanded=False):
    accuracy_option = st.selectbox("Select Accuracy Visual", [
        "Logistic Regression Accuracy",
        "Random Forest Accuracy",
        "SVM Accuracy",
        "Random Forest Hyperparameter Tuning"
    ])

    if accuracy_option == "Logistic Regression Accuracy":
        st.image("accuracy-lr.png", use_container_width =True)
    elif accuracy_option == "Random Forest Accuracy":
        st.image("accuracy-rf.png", use_container_width =True)
    elif accuracy_option == "SVM Accuracy":
        st.image("accuracy-svm.png", use_container_width =True)
    elif accuracy_option == "Random Forest Hyperparameter Tuning":
        st.image("tuning-rf.png", use_container_width =True)

st.markdown("### ✅ Key Insights")
st.markdown("""
- Higher Light and CO2 levels are strong indicators of occupancy.  
- The Random Forest model shows the best accuracy among the models evaluated.  
- The model can be used for real-time and batch occupancy detection in smart buildings.
""")

# Use Cases Section
st.markdown("### 🔍 Real-World Use Cases")
st.markdown("""
- 🏢 Smart Office Spaces: Monitor which rooms are occupied to save energy.  
- 🏫 Educational Institutions: Track room usage for classroom optimization.  
- 🏠 Smart Homes: Integrate with HVAC or lighting systems for automation.  
- 🧪 Research Labs: Maintain safe and efficient lab environments based on presence.
""")


# Sidebar for manual input
st.header("Occupancy Prediction")

def get_min_max(column):
    return df_train[column].min(), df_train[column].max()

temp_min, temp_max = get_min_max("Temperature")
hum_min, hum_max = get_min_max("Humidity")
light_min, light_max = get_min_max("Light")
co2_min, co2_max = get_min_max("CO2")
humratio_min, humratio_max = get_min_max("HumidityRatio")

temperature = st.number_input("Temperature in °C", min_value=temp_min, max_value=temp_max, value=temp_min)
humidity = st.number_input("Humidity in %", min_value=hum_min, max_value=hum_max, value=hum_min)
light = st.number_input("Light in Lux", min_value=light_min, max_value=light_max, value=light_min)
co2 = st.number_input("CO2 in ppm", min_value=co2_min, max_value=co2_max, value=co2_min)
humidity_ratio = st.number_input("Humidity Ratio in kgwater-vapor/kg-air", min_value=humratio_min, max_value=humratio_max, value=humratio_min)

# Predict button
if st.button("Predict Occupancy"):
    input_data = np.array([[temperature, humidity, light, co2, humidity_ratio]])
    prediction = rf_model.predict(input_data)[0]
    occupancy_status = "Occupied" if prediction == 1 else "Unoccupied"
    st.success(f"Predicted Occupancy: {occupancy_status}")

# Sidebar for file upload
st.sidebar.header("Upload CSV for Batch Analysis")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data:")
    st.write(df.head())
    
    # Ensure correct columns
    feature_cols = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    if all(col in df.columns for col in feature_cols):
        X = df[feature_cols]
        predictions = rf_model.predict(X)
        df['Predicted Occupancy'] = predictions
        
        st.write("### Predictions:")
        st.write(df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'Predicted Occupancy']].head())
        
        
        # Feature Importance
        st.subheader("🔥 Feature Importance (Random Forest)")
        feature_importance = rf_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': feature_importance})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        
        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=importance_df, palette="magma", ax=ax)
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        st.pyplot(fig)
        
    else:
        st.error("Uploaded file does not have the required columns: Temperature, Humidity, Light, CO2, HumidityRatio")
else:
    st.write("👈 Upload a CSV file or enter data to analyze occupancy!")
