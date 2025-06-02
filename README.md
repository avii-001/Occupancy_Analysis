## 🏠 Occupancy Detection Dashboard
An intelligent, interactive Streamlit-powered dashboard that analyzes and predicts room occupancy using machine learning techniques. This project showcases practical applications of AI/ML in smart environments and energy optimization using real-world sensor data.

🔗 Streamlit App: Occupancy Detection Dashboard
https://occupancyanalysis.streamlit.app/

🚀 Features
📊 Dataset Overview: Quickly view key sensor data like Temperature, Humidity, CO₂, Light, and Humidity Ratio.
📈 Feature Distributions: Visualize distributions for each variable using elegant histograms.
🌀 Correlation Heatmap: Detect multicollinearity and key influencing features.
🎯 Model Accuracy Section: Compare multiple ML models (Logistic Regression, Random Forest, SVM) with visuals and performance scores.
📷 Model Visuals Dropdown: Interactive dropdown for model accuracy and tuning charts.
📂 Batch CSV Upload: Upload datasets and get real-time predictions for multiple observations.
🧠 Manual Prediction Tool: Enter sensor values manually and get instant occupancy predictions.
🔥 Feature Importance: Understand which features drive the model’s predictions.
📊 Occupancy Distribution Plot: Analyze the frequency of occupied vs. unoccupied states.

🗃️ Dataset
Source: UCI Occupancy Detection Dataset

Key Columns:

Temperature

Humidity

Light

CO2

HumidityRatio

Occupancy (Target)

🧠 Machine Learning Models
Model	Accuracy
Logistic Regression	81%
Random Forest	95% ✅
Support Vector Machine	89%

Best Model: Random Forest Classifier

Tuning: GridSearchCV used to fine-tune hyperparameters of RF model

🛠️ Tech Stack
Streamlit – Interactive UI for web-based analysis

Scikit-learn – ML models & evaluation

Pandas & NumPy – Data manipulation and preprocessing

Matplotlib & Seaborn – Static data visualizations

Plotly – Interactive bar plots and charts

Joblib – Model persistence

Python – Primary language used for modeling, logic, and UI

📷 Screenshots
Accuracy Dropdown	Predictions Table	Feature Importance	Manual Prediction
	

📌 Insights Derived
🌞 Light and CO₂ are the strongest predictors of room occupancy.

🤖 The Random Forest model outperforms others with ~95% accuracy.

⚙️ Feature importance shows explainability and transparency of model behavior.

💡 Can be integrated into Smart Building Systems for energy-saving automation.

👩‍💻 Author

Kusumm Maharjan - https://github.com/avii-001

📈 Future Improvements
🔗 Integrate live IoT sensor streams via APIs

🧠 Add deep learning models or ensemble stacking for prediction

🧩 Enable real-time model re-training with new data

🗂️ Add multi-room or time-based prediction for larger facilities
