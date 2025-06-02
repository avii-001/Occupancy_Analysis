# 🏠 Occupancy Detection Dashboard
An interactive Streamlit-based dashboard that detects and predicts room occupancy based on real-time environmental sensor data. This project combines machine learning, feature engineering, and intuitive visualization to power smart building decisions and optimize energy efficiency.

## 🚀 Features
📊 **Data Exploration**: View detailed distributions of temperature, humidity, light, CO₂, and humidity ratio.
🔥 **Feature Importance Display**: See which variables most strongly impact occupancy predictions.
🧠 **ML Model Comparison**: Evaluate Logistic Regression, Random Forest, and SVM with accuracy charts.
📉 **Interactive Visualizations**: Built with Plotly and Seaborn for responsive, aesthetic charts.
📂 **Batch Prediction** : Upload your own CSV and get occupancy predictions for multiple rooms at once.
🧮 **Manual Prediction Tool**: Enter sensor values and instantly know if the room is likely occupied.
🎛️ **Model Insights & Tuning**: Explore accuracy, F1 scores, and hyperparameter optimization.
📈 **Occupancy Distribution Char**t: Quickly understand data imbalance and prediction difficulty.

## 🗃️ Dataset
https://www.kaggle.com/datasets/pooriamst/occupancy-detection/data

Columns include:
- `Temperature`
- `Humidity`
- `Light`
- `CO2`
- `HumidityRatio`
- `Occupancy` (Target: 0 or 1)

## 🛠️ Technologies Used

- Streamlit – for building the dashboard UI
- Scikit-learn – for model training and evaluation
- Pandas & NumPy – for data handling
- Seaborn & Matplotlib – for visual analytics
- Plotly – for interactive visuals
- Joblib – for model saving and loading

## 🧪 Model Evaluation
- ✅ Random Forest Classifier achieved ~95% accuracy
- Logistic Regression and SVM also performed well (~81-89%)

## Evaluation Metrics:
- Accuracy
- Precision/Recall
- F1 Score
- Confusion Matrix
- ROC AUC Curve

## 📌 Insights Derived
- 💡 Light and CO₂ are the strongest indicators of occupancy.
- 🧠 Random Forest outperforms linear models due to its non-linearity and ensemble nature.
- 📉 Class imbalance observed, requiring balanced metric evaluation.
- 🏢 Potential for integration into smart HVAC systems for dynamic control.

## 📈 Future Improvements
- 🌐 Integrate live IoT sensor API for real-time predictions.
- 📦 Add LSTM or Deep Learning-based temporal models.
- 📊 Support time-series occupancy trend forecasting.
- 🔒 Secure user input and improve error handling.

### 👩‍💻 Author
Kusumm Maharjan
🔗 GitHub: https://github.com/avii-001
