

from google.colab import drive
drive.mount('/content/drive')


"""1. Import Libraries"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

"""2. Load dataset

"""

df =pd.read_csv('/content/drive/MyDrive/kusumm/Data Science/Final-Project/datatrain.csv')
df.head()

# Display basic info
df.info()

# Check if theres any null values
df.isnull().sum()

"""3. Exploratory Data Analysis (EDA)

"""

df['date']=pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

df.info()

df.head()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

features = ['Temperature', 'Humidity', 'Light', 'CO2']
for col in features:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(df[df['Occupancy'] == 1][col], shade=True, label="Occupied")
    sns.kdeplot(df[df['Occupancy'] == 0][col], shade=True, label="Unoccupied")
    plt.title(f'Distribution of {col} by Occupancy')
    plt.legend()
    plt.show()

"""### Feature Engineering"""

# Selecting features and target variable
X= df.drop(columns=['Occupancy'])
y=df['Occupancy']

# Normalize/Standarized features
scaler=StandardScaler()
X=scaler.fit_transform(X)

# Spliting dataset into train and test sets(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

"""### Train Multiple Models

"""

# Model Training
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(max_depth=None, min_samples_split=2, n_estimators=150),
    'Support Vector Machine': SVC(C=10, gamma='scale', kernel='rbf')
}

predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, predictions[name]):.2f}")
    print(classification_report(y_test, predictions[name]))

"""### Hyperparameter Tuning"""

# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10]
# }

# grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# print("Best Parameters:", grid_search.best_params_)
# best_model = grid_search.best_estimator_

# O/p
# Best Parameters: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 150}

# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'C': [0.1, 1, 10],
#     'kernel': ['linear', 'rbf'],
#     'gamma': ['scale', 'auto']
# }

# grid_search= GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# print("Best Parameters:", grid_search.best_params_)
# best_model = grid_search.best_estimator_

# O/p
# Best Parameters: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}

"""### Model Evaluation

1. Confusion Matrix
"""

# Model Evaluation
fig, axes = plt.subplots(2, 2, figsize=(9, 7))

labels = ['Unoccupied', 'Occupied']
for ax, (name, pred) in zip(axes.flatten(), predictions.items()):
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'Confusion Matrix - {name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
axes[1, 1].axis("off")
plt.tight_layout()
plt.show()

"""2. Comparing Model Performance"""

plt.figure(figsize=(6, 4))
Model=list(predictions.keys())
AccuracyScore=[accuracy_score(y_test,pred) for pred in predictions.values()]
sns.barplot(x=Model, y=AccuracyScore, palette='viridis')
plt.ylabel('Accuracy')
plt.title('Comparing Model Performance')
plt.legend()
plt.show()

"""Insight: Random Forest or SVM might perform the best.
* Can learn complex relationship between features and target variables.
* Measures feature importance
* Robustness to Overfitting compared to individual decision trees due to ensemble nature of model.
*Less sensituve to outliers
"""

feature_importance = rf.feature_importances_
features = np.array(df.drop(columns=['Occupancy']).columns)
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance Random Forest")
plt.show()

# Convert model into a pickle file
import joblib
joblib.dump(rf, 'occupancy_model_rf.pkl')              # Save the trained model

# # Load and make predictions
# loaded_model = joblib.load('occupancy_model.pkl')
# sample_input = X_test[0].reshape(1, -1)
# prediction = loaded_model.predict(sample_input)
# print("Predicted Occupancy:", prediction)

