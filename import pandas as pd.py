import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. LOAD AND ANALYZE CUSTOMER DATA
# ==========================================
print("--- 1. Loading and Analyzing Customer Data ---")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'IT_customer_churn.csv')
if not os.path.exists(data_path):
    data_path = os.path.abspath('IT_customer_churn.csv')
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found. Expected file: {data_path}")

df = pd.read_csv(data_path)
df = df.copy()

print(f"Dataset Shape: {df.shape}")
print("Churn Distribution:\n", df['Churn'].value_counts(normalize=True) * 100)

# 2. DATA PREPROCESSING & FEATURE ENGINEERING
# ==========================================
print("\n--- 2. Preprocessing Data ---")
if 'customerID' in df.columns:
    df = df.drop(columns='customerID')

# Handle numerical columns with missing or blank values
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Handle missing values in the remaining columns
for col in df.columns:
    if df[col].dtype.kind in 'ifc':
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    elif df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
        df[col] = df[col].replace(['', ' ', 'nan', 'NaN', 'None', None], np.nan)
        df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0] if not df[col].mode(dropna=True).empty else 'Unknown')

# Encode categorical variables using explicit string dtype selection for Pandas compatibility
for col in df.select_dtypes(include=['object', 'string']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Drop duplicate rows
df = df.drop_duplicates()

# Define features (X) and target (y - Churn)
target_column = 'Churn' if 'Churn' in df.columns else df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# Split data: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Crucial for distance-based KNN and margin-based SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. MODEL 1: KNN (K-Nearest Neighbors) WITH HYPERPARAMETER TUNING
# ==========================================
print("\n--- 3. Training and Tuning KNN Model ---")
knn_params = {
    'n_neighbors': [3, 5, 7, 9, 11], 
    'metric': ['euclidean', 'manhattan']
}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy')
knn_grid.fit(X_train_scaled, y_train)

best_knn = knn_grid.best_estimator_
y_pred_knn = best_knn.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, y_pred_knn)

print(f"Best KNN Parameters: {knn_grid.best_params_}")
print(f"KNN Accuracy: {knn_accuracy:.4f}")
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))


# 4. MODEL 2: SVM (Support Vector Machine) WITH HYPERPARAMETER TUNING
# ==========================================
print("\n--- 4. Training and Tuning SVM Model ---")
svm_params = {
    'C': [0.1, 1, 10], 
    'kernel': ['linear', 'rbf']
}
# Removed probability=True to ensure fast execution and avoid warnings
svm_grid = GridSearchCV(SVC(), svm_params, cv=3, scoring='accuracy')
svm_grid.fit(X_train_scaled, y_train)

best_svm = svm_grid.best_estimator_
y_pred_svm = best_svm.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

print(f"Best SVM Parameters: {svm_grid.best_params_}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# 5. MODEL COMPARISON SUMMARY
# ==========================================
print("\n--- 5. Model Comparison Summary ---")
comparison_df = pd.DataFrame({
    'Model': ['KNN (Optimized)', 'SVM (Optimized)'],
    'Accuracy': [knn_accuracy, svm_accuracy]
})
print(comparison_df)

# Plot model comparison accuracy
plt.figure(figsize=(8, 5))
plt.bar(comparison_df['Model'], comparison_df['Accuracy'], color=['#4C72B0', '#55A868'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
for i, value in enumerate(comparison_df['Accuracy']):
    plt.text(i, value + 0.02, f'{value:.4f}', ha='center', va='bottom', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()