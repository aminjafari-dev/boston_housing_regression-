import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Define column names for the Boston Housing dataset
column_names = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']

# 2. Load the data with space as delimiter and assign column names
data = pd.read_csv('housing.csv', delim_whitespace=True, names=column_names)

# 3. Verify the data
print("First few rows of the dataset:")
print(data.head())
print("\nMissing values:")
print(data.isnull().sum())

# 4. Define features and target
X = data.drop('medv', axis=1)  # All columns except medv
y = data['medv']  # Target column

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Gradient Descent implementation
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    X = np.hstack([np.ones((m, 1)), X])  # Add bias term
    theta = np.zeros(n + 1)  # Initialize weights
    y = np.array(y)
    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradients = (1/m) * X.T.dot(errors)
        theta -= learning_rate * gradients
    return theta

# --- Case A: Using only lstat ---
# Select lstat column
X_lstat = data[['lstat']]
X_lstat_train, X_lstat_test, y_lstat_train, y_lstat_test = train_test_split(X_lstat, y, test_size=0.2, random_state=42)

# Standardize lstat
X_lstat_train_scaled = scaler.fit_transform(X_lstat_train)
X_lstat_test_scaled = scaler.transform(X_lstat_test)

# Train model with Gradient Descent
theta_lstat = gradient_descent(X_lstat_train_scaled, y_lstat_train)

# Predict on test set
X_lstat_test_scaled = np.hstack([np.ones((X_lstat_test_scaled.shape[0], 1)), X_lstat_test_scaled])
y_lstat_pred = X_lstat_test_scaled.dot(theta_lstat)

# Calculate SSE for Case A
sse_lstat = np.sum((y_lstat_test - y_lstat_pred) ** 2)
print(f"\nSSE for Case A (lstat only): {sse_lstat:.2f}")

# --- Case B: Using all features ---
# Train model with Gradient Descent
theta_all = gradient_descent(X_train_scaled, y_train)

# Predict on test set
X_test_scaled = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])
y_all_pred = X_test_scaled.dot(theta_all)

# Calculate SSE for Case B
sse_all = np.sum((y_test - y_all_pred) ** 2)
print(f"SSE for Case B (all features): {sse_all:.2f}")