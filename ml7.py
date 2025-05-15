# ML Lab Program 7
# Part 1: Linear Regression using California Housing Dataset
# Part 2: Polynomial Regression using Auto MPG Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# =================== Part 1: Linear Regression ===================

print("Part 1: Linear Regression - California Housing Dataset")
print("-" * 60)

from sklearn.datasets import fetch_california_housing

# Load dataset
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target

# Split features and target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict
y_pred_train = lr_model.predict(X_train)
y_pred_test = lr_model.predict(X_test)

# Evaluate
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Display Results
print(f"Training MSE: {train_mse:.2f}")
print(f"Testing MSE: {test_mse:.2f}")
print(f"Training R²: {train_r2:.2f}")
print(f"Testing R²: {test_r2:.2f}")

# Plot Feature Importance
plt.figure(figsize=(12, 6))
plt.bar(X.columns, lr_model.coef_)
plt.xticks(rotation=45)
plt.title('Feature Importance in Linear Regression')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.tight_layout()
plt.show()

# =================== Part 2: Polynomial Regression ===================

print("\nPart 2: Polynomial Regression - Auto MPG Dataset")
print("-" * 60)

# Load Auto MPG dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url)

# Clean data
df = df.dropna()
df = df[df['horsepower'] != '?']  # Remove invalid entries
df['horsepower'] = df['horsepower'].astype(float)

# Use 'horsepower' to predict 'mpg'
X_poly = df[['horsepower']]
y_poly = df['mpg']

# Convert to polynomial features (degree=2)
poly = PolynomialFeatures(degree=2)
X_poly_transformed = poly.fit_transform(X_poly)

# Train-test split
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_poly_transformed, y_poly, test_size=0.2, random_state=42)

# Train polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_train_p, y_train_p)

# Predict
y_pred_poly = poly_model.predict(X_test_p)

# Evaluate
poly_mse = mean_squared_error(y_test_p, y_pred_poly)
poly_r2 = r2_score(y_test_p, y_pred_poly)

print(f"Polynomial Regression MSE: {poly_mse:.2f}")
print(f"Polynomial Regression R²: {poly_r2:.2f}")

# Plot Polynomial Curve
plt.figure(figsize=(10, 6))
# Sort X for smooth curve
sorted_hp = np.sort(X_poly.values.flatten())
sorted_hp_poly = poly.transform(sorted_hp.reshape(-1, 1))
plt.scatter(X_poly, y_poly, color='gray', alpha=0.5, label='Data')
plt.plot(sorted_hp, poly_model.predict(sorted_hp_poly), color='red', label='Polynomial Fit')
plt.title("Polynomial Regression (MPG vs Horsepower)")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

