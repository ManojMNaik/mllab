import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


# Linear Regression on California Housing dataset
def LRC():
    data = fetch_california_housing(as_frame=True)
    X = data.data[["AveRooms"]]
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Plot
    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.plot(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel("Average number of rooms (AveRooms)")
    plt.ylabel("Median Value of Homes ($100,000)")
    plt.title("Linear Regression - California Housing")
    plt.legend()
    plt.show()

    # Metrics
    print("Linear Regression - California Housing")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))


# Polynomial Regression on Auto MPG dataset
def PRAM():
    import urllib.request

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names = ["mpg", "cylinders", "displacement", "horsepower", "weight",
                    "acceleration", "model_year", "origin", "car_name"]

    # Download and decode lines
    response = urllib.request.urlopen(url)
    lines = [line.decode("utf-8").strip() for line in response.readlines()]

    rows = []
    for line in lines:
        parts = line.split(maxsplit=8)
        if len(parts) == 9:
            rows.append(parts)

    # Create DataFrame
    data = pd.DataFrame(rows, columns=column_names)

    # Convert numeric columns
    for col in column_names[:-1]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna()

    x = data["displacement"].values.reshape(-1, 1)
    y = data["mpg"].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), LinearRegression())
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Plot
    plt.scatter(x_test, y_test, color="blue", label="Actual")
    plt.scatter(x_test, y_pred, color="red", label="Predicted")
    plt.xlabel("Displacement")
    plt.ylabel("Miles Per Gallon (MPG)")
    plt.title("Polynomial Regression - Auto MPG")
    plt.legend()
    plt.show()

    # Metrics
    print("Polynomial Regression - Auto MPG")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))


# Main execution
print("Demonstrating Linear and Polynomial Regression")
LRC()
PRAM()
