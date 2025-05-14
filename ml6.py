import numpy as np
import matplotlib.pyplot as plt

# 1. 🎲 Generate a wave-like dataset (sin(x)) with some random noise
np.random.seed(0)
X = np.linspace(-3, 3, 100)  # 100 values from -3 to +3
Y = np.sin(X) + np.random.normal(0, 0.1, 100)  # y = sin(x) + some noise

# 2. ➕ Add a column of 1s to X for bias (so our math can learn intercepts too)
X_mat = np.c_[np.ones(X.shape[0]), X]  # Shape becomes (100, 2)

# 3. 🔍 Define a Gaussian kernel to give weights to nearby points
def kernel(x0, X, tau):
    # Compute closeness of x0 to every row in X using Gaussian formula
    return np.exp(-np.sum((X - x0)**2,) / (2 * tau**2))

# 4. 🧠 Perform Locally Weighted Regression
def locally_weighted_regression(x0, X, Y, tau):
    m = X.shape[0]
    W = np.eye(m)  # Make identity matrix (diagonal matrix for weights)
    for i in range(m):
        W[i, i] = kernel(x0, X[i], tau)  # Assign weights to each point
    # Calculate theta using weighted linear regression formula
    theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ Y
    return x0 @ theta  # Return predicted value at x0

# 5. 🔁 Predict Y for each X using LWR
tau = 0.5  # Smaller = more local, larger = smoother
y_pred = np.array([locally_weighted_regression(x0, X_mat, Y, tau) for x0 in X_mat])

# 6. 🖍️ Plot original data and prediction line
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label='Training Data', alpha=0.6)  # Actual noisy points
plt.plot(X, y_pred, color='red', label=f'LWR Prediction (tau={tau})')  # Smooth line
plt.title("Locally Weighted Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

