import numpy as np  # For numerical operations and random numbers
from collections import Counter  # For counting most common elements
import matplotlib.pyplot as plt  # For plotting

# Step 1: Generate 100 random values between 0 and 1
np.random.seed(42)  # Set seed for reproducibility
x = np.random.rand(100)  # Generate 100 random numbers in [0, 1)

# Step 2: Label the first 50 points
labels = []  # List to store labels
for xi in x[:50]:  # For each of the first 50 points
    if xi <= 0.5:
        labels.append('Class1')  # Label as Class1 if value <= 0.5
    else:
        labels.append('Class2')  # Otherwise, label as Class2

# Step 3: Define k-NN classification function
def knn_classify(x_train, y_train, x_test, k):
    predictions = []  # List to store predictions
    for test_point in x_test:  # For each test point
        # Compute absolute distances to all training points
        distances = [abs(test_point - train_point) for train_point in x_train]
        # Get indices of k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        # Get their labels
        nearest_labels = [y_train[i] for i in nearest_indices]
        # Find the most common label among neighbors
        most_common = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(most_common)  # Add prediction
    return predictions  # Return all predictions

# Step 4: Classify the remaining 50 points for different k values
k_values = [1, 2, 3, 4, 5, 20, 30]  # Different k values to try
results = {}  # Dictionary to store results

x_train = x[:50]  # Training data (first 50 points)
y_train = labels  # Training labels
x_test = x[50:]   # Test data (last 50 points)

for k in k_values:  # For each k value
    predictions = knn_classify(x_train, y_train, x_test, k)  # Classify test points
    results[k] = predictions  # Store predictions

# Step 5: Print the predictions for each k
for k in k_values:
    print(f"\nPredictions for k = {k}:")
    for i, pred in enumerate(results[k], start=51):  # x[50] is x51
        print(f"x{i} = {x[i-1]:.3f} → {pred}")

# Step 6: Visualization
colors = {'Class1': 'blue', 'Class2': 'red'}  # Color mapping for classes
fig, ax = plt.subplots(figsize=(10, 5))  # Create a figure

# Plot training points at y=0
ax.scatter(x[:50], [0]*50, c=[colors[l] for l in y_train], label='Training Points')

# Plot test points for each k at y=k
for k in k_values:
    ax.scatter(x[50:], [k]*50, c=[colors[l] for l in results[k]], label=f'k={k}')

ax.set_yticks([0] + k_values)  # Set y-ticks for clarity
ax.set_xlabel("x values")  # Label x-axis
ax.set_title("k-NN Classification Results")  # Title
ax.legend()  # Show legend
plt.grid(True)  # Show grid
plt.tight_layout()  # Adjust layout
plt.show()  # Display plot
