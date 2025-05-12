# Import required libraries
import numpy as np                      # For numerical operations
import pandas as pd                    # For creating and handling dataframes
from sklearn.datasets import load_iris # To load the Iris dataset
from sklearn.decomposition import PCA  # For performing Principal Component Analysis
import matplotlib.pyplot as plt        # For plotting

# Load the Iris dataset
iris = load_iris()
print(iris)  # Show structure of the dataset

# Extract features (measurements of flowers)
data = iris.data
print(data)  # Show feature data

# Extract target labels (0 = setosa, 1 = versicolor, 2 = virginica)
labels = iris.target
print(labels)  # Show species as numbers

# Get the names of the species
label_names = iris.target_names
print(label_names)  # ['setosa', 'versicolor', 'virginica']

# Get the feature (column) names
print(iris.feature_names)  # Show names like 'sepal length (cm)', etc.

# Convert the data into a DataFrame for easier handling
iris_df = pd.DataFrame(data, columns=iris.feature_names)
print(iris_df)  # Show full dataframe

# Apply PCA to reduce dimensions from 4 to 2
pca = PCA(n_components=2)               # Create a PCA object to reduce to 2 components
data_reduced = pca.fit_transform(data)  # Apply PCA transformation
print(data_reduced)                     # Show the reduced 2D data

# Convert reduced data into a new DataFrame
reduced_df = pd.DataFrame(data_reduced, columns=['Principle Component 1', 'Principle Component 2'])
print(reduced_df)  # Show new dataframe

# Add the label column to this new DataFrame
reduced_df['label'] = labels
print(reduced_df)  # Show with labels

# Start plotting the PCA result
plt.figure(figsize=(8, 6))  # Set the figure size
colors = ['r', 'g', 'b']    # Colors for each class

# Loop through each unique label and plot its data
for i, lab in enumerate(np.unique(labels)):
    filtered_df = reduced_df[reduced_df['label'] == lab]  # Filter rows with current label
    plt.scatter(filtered_df['Principle Component 1'],
                filtered_df['Principle Component 2'],
                label=label_names[lab],  # Use label name like 'setosa'
                color=colors[i])         # Assign a color to each class

# Add plot title and axis labels
plt.title('PCA on Iris Dataset')
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.legend()  # Show legend
plt.grid()    # Show grid
plt.show()    # Display the plot

