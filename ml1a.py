# Import the required libraries
import pandas as pd                      # For handling data in table format (DataFrame)
import matplotlib.pyplot as plt          # For drawing normal plots
import seaborn as sns                    # For drawing beautiful box plots

from sklearn.datasets import fetch_california_housing  # To load the housing dataset

# 📥 Step 1: Load the California Housing dataset
california_housing = fetch_california_housing()

# 🧾 Step 2: Put the data into a DataFrame (like Excel sheet)
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)

# ➕ Add the target column (house value) to the DataFrame
df['MedHouseVal'] = california_housing.target

# 📊 Step 3: Plot histograms for all numerical columns
df.hist(bins=30, figsize=(12, 8), edgecolor='black')  # 30 bars per column
plt.tight_layout()  # Adjust layout so nothing overlaps
plt.show()  # Display the histogram plots

# 📦 Step 4: Plot box plots to visualize spread and outliers
plt.figure(figsize=(12, 8))  # Create a new big figure

# Go through every column and draw its boxplot
for i, column in enumerate(df.columns):
    plt.subplot(3, 4, i + 1)  # Place each plot in a 3x4 grid
    sns.boxplot(x=df[column])  # Draw boxplot horizontally
    plt.title(column)  # Set the title of the plot

plt.tight_layout()  # Make spacing clean
plt.show()  # Show the box plots

# 🚨 Step 5: Detect and count outliers using IQR method
for column in df.columns:
    Q1 = df[column].quantile(0.25)  # 25% value
    Q3 = df[column].quantile(0.75)  # 75% value
    IQR = Q3 - Q1  # Interquartile range

    # Set limits beyond which values are considered outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find all rows where values are below or above the bounds
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    # Print how many outliers were found in this column
    print(f'{column} has {outliers.shape[0]} outliers.')

