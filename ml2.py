# 📚 Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# 🏠 Load the California Housing dataset
california_housing = fetch_california_housing()

# 📊 Put the data into a DataFrame and call it 'df'
df = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)

# ➕ Add the target column to df
df['MedHouseVal'] = california_housing.target

# 🔢 Compute the correlation matrix
correlation_matrix = df.corr()

# 🌡️ Show correlation as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of California Housing Dataset')
plt.show()

# 👯‍♂️ Create pair plot to see pairwise relationships
sns.pairplot(df)
plt.suptitle('Pair Plot of California Housing Dataset', y=1.02)
plt.show()

