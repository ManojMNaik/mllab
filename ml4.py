import pandas as pd  # Import pandas for data manipulation

# Define the dataset as a dictionary
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'Play': ['Yes', 'Yes', 'No', 'Yes']  # Target variable
}

df = pd.DataFrame(data)  # Convert the dictionary to a pandas DataFrame

df.to_csv('training_data.csv', index=False)  # Save the DataFrame to a CSV file (no row indices)
df = pd.read_csv('training_data.csv')  # Read the CSV file back into a DataFrame

print(df)  # Print the DataFrame to show the data

X = df.iloc[:, :-1]  # Select all columns except the last as features (X)
y = df.iloc[:, -1]   # Select the last column as the target (y)

# Define the Find-S algorithm function
def find_s_algorithm(X, y):
    hypothesis = ['?'] * X.shape[1]
    for xi, label in zip(X.values, y):
        if label == 'Yes':
            for j, value in enumerate(xi):
                if hypothesis[j] == '?':
                    hypothesis[j] = value
                elif hypothesis[j] != value:
                    hypothesis[j] = '?'
    return hypothesis  # Return the final hypothesis

hypothesis = find_s_algorithm(X, y)  # Run the Find-S algorithm
print("Hypothesis Consistent With the positive examples:", hypothesis)  # Print the result
