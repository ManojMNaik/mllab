from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = clf.predict(X_test)
print('Model Accuracy:', accuracy_score(y_test, y_pred))
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Step 5: Print decision tree rules
print('\nDecision Tree Rules:\n')
print(export_text(clf, feature_names=list(data.feature_names)))

# Step 6: Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title('Decision Tree Visualization')
plt.show()

# Step 7: Classify a new sample
sample = X_test[0].reshape(1, -1)
prediction = clf.predict(sample)
print("\nPrediction for a new sample:", data.target_names[prediction[0]])

