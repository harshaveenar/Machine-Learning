import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
iris_df = pd.read_csv('C:/Users/HP/Documents/python(ML)/IRIS.csv')

# Split the dataset into features and target
X = iris_df.drop('species', axis=1)
y = iris_df['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Perceptron model
perceptron = Perceptron(max_iter=100, eta0=0.1, random_state=42)

# Fit the model to the training data
perceptron.fit(X_train, y_train)

# Predict the target values for the test data
y_pred = perceptron.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Cross-validation
cv_scores = cross_val_score(perceptron, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean Cross-validation score: {np.mean(cv_scores):.2f}")
