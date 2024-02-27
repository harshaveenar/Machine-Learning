import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import datasets

# Load iris dataset as an example
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Polynomial Features object
poly = PolynomialFeatures(degree=2)

# Transform the X data for polynomial regression
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

# Perform the polynomial regression
polyreg = LinearRegression()
polyreg.fit(X_train_poly, y_train)

# Make predictions using the testing set
y_pred = polyreg.predict(X_test_poly)

# Plot predicted values against true values
plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Polynomial Regression: True Values vs Predicted Values")
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import datasets

# Load iris dataset as an example
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Polynomial Features object
poly = PolynomialFeatures(degree=2)

# Transform the X data for polynomial regression
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

# Perform the polynomial regression
polyreg = LinearRegression()
polyreg.fit(X_train_poly, y_train)

# Make predictions using the testing set
y_pred = polyreg.predict(X_test_poly)

# Plot predicted values against true values
plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Polynomial Regression: True Values vs Predicted Values")
plt.show()
