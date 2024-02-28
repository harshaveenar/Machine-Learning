import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Sample data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

# Fitting polynomial regression model
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
model = LinearRegression()
model.fit(x_poly, y)

# Plotting regression line
plt.scatter(x, y, color="m", marker="o", s=30)
plt.plot(x, model.predict(x_poly), color="g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
