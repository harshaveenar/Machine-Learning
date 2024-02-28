import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Sample data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Fitting logistic regression model
model = LogisticRegression()
model.fit(x, y)

# Plotting decision boundary
plt.scatter(x, y, color="m", marker="o", s=30)
plt.plot(x, model.predict_proba(x)[:, 1], color="g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
