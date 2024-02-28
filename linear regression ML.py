import numpy as np
import matplotlib.pyplot as plt
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
b = np.polyfit(x, y, 1)
plt.scatter(x, y, color="m", marker="o", s=30)
plt.plot(x, np.polyval(b, x), color="g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plot_regression_line(x, y, b)

plt.show()
