import numpy as np
import matplotlib.pyplot as plt

# data-set
x = np.arange(-3, 10, 0.1)
y = 3 + 2 * np.cos(x) + (1/2) * x + np.random.normal(0.0, 1.0, len(x)) 

n = 11
X = np.zeros((len(x), n), float)
for i in range(n):
    X[:,i] = pow(x,i)

# Least square method
(theta, residuals, rank, s) = np.linalg.lstsq(X, y)

# data-set plot
plt.plot(x, y, 'b.')
# h(theta) plot
y_ = theta[0]
for i in range(1, n):
    y_ += theta[i] * X[:,i]
plt.plot(x, y_, 'r-')

plt.title("linalg.lstsq")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.show()

for i in range(n):
    print('θ[' + str(i) + ']: ' + str(theta[i]))