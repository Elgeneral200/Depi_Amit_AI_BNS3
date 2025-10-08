import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 7, 11])
w = 0.0
b = 0.0
alpha = 0.01  
num_iteration = 20
sse_values = []

n = len(x)

# Gradient Descent
for i in range(num_iteration):
    y_hat = w * x + b

    D_w = (2 / n) * np.sum((y_hat - y) * x)
    D_b = (2 / n) * np.sum(y_hat - y)

    w -= alpha * D_w
    b -= alpha * D_b

    sse = np.sum((y_hat - y) ** 2)
    sse_values.append(sse)

    if (i+1) % 10 == 0 or i == 0:
        print(f"Iteration {i+1}, SSE: {sse:.4f}")

# Plot SSE over iterations
#plt.plot(range(1, num_iteration + 1), sse_values, marker='o')
#plt.xlabel('Iteration')
#plt.ylabel('Sum of Squared Errors (SSE)')
#plt.title('SSE over Iterations')
#plt.show()

plt.figure(figsize=(12, 5))
plt.scatter(x, y, color="blue", label='Data points')
plt.plot(x, w * x + b, color='red', linewidth=2, label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)
plt.show()