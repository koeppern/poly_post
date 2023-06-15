# Example: fit 1D polynomial
# Create example data first
# 2023-06-15
# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
c = np.array([1,2,3])

x = np.array([10,12,15,17,20])

p = np.poly1d(c)

y = p(x)

# %%
np.random.seed(1)

y_m = y + np.random.normal(0, 0.05 * np.abs(y), size=len(y))


# %%
p_est = np.poly1d(np.polyfit(x, y, deg=2))

display(p_est)
# %%
plt.figure(dpi=150)

# Scatter plot of blue points
plt.scatter(x, y_m, color='blue', label='Measurement')

# Plot polynomial curve in red
x_vals = np.linspace(x.min(), x.max(), 100)
plt.plot(x_vals, p_est(x_vals), color='red', label='Polynomial, degree=2')

# Add legend
plt.legend()

plt.grid(True)

plt.xlabel("x")

plt.ylabel("y")

plt.title("Fiiting of 1D poly of degree 2 into data points")

# Display the plot
plt.show()

plt.savefig('figures/poly1d_scatter.png')
# %%