import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# --------------------------------
# 1. Symbolic definition
# --------------------------------
x = sp.symbols('x')
g_expr = x**2 + 16/x**2
g_prime_expr = sp.diff(g_expr, x)

print("g(x) =", g_expr)
print("g'(x) =", g_prime_expr)

# Numerical functions
g = sp.lambdify(x, g_expr, 'numpy')
g_prime = sp.lambdify(x, g_prime_expr, 'numpy')

# --------------------------------
# 2. Domains (avoid x = 0)
# --------------------------------
x_left = np.linspace(-4, -0.4, 500)
x_right = np.linspace(0.4, 4, 500)

y_left = g(x_left)
y_right = g(x_right)

dy_left = g_prime(x_left)
dy_right = g_prime(x_right)

# --------------------------------
# 3. Figure and subplots
# --------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 10))

# =================================
# Top plot: g(x)
# =================================
ax1 = axes[0]
ax1.plot(x_left, y_left, label='g(x)')
ax1.plot(x_right, y_right)

# Critical points
crit_x = np.array([-2, 2])
crit_y = g(crit_x)
ax1.plot(crit_x, crit_y, 'ro', label='critical points')

# Tangent at x = 2
x0 = 2
y0 = g(x0)
m = g_prime(x0)
t = np.linspace(1, 3, 100)
tangent_y = y0 + m * (t - x0)
ax1.plot(t, tangent_y, '--', label='tangent at x = 2')

# Labels for intervals
ax1.text(-3.4, 18, 'decreasing', fontsize=10)
ax1.text(-1.3, 20, 'increasing', fontsize=10)
ax1.text(0.9, 20, 'decreasing', fontsize=10)
ax1.text(2.5, 18, 'increasing', fontsize=10)

# Vertical guide near x=0
ax1.axvline(0, linestyle=':', linewidth=1)

ax1.set_title(r"Graph of $g(x)=x^2+\frac{16}{x^2}$")
ax1.set_xlabel("x")
ax1.set_ylabel("g(x)")
ax1.set_xlim(-4.5, 4.5)
ax1.set_ylim(0, 60)
ax1.grid(True)
ax1.legend()

# =================================
# Bottom plot: g'(x)
# =================================
ax2 = axes[1]
ax2.plot(x_left, dy_left, label=r"$g'(x)$")
ax2.plot(x_right, dy_right)

# Zeros of derivative
ax2.plot([-2, 2], [0, 0], 'ro', label="where $g'(x)=0$")

# Sign labels
ax2.text(-3.3, -20, r"$g'(x)<0$", fontsize=10)
ax2.text(-1.4, 20, r"$g'(x)>0$", fontsize=10)
ax2.text(0.9, -20, r"$g'(x)<0$", fontsize=10)
ax2.text(2.5, 20, r"$g'(x)>0$", fontsize=10)

# Axes guides
ax2.axhline(0, linewidth=1)
ax2.axvline(0, linestyle=':', linewidth=1)

ax2.set_title(r"Graph of $g'(x)=2x-\frac{32}{x^3}$")
ax2.set_xlabel("x")
ax2.set_ylabel(r"$g'(x)$")
ax2.set_xlim(-4.5, 4.5)
ax2.set_ylim(-60, 60)
ax2.grid(True)
ax2.legend()

# --------------------------------
# 4. Layout and save
# --------------------------------
plt.tight_layout()
plt.savefig("g_function_static.png", dpi=300)
plt.show()
