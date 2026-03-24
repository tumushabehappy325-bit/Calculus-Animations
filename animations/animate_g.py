import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

# --------------------------------
# 1. Symbolic definition
# --------------------------------
x = sp.symbols('x')
g_expr = x**2 + 16 / x**2
g_prime_expr = sp.diff(g_expr, x)

print("g(x) =", g_expr)
print("g'(x) =", g_prime_expr)

g = sp.lambdify(x, g_expr, 'numpy')
g_prime = sp.lambdify(x, g_prime_expr, 'numpy')

# --------------------------------
# 2. Split domain to avoid x = 0
# --------------------------------
x_left = np.linspace(-4, -0.4, 250)
x_right = np.linspace(0.4, 4, 250)

y_left = g(x_left)
y_right = g(x_right)

# --------------------------------
# 3. Figure setup
# --------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(r"Animation of $g(x)=x^2+\frac{16}{x^2}$ with tangent")
ax.set_xlabel("x")
ax.set_ylabel("g(x)")
ax.grid(True)

ax.set_xlim(-4.5, 4.5)
ax.set_ylim(0, 60)

# faint full graph in the background
ax.plot(x_left, y_left, alpha=0.25)
ax.plot(x_right, y_right, alpha=0.25)

curve_line, = ax.plot([], [], lw=2, label='g(x)')
tangent_line, = ax.plot([], [], '--', lw=2, label='tangent')
point_dot, = ax.plot([], [], 'o', markersize=8, label='moving point')

# critical points
ax.plot([-2, 2], [g(-2), g(2)], 'ro', label='critical points')

ax.legend()

# --------------------------------
# 4. Build one animation path
# --------------------------------
x_full = np.concatenate([x_left, [np.nan], x_right])

def safe_g(vals):
    return np.where(np.isnan(vals), np.nan, g(vals))

def update(frame):
    current_x = x_full[:frame + 1]
    current_y = safe_g(current_x)

    curve_line.set_data(current_x, current_y)

    x0 = x_full[frame]
    if np.isnan(x0):
        point_dot.set_data([], [])
        tangent_line.set_data([], [])
        return curve_line, tangent_line, point_dot

    y0 = g(x0)
    m = g_prime(x0)

    point_dot.set_data([x0], [y0])

    t = np.linspace(x0 - 0.8, x0 + 0.8, 100)
    tangent_y = y0 + m * (t - x0)
    tangent_line.set_data(t, tangent_y)

    return curve_line, tangent_line, point_dot

ani = FuncAnimation(
    fig,
    update,
    frames=len(x_full),
    interval=25,
    blit=True,
    repeat=True
)

# --------------------------------
# 5. Save outputs
# --------------------------------

# GIF
gif_writer = PillowWriter(fps=20, metadata={"artist": "Happy"})
ani.save("g_function_animation.gif", writer=gif_writer)

# MP4
# Requires ffmpeg installed on your system
mp4_writer = FFMpegWriter(fps=20, metadata={"artist": "Happy"}, bitrate=1800)
ani.save("g_function_animation.mp4", writer=mp4_writer)

print("Saved:")
print(" - g_function_animation.gif")
print(" - g_function_animation.mp4")

plt.show()
