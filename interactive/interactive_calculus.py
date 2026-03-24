import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

x = sp.symbols("x")


def parse_function(user_input: str):
    expr = sp.sympify(user_input, locals={"x": x})
    if isinstance(expr, (list, tuple, dict, set)):
        raise ValueError("Enter a mathematical expression in x, not a list or brackets.")
    if expr.free_symbols and expr.free_symbols != {x}:
        raise ValueError("Use only the variable x.")
    return sp.simplify(expr)


def safe_numeric_function(expr):
    f = sp.lambdify(x, expr, "numpy")

    def wrapped(vals):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            y = f(vals)
        y = np.array(y, dtype=float)
        y[~np.isfinite(y)] = np.nan
        return y

    return wrapped


def try_numeric_derivative(expr):
    derivative = sp.diff(expr, x)
    try:
        f_prime = safe_numeric_function(derivative)
        test_vals = np.array([-1.0, -0.5, 0.5, 1.2], dtype=float)
        _ = f_prime(test_vals)
        return derivative, f_prime
    except Exception:
        return derivative, None


def build_segments(x_vals, y_vals):
    finite = np.isfinite(y_vals)
    segments = []
    start = None

    for i, ok in enumerate(finite):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > 1:
                segments.append((x_vals[start:i], y_vals[start:i]))
            start = None

    if start is not None and len(x_vals) - start > 1:
        segments.append((x_vals[start:], y_vals[start:]))

    return segments


def choose_point_for_tangent(x_vals, y_vals):
    finite_idx = np.where(np.isfinite(y_vals))[0]
    if len(finite_idx) == 0:
        raise ValueError("No valid points found in the chosen domain.")
    return x_vals[finite_idx[len(finite_idx) // 2]]


def make_static(expr, outdir="output"):
    os.makedirs(outdir, exist_ok=True)

    f = safe_numeric_function(expr)
    derivative, f_prime = try_numeric_derivative(expr)

    x_vals = np.linspace(-6, 6, 1200)
    y_vals = f(x_vals)
    segments_y = build_segments(x_vals, y_vals)

    has_derivative_plot = f_prime is not None

    if has_derivative_plot:
        dy_vals = f_prime(x_vals)
        segments_dy = build_segments(x_vals, dy_vals)
        fig, axes = plt.subplots(2, 1, figsize=(11, 10))
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(figsize=(11, 6))
        ax2 = None

    # Function plot
    for xs, ys in segments_y:
        ax1.plot(xs, ys, label="f(x)" if xs is segments_y[0][0] else None)

    crit_points = []
    if f_prime is not None:
        try:
            roots = sp.solve(sp.Eq(derivative, 0), x)
            for cp in roots:
                cp_eval = sp.N(cp)
                if cp_eval.is_real:
                    cp_float = float(cp_eval)
                    if -6 <= cp_float <= 6:
                        y_cp = f(np.array([cp_float]))[0]
                        if np.isfinite(y_cp):
                            crit_points.append((cp_float, y_cp))
        except Exception:
            pass

    if crit_points:
        ax1.plot(
            [p[0] for p in crit_points],
            [p[1] for p in crit_points],
            "ro",
            label="critical points",
        )

    if f_prime is not None:
        try:
            x0 = choose_point_for_tangent(x_vals, y_vals)
            y0 = f(np.array([x0]))[0]
            m0 = f_prime(np.array([x0]))[0]

            if np.isfinite(y0) and np.isfinite(m0):
                t = np.linspace(x0 - 1.2, x0 + 1.2, 120)
                tangent_y = y0 + m0 * (t - x0)
                ax1.plot(t, tangent_y, "--", label=f"tangent at x={x0:.2f}")
        except Exception:
            pass

    ax1.axvline(0, linestyle=":", linewidth=1)
    ax1.set_title(f"Function: f(x) = {sp.sstr(expr)}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.set_xlim(-6, 6)

    finite_y = y_vals[np.isfinite(y_vals)]
    if len(finite_y) > 0:
        ymin = np.percentile(finite_y, 5)
        ymax = np.percentile(finite_y, 95)
        if ymin == ymax:
            ymax = ymin + 1
        ax1.set_ylim(ymin - 1, min(ymax + 1, 80))

    ax1.grid(True)
    ax1.legend()

    # Derivative plot only if supported
    if ax2 is not None:
        for xs, ys in segments_dy:
            ax2.plot(xs, ys, label="f'(x)" if xs is segments_dy[0][0] else None)

        if crit_points:
            ax2.plot([p[0] for p in crit_points], [0 for _ in crit_points], "ro")

        ax2.axhline(0, linewidth=1)
        ax2.axvline(0, linestyle=":", linewidth=1)
        ax2.set_title(f"Derivative: f'(x) = {sp.sstr(derivative)}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("f'(x)")
        ax2.set_xlim(-6, 6)

        finite_dy = dy_vals[np.isfinite(dy_vals)]
        if len(finite_dy) > 0:
            dymin = np.percentile(finite_dy, 5)
            dymax = np.percentile(finite_dy, 95)
            if dymin == dymax:
                dymax = dymin + 1
            ax2.set_ylim(dymin - 1, dymax + 1)

        ax2.grid(True)
        ax2.legend()
    else:
        ax1.text(
            0.02,
            0.95,
            "Derivative plot not supported for this function.",
            transform=ax1.transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    outpath = os.path.join(outdir, "interactive_static.png")
    plt.savefig(outpath, dpi=300)
    plt.show()
    print(f"Saved static plot to: {outpath}")


def make_animation(expr, outdir="output"):
    os.makedirs(outdir, exist_ok=True)

    f = safe_numeric_function(expr)
    derivative, f_prime = try_numeric_derivative(expr)

    x_vals = np.linspace(-6, 6, 900)
    y_vals = f(x_vals)
    finite_mask = np.isfinite(y_vals)

    valid_x = x_vals[finite_mask]
    valid_y = y_vals[finite_mask]

    if len(valid_x) < 10:
        raise ValueError("Not enough valid points to animate.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Animation of f(x) = {sp.sstr(expr)}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)

    ax.set_xlim(-6, 6)
    ymin = np.percentile(valid_y, 5)
    ymax = np.percentile(valid_y, 95)
    if ymin == ymax:
        ymax = ymin + 1
    ax.set_ylim(ymin - 1, min(ymax + 1, 80))

    ax.plot(valid_x, valid_y, alpha=0.25)

    curve_line, = ax.plot([], [], lw=2, label="f(x)")
    tangent_line, = ax.plot([], [], "--", lw=2, label="tangent")
    point_dot, = ax.plot([], [], "o", markersize=8, label="moving point")
    info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")
    ax.legend()

    def update(frame):
        x_now = valid_x[: frame + 1]
        y_now = valid_y[: frame + 1]
        curve_line.set_data(x_now, y_now)

        x0 = valid_x[frame]
        y0 = valid_y[frame]
        point_dot.set_data([x0], [y0])

        if f_prime is None:
            tangent_line.set_data([], [])
            info_text.set_text(
                f"x = {x0:.3f}\nf(x) = {y0:.3f}\nf'(x) not supported for animation"
            )
        else:
            try:
                m0 = f_prime(np.array([x0]))[0]
                if np.isfinite(m0):
                    t = np.linspace(x0 - 1.0, x0 + 1.0, 120)
                    tangent_y = y0 + m0 * (t - x0)
                    tangent_line.set_data(t, tangent_y)
                    info_text.set_text(
                        f"x = {x0:.3f}\nf(x) = {y0:.3f}\nf'(x) = {m0:.3f}"
                    )
                else:
                    tangent_line.set_data([], [])
                    info_text.set_text(
                        f"x = {x0:.3f}\nf(x) = {y0:.3f}\nf'(x) undefined"
                    )
            except Exception:
                tangent_line.set_data([], [])
                info_text.set_text(
                    f"x = {x0:.3f}\nf(x) = {y0:.3f}\nf'(x) not supported"
                )

        return curve_line, tangent_line, point_dot, info_text

    ani = FuncAnimation(
        fig,
        update,
        frames=len(valid_x),
        interval=20,
        blit=True,
        repeat=True,
    )

    gif_path = os.path.join(outdir, "interactive_animation.gif")
    ani.save(gif_path, writer=PillowWriter(fps=20))
    print(f"Saved animation to: {gif_path}")
    plt.show()


def main():
    print("\nInteractive Calculus Visualizer")
    print("Examples:")
    print("  x**2 + 16/x**2")
    print("  sin(x) + x**2")
    print("  x**3 - 3*x")
    print("  floor(x)")
    print("  Abs(x)\n")

    user_input = input("Enter a function in x: ").strip()
    if not user_input:
        print("No function entered.")
        return

    try:
        expr = parse_function(user_input)
    except Exception as e:
        print(f"Could not parse the function: {e}")
        return

    derivative = sp.diff(expr, x)

    print(f"\nParsed function: f(x) = {sp.sstr(expr)}")
    print(f"Derivative: f'(x) = {sp.sstr(derivative)}\n")

    if expr.has(sp.floor):
        print("Warning: floor(x) is non-smooth. Static plotting is more meaningful than tangent animation.\n")

    mode = input("Choose mode: [1] static  [2] animation  [3] both : ").strip()

    try:
        if mode == "1":
            make_static(expr)
        elif mode == "2":
            make_animation(expr)
        elif mode == "3":
            make_static(expr)
            make_animation(expr)
        else:
            print("Invalid choice. Use 1, 2, or 3.")
    except Exception as e:
        print(f"Error while generating output: {e}")


if __name__ == "__main__":
    main()
