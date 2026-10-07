"""
Part (c) graphs. Reads results/*.csv, writes results/*.png (+ a fitted-exponent table).

    python part_c/plot_comparison.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# Two-series palette validated for colour-blind separation (blue / orange).
MATRIX_C = "#2a78d6"   # (a) adjacency matrix + array PQ
HEAP_C = "#eb6834"     # (b) adjacency list + min-heap
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 130,
})


def load(name):
    with open(os.path.join(RES, name)) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k != "distances_match":
                r[k] = float(v)
    return rows


def col(rows, key):
    return np.array([r[key] for r in rows])


def save(fig, name):
    path = os.path.join(RES, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def loglog_slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]


# ------------------------------------------------------------------
# 1. Standard grid: runtime vs V for each density, both implementations
# ------------------------------------------------------------------
def fig_grid_vs_vertices(grid):
    densities = sorted({round(r["target_density"], 2) for r in grid})
    shades = plt.cm.Greys(np.linspace(0.35, 0.95, len(densities)))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for d, shade in zip(densities, shades):
        sub = [r for r in grid if round(r["target_density"], 2) == d]
        V = col(sub, "V")
        ax.plot(V, col(sub, "matrix_mean_s"), "-o", color=MATRIX_C, alpha=0.35 + 0.65 * d, lw=1.8, ms=4)
        ax.plot(V, col(sub, "heap_mean_s"), "-s", color=HEAP_C, alpha=0.35 + 0.65 * d, lw=1.8, ms=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of vertices |V|")
    ax.set_ylabel("Mean runtime (seconds)")
    ax.set_title("Runtime vs |V| across densities 10%–90%")
    ax.plot([], [], "-o", color=MATRIX_C, label="(a) matrix + array PQ")
    ax.plot([], [], "-s", color=HEAP_C, label="(b) list + min-heap")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.04, "darker = denser graph (10% → 90%)", transform=ax.transAxes,
            ha="right", color=MUTED, fontsize=9)
    save(fig, "compare_runtime_vs_vertices.png")


# ------------------------------------------------------------------
# 2. Standard grid: runtime vs E (all 25 configurations)
# ------------------------------------------------------------------
def fig_grid_vs_edges(grid):
    E = col(grid, "E")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(E, col(grid, "matrix_mean_s"), color=MATRIX_C, s=34, label="(a) matrix + array PQ", zorder=3)
    ax.scatter(E, col(grid, "heap_mean_s"), color=HEAP_C, marker="s", s=34, label="(b) list + min-heap", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of edges |E|")
    ax.set_ylabel("Mean runtime (seconds)")
    ax.set_title("Runtime vs |E|: heap tracks E, matrix tracks |V|²")
    ax.legend(loc="upper left")
    save(fig, "compare_runtime_vs_edges.png")


# ------------------------------------------------------------------
# 3. Speed-up heatmap (matrix time / heap time) on the standard grid
# ------------------------------------------------------------------
def fig_speedup_heatmap(grid):
    sizes = sorted({int(r["V"]) for r in grid})
    dens = sorted({round(r["target_density"], 2) for r in grid})
    M = np.zeros((len(sizes), len(dens)))
    for r in grid:
        i = sizes.index(int(r["V"])); j = dens.index(round(r["target_density"], 2))
        M[i, j] = r["speedup_matrix_over_heap"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(dens))); ax.set_xticklabels([f"{int(d*100)}%" for d in dens])
    ax.set_yticks(range(len(sizes))); ax.set_yticklabels([f"{v:,}" for v in sizes])
    ax.set_xlabel("Graph density"); ax.set_ylabel("|V|")
    ax.set_title("How many times faster is (b) than (a)?")
    for i in range(len(sizes)):
        for j in range(len(dens)):
            ax.text(j, i, f"{M[i, j]:.1f}×", ha="center", va="center",
                    color="white" if M[i, j] > M.max() * 0.55 else INK, fontsize=10)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    save(fig, "compare_speedup_heatmap.png")


# ------------------------------------------------------------------
# 4. Density sweep at fixed V: where does the gap close?
# ------------------------------------------------------------------
def fig_density_sweep(sweep):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    for V, ls in ((1000, "-"), (2000, "--")):
        sub = [r for r in sweep if int(r["V"]) == V]
        x = col(sub, "actual_density") * 100
        axes[0].plot(x, col(sub, "matrix_mean_s"), ls, marker="o", color=MATRIX_C, ms=4, lw=1.8)
        axes[0].plot(x, col(sub, "heap_mean_s"), ls, marker="s", color=HEAP_C, ms=4, lw=1.8)
        axes[1].plot(x, col(sub, "speedup_matrix_over_heap"), ls, marker="o", color=INK, ms=4, lw=1.8,
                     label=f"|V| = {V:,}")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Graph density (% of possible edges)")
    axes[0].set_ylabel("Mean runtime (seconds)")
    axes[0].set_title("Matrix is flat in density; heap grows with |E|")
    axes[0].plot([], [], "-o", color=MATRIX_C, label="(a) matrix + array PQ")
    axes[0].plot([], [], "-s", color=HEAP_C, label="(b) list + min-heap")
    axes[0].plot([], [], "-", color=MUTED, label="solid |V|=1000, dashed |V|=2000")
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].axhline(1, color=MUTED, lw=1.2)
    axes[1].text(0.2, 1.08, "break-even (1×)", color=MUTED, fontsize=9)
    axes[1].set_xlabel("Graph density (% of possible edges)")
    axes[1].set_ylabel("Runtime(a) ÷ Runtime(b)")
    axes[1].set_title("Heap's lead shrinks with density but stays above 1×")
    axes[1].legend(loc="upper right")
    save(fig, "compare_density_sweep.png")


# ------------------------------------------------------------------
# 5. Sparse scaling (E = 5V): quadratic vs near-linear growth
# ------------------------------------------------------------------
def fig_sparse(sparse):
    V = col(sparse, "V")
    tm, th = col(sparse, "matrix_mean_s"), col(sparse, "heap_mean_s")
    sm, sh = loglog_slope(V, tm), loglog_slope(V, th)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(V, tm, "-o", color=MATRIX_C, lw=1.8, label=f"(a) matrix + array PQ  (fitted slope {sm:.2f})")
    ax.plot(V, th, "-s", color=HEAP_C, lw=1.8, label=f"(b) list + min-heap  (fitted slope {sh:.2f})")
    ref = np.array([V[0], V[-1]], dtype=float)
    ax.plot(ref, tm[0] * (ref / V[0]) ** 2, ":", color=MATRIX_C, lw=1.4, label="O(|V|²) reference")
    ax.plot(ref, th[0] * (ref / V[0]) ** 1, ":", color=HEAP_C, lw=1.4, label="O(|V|) reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of vertices |V|  (average degree fixed at 10, so |E| = 5|V|)")
    ax.set_ylabel("Mean runtime (seconds)")
    ax.set_title("Sparse graphs: the gap widens as the graph grows")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "compare_sparse_scaling.png")
    return sm, sh


# ------------------------------------------------------------------
# 6. Dense worst case vs dense random weights
# ------------------------------------------------------------------
def fig_dense_cases(adv, ctrl):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True, sharex=True)
    for ax, data, title in (
        (axes[0], ctrl, "Complete graph, random weights 1–100"),
        (axes[1], adv, "Complete graph, worst-case weights for the heap"),
    ):
        V = col(data, "V")
        ax.plot(V, col(data, "matrix_mean_s"), "-o", color=MATRIX_C, lw=1.8, label="(a) matrix + array PQ")
        ax.plot(V, col(data, "heap_mean_s"), "-s", color=HEAP_C, lw=1.8, label="(b) list + min-heap")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Number of vertices |V|  (|E| = |V|(|V|−1)/2)")
        ax.set_title(title)
        r = col(data, "speedup_matrix_over_heap")[-1]
        msg = f"at |V| = {int(V[-1]):,}:\n(b) is {r:.1f}× faster" if r > 1 else f"at |V| = {int(V[-1]):,}:\n(a) is {1 / r:.0f}× faster"
        ax.text(0.04, 0.60, msg, transform=ax.transAxes, fontsize=10, color=INK)
    axes[0].set_ylabel("Mean runtime (seconds)")
    axes[0].legend(loc="upper left", fontsize=9)
    save(fig, "compare_dense_cases.png")


# ------------------------------------------------------------------
# 7. Why: fraction of edges that actually cause a heap push
# ------------------------------------------------------------------
def fig_push_fraction(grid, adv):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for V, marker in ((500, "o"), (1000, "s"), (2000, "^"), (5000, "D")):
        sub = [r for r in grid if int(r["V"]) == V]
        ax.plot(col(sub, "actual_density") * 100, col(sub, "heap_pushes") / col(sub, "E") * 100,
                "-" + marker, color=HEAP_C, alpha=0.35 + 0.65 * (V / 5000), ms=5, lw=1.5, label=f"random, |V|={V:,}")
    ax.axhline(100, color=INK, lw=1.4)
    ax.text(11, 62, "worst-case construction: every edge pushes (100%)", fontsize=9, color=INK, va="top")
    ax.set_yscale("log")
    ax.set_xlabel("Graph density (%)")
    ax.set_ylabel("Edges causing a heap push (% of |E|)")
    ax.set_title("On random weights, almost no edge ever pushes")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.80), fontsize=9)
    save(fig, "compare_push_fraction.png")


def write_exponent_table(grid, sparse):
    lines = ["| Setting | (a) matrix slope | (b) heap slope |", "|---|---|---|"]
    for d in (0.10, 0.50, 0.90):
        sub = [r for r in grid if round(r["target_density"], 2) == d and r["V"] >= 500]
        V = col(sub, "V")
        lines.append(f"| density {int(d*100)}%, |V| 500→5000 | "
                     f"{loglog_slope(V, col(sub, 'matrix_mean_s')):.2f} | "
                     f"{loglog_slope(V, col(sub, 'heap_mean_s')):.2f} |")
    V = col(sparse, "V")
    sub_idx = V >= 500
    lines.append(f"| sparse, |E|=5|V|, |V| 500→5000 | "
                 f"{loglog_slope(V[sub_idx], col(sparse, 'matrix_mean_s')[sub_idx]):.2f} | "
                 f"{loglog_slope(V[sub_idx], col(sparse, 'heap_mean_s')[sub_idx]):.2f} |")
    path = os.path.join(RES, "fitted_exponents.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)
    print("\n".join(lines))


if __name__ == "__main__":
    grid = load("comparison_grid.csv")
    sweep = load("density_sweep.csv")
    sparse = load("sparse_scaling.csv")
    adv = load("adversarial_dense.csv")
    ctrl = load("complete_random.csv")

    fig_grid_vs_vertices(grid)
    fig_grid_vs_edges(grid)
    fig_speedup_heatmap(grid)
    fig_density_sweep(sweep)
    sm, sh = fig_sparse(sparse)
    fig_dense_cases(adv, ctrl)
    fig_push_fraction(grid, adv)
    write_exponent_table(grid, sparse)
