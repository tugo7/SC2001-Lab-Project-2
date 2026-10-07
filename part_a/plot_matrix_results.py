import csv
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# Read CSV
results = []

with open("results/matrix_results.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        results.append({
            "V": int(row["V"]),
            "E": int(row["E"]),
            "density": float(row["density"]),
            "runtime": float(row["average_runtime_seconds"])
        })


os.makedirs("results", exist_ok=True)

print("Loaded", len(results), "experiment results.")


# -------------------------------------------------
# GRAPH 1: Runtime vs V
# -------------------------------------------------

target_densities = [0.10, 0.30, 0.50, 0.70, 0.90]

plt.figure(figsize=(8, 5))

for target in target_densities:

    matching = [
        row for row in results
        if abs(row["density"] - target) < 0.03
    ]

    matching.sort(key=lambda row: row["V"])

    V_values = [row["V"] for row in matching]
    runtimes = [row["runtime"] for row in matching]

    if V_values:
        plt.plot(
            V_values,
            runtimes,
            marker="o",
            label=f"{int(target * 100)}% density"
        )


plt.xlabel("Number of Vertices |V|")
plt.ylabel("Average Runtime (seconds)")
plt.title("Runtime vs Number of Vertices")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/runtime_vs_vertices.png",
    dpi=300
)

plt.close()

print("Created runtime_vs_vertices.png")


# -------------------------------------------------
# GRAPH 2: Runtime vs E
# -------------------------------------------------

largest_V = max(row["V"] for row in results)

fixed_results = [
    row for row in results
    if row["V"] == largest_V
]

fixed_results.sort(key=lambda row: row["E"])

E_values = [row["E"] for row in fixed_results]
runtimes = [row["runtime"] for row in fixed_results]


plt.figure(figsize=(8, 5))

plt.plot(
    E_values,
    runtimes,
    marker="o"
)

plt.xlabel("Number of Edges |E|")
plt.ylabel("Average Runtime (seconds)")
plt.title(
    f"Runtime vs Number of Edges (|V| = {largest_V})"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/runtime_vs_edges.png",
    dpi=300
)

plt.close()

print("Created runtime_vs_edges.png")

print("DONE")