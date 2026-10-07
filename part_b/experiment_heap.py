import random
import time
import csv
import os
from statistics import mean

from dijkstra_heap import dijkstra_heap


def generate_list_graph(V, density, seed=42):
    """
    Generate a connected, undirected, weighted graph
    using an adjacency list.

    IMPORTANT:
    This generator intentionally mirrors Person 1's
    generate_matrix_graph() logic and random-number call order.

    Therefore, for the same V, density and seed, both implementations
    receive the same graph: same edges and same edge weights.

    density is approximately the desired fraction
    of all possible edges.
    """

    rng = random.Random(seed)

    graph = [[] for _ in range(V)]

    # Maximum possible edges in an undirected graph
    max_edges = V * (V - 1) // 2

    # Desired number of edges
    target_edges = max(V - 1, round(density * max_edges))

    # -------------------------------------------------
    # Step 1: Make graph connected using a spanning tree
    # -------------------------------------------------
    current_edges = 0

    # Only spanning-tree edges need to be remembered here.
    # During Step 2, each possible pair (u, v) is visited only once,
    # so extra edges cannot be duplicated.
    tree_edges = set()

    for v in range(1, V):
        u = rng.randint(0, v - 1)
        weight = rng.randint(1, 100)

        graph[u].append((v, weight))
        graph[v].append((u, weight))

        tree_edges.add((u, v))
        current_edges += 1

    # -------------------------------------------------
    # Step 2: Add extra edges
    # -------------------------------------------------
    remaining_possible = max_edges - current_edges
    edges_needed = target_edges - current_edges

    if remaining_possible > 0:
        probability = edges_needed / remaining_possible
    else:
        probability = 0

    # This loop order matches Person 1's matrix generator exactly.
    for u in range(V):
        for v in range(u + 1, V):

            # Person 1 skips pairs that already belong to the
            # spanning tree. We do the same here.
            if (u, v) not in tree_edges:

                if rng.random() < probability:
                    weight = rng.randint(1, 100)

                    graph[u].append((v, weight))
                    graph[v].append((u, weight))

                    current_edges += 1

    return graph, current_edges


def run_experiment(V, density, repeats=3):
    """
    Generate one graph and run Dijkstra several times.
    Return the average runtime.
    """

    # Same seed rule as Person 1
    seed = 42 + V + int(density * 100)

    graph, E = generate_list_graph(V, density, seed)

    runtimes = []

    for _ in range(repeats):

        start = time.perf_counter()

        dijkstra_heap(graph, 0)

        end = time.perf_counter()

        runtimes.append(end - start)

    average_runtime = mean(runtimes)

    actual_density = E / (V * (V - 1) / 2)

    return E, actual_density, average_runtime


if __name__ == "__main__":

    # Same V values as Person 1
    vertex_sizes = [100, 500, 1000, 2000]

    # Same density values as Person 1
    densities = [
        0.10,
        0.30,
        0.50,
        0.70,
        0.90
    ]

    results = []

    for V in vertex_sizes:

        for density in densities:

            print(
                f"Running V={V}, "
                f"target density={density:.0%}..."
            )

            E, actual_density, runtime = run_experiment(
                V,
                density
            )

            results.append([
                V,
                E,
                actual_density,
                runtime
            ])

            print(
                f"  E={E}, "
                f"density={actual_density:.2%}, "
                f"time={runtime:.6f}s"
            )

    # -------------------------------------------------
    # Save results to CSV
    # -------------------------------------------------

    os.makedirs("results", exist_ok=True)

    filename = "results/heap_results.csv"

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "V",
            "E",
            "density",
            "average_runtime_seconds"
        ])

        writer.writerows(results)

    print()
    print("Experiment complete.")
    print("Results saved to:", filename)
