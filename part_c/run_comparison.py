"""
Part (c) experiments: adjacency matrix + array PQ  vs  adjacency list + min-heap.

Usage (from anywhere):
    python part_c/run_comparison.py grid         # standard V x density grid
    python part_c/run_comparison.py density      # density sweep (finds the crossover)
    python part_c/run_comparison.py sparse       # sparse graphs, E = 5V, growing V
    python part_c/run_comparison.py adversarial  # worst case for the heap (dense)
    python part_c/run_comparison.py complete     # complete graph, random weights (control for adversarial)
    python part_c/run_comparison.py all
    add --quick for a small smoke test (writes to results/quick_*)

Every row is only written if the two implementations (and SciPy's
reference Dijkstra) return identical distances.
"""

import csv
import gc
import os
import sys
import time

from compare_common import (
    RESULTS_DIR,
    build_adversarial_complete,
    build_pair,
    matrix_to_list,
    measure,
    same_graph_as_part_b,
    standard_seed,
)

VERTEX_SIZES = [100, 500, 1000, 2000, 5000]      # agreed standard
DENSITIES = [0.10, 0.30, 0.50, 0.70, 0.90]       # agreed standard
REPEATS = 3

COLUMNS = [
    "V", "E", "actual_density",
    "matrix_mean_s", "matrix_std_s", "heap_mean_s", "heap_std_s",
    "speedup_matrix_over_heap",
    "heap_pushes", "heap_pops", "heap_stale_pops", "edge_scans",
    "matrix_steps", "distances_match",
]


class CsvLog:
    """Append rows as soon as they exist so a long run is never lost."""

    def __init__(self, name, extra_cols):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.path = os.path.join(RESULTS_DIR, name)
        self.cols = extra_cols + COLUMNS
        self.file = open(self.path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.cols, extrasaction="ignore")
        self.writer.writeheader()
        self.file.flush()

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def log_row(tag, row):
    print(
        f"  {tag}: E={row['E']:,}  matrix={row['matrix_mean_s']:.4f}s  "
        f"heap={row['heap_mean_s']:.4f}s  matrix/heap={row['speedup_matrix_over_heap']:.2f}x  "
        f"pushes/E={row.get('heap_pushes', 0) / max(row['E'], 1):.3f}  ok={row['distances_match']}",
        flush=True,
    )
    if not row["distances_match"]:
        raise SystemExit("DISTANCE MISMATCH between implementations - stop and investigate.")


# ------------------------------------------------------------------

def run_grid(quick=False):
    """E1: the agreed standard grid, V x density, both implementations on the SAME graph."""
    sizes = [100, 300] if quick else VERTEX_SIZES
    dens = [0.10, 0.90] if quick else DENSITIES
    log = CsvLog("quick_grid.csv" if quick else "comparison_grid.csv", ["target_density"])
    for V in sizes:
        for d in dens:
            seed = standard_seed(V, d)
            print(f"[grid] V={V} density={d:.0%} seed={seed}", flush=True)
            t0 = time.perf_counter()
            matrix, adj, E = build_pair(V, d, seed)
            if V <= 500:
                assert same_graph_as_part_b(matrix, V, d, seed), "graph differs from part (b)'s generator"
            row = measure(matrix, adj, E, repeats=REPEATS)
            row["target_density"] = d
            log_row("done", row)
            log.write(row)
            del matrix, adj
            gc.collect()
            print(f"  (config took {time.perf_counter() - t0:.1f}s)", flush=True)
    log.close()


def run_density_sweep(quick=False):
    """E2: fixed V, density from the spanning-tree minimum up to the complete graph."""
    sizes = [300] if quick else [1000, 2000]
    dens = [0.01, 0.50, 1.00] if quick else [
        0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00
    ]
    log = CsvLog("quick_density.csv" if quick else "density_sweep.csv", ["target_density"])
    for V in sizes:
        for d in dens:
            seed = 42 + V + round(d * 1000)       # distinct seed per fractional density
            print(f"[density] V={V} density={d:.1%} seed={seed}", flush=True)
            matrix, adj, E = build_pair(V, d, seed)
            row = measure(matrix, adj, E, repeats=REPEATS)
            row["target_density"] = d
            log_row("done", row)
            log.write(row)
            del matrix, adj
            gc.collect()
    log.close()


def run_sparse(quick=False, avg_degree=10):
    """E3: sparse graphs with E = (avg_degree/2) * V, so E grows only linearly with V."""
    sizes = [100, 300] if quick else [100, 200, 500, 1000, 2000, 3000, 5000]
    log = CsvLog("quick_sparse.csv" if quick else "sparse_scaling.csv", ["target_density", "avg_degree"])
    for V in sizes:
        d = avg_degree / (V - 1)                  # => E = avg_degree * V / 2
        seed = 42 + V + avg_degree
        print(f"[sparse] V={V} avg_degree={avg_degree} seed={seed}", flush=True)
        matrix, adj, E = build_pair(V, d, seed)
        row = measure(matrix, adj, E, repeats=REPEATS)
        row["target_density"] = d
        row["avg_degree"] = avg_degree
        log_row("done", row)
        log.write(row)
        del matrix, adj
        gc.collect()
    log.close()


def run_adversarial(quick=False):
    """E4: dense graph built so EVERY edge triggers a heap push (worst case for part b)."""
    sizes = [100, 200] if quick else [100, 200, 500, 1000, 1500, 2000, 3000]
    log = CsvLog("quick_adversarial.csv" if quick else "adversarial_dense.csv", ["target_density"])
    for V in sizes:
        print(f"[adversarial] V={V}", flush=True)
        matrix, E = build_adversarial_complete(V)
        adj = matrix_to_list(matrix)
        row = measure(matrix, adj, E, repeats=REPEATS)
        row["target_density"] = 1.0
        # Extra sanity check specific to this construction: d(i) == i
        from dijkstra_heap import dijkstra_heap
        assert dijkstra_heap(adj, 0) == list(range(V)), "adversarial graph did not behave as designed"
        log_row("done", row)
        log.write(row)
        del matrix, adj
        gc.collect()
    log.close()


def run_complete_random(quick=False):
    """E5: complete graph with ordinary random weights 1-100 - the control for E4
    (same |V| sizes, same density = 100%, only the weight pattern differs)."""
    sizes = [100, 200] if quick else [100, 200, 500, 1000, 1500, 2000, 3000]
    log = CsvLog("quick_complete_random.csv" if quick else "complete_random.csv", ["target_density"])
    for V in sizes:
        seed = 42 + V + 100
        print(f"[complete-random] V={V} seed={seed}", flush=True)
        matrix, adj, E = build_pair(V, 1.0, seed)
        row = measure(matrix, adj, E, repeats=REPEATS)
        row["target_density"] = 1.0
        log_row("done", row)
        log.write(row)
        del matrix, adj
        gc.collect()
    log.close()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv
    which = args[0] if args else "all"

    runners = {
        "grid": run_grid,
        "density": run_density_sweep,
        "sparse": run_sparse,
        "adversarial": run_adversarial,
        "complete": run_complete_random,
    }
    todo = list(runners) if which == "all" else [which]
    for name in todo:
        runners[name](quick=quick)
    print("\nAll requested experiments finished.", flush=True)
