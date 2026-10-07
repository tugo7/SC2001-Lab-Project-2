"""
Part (c) - shared helpers for comparing Dijkstra (a) vs (b).

Design rules (from the team's agreed standardisation):
  * Python, undirected, connected, weighted graphs, edge weights 1-100
  * source vertex = 0
  * time.perf_counter() for timing
  * the SAME graph is generated once, then converted to the adjacency
    matrix and the adjacency list, so both implementations see
    exactly the same input
  * both implementations must return identical distances before any
    timing result is trusted

Parts (a) and (b) are IMPORTED as-is (read-only). Nothing in part_a/ or
part_b/ is modified or copied.
"""

import gc
import os
import sys
import time
from statistics import mean, pstdev

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "part_a"))
sys.path.insert(0, os.path.join(ROOT, "part_b"))

from dijkstra_matrix import dijkstra_matrix            # Part (a), unmodified
from dijkstra_heap import dijkstra_heap                # Part (b), unmodified
from experiment_matrix import generate_matrix_graph    # Part (a) graph generator
from experiment_heap import generate_list_graph        # Part (b) generator (cross-check only)

import heapq

INF = float("inf")
SOURCE = 0
RESULTS_DIR = os.path.join(HERE, "results")


# ------------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------------

def standard_seed(V, density):
    """Same per-configuration seed rule used in parts (a) and (b)."""
    return 42 + V + int(density * 100)


def matrix_to_list(matrix):
    """
    Convert the adjacency matrix into an array of adjacency lists.
    graph[u] = [(v, weight), ...]  (0 in the matrix means 'no edge').
    This is the exact same graph, just stored differently.
    """
    return [
        [(v, w) for v, w in enumerate(row) if w]
        for row in matrix
    ]


def build_pair(V, density, seed):
    """Generate one graph, return (matrix, adjacency_list, E)."""
    matrix, E = generate_matrix_graph(V, density, seed)
    return matrix, matrix_to_list(matrix), E


def build_adversarial_complete(V):
    """
    Complete graph with w(i, j) = 2j - 1 - 2i for i < j.

    Shortest distances from vertex 0 are d(i) = i (the path 0-1-2-...
    uses weight-1 edges). Dijkstra settles vertices in order 0,1,2,...
    and settling vertex i improves the tentative distance of EVERY
    later vertex j by exactly 1, so every one of the E edges causes a
    successful relaxation. That is the worst case for a heap-based
    queue (E pushes), used to show when the matrix version can win.

    NOTE: weights go up to ~2V, i.e. outside the 1-100 range of the main
    experiments. It is a deliberate worst-case construction, reported
    separately from the standard experiments.
    """
    matrix = [[0] * V for _ in range(V)]
    for i in range(V):
        for j in range(i + 1, V):
            w = 2 * j - 1 - 2 * i
            matrix[i][j] = w
            matrix[j][i] = w
    return matrix, V * (V - 1) // 2


# ------------------------------------------------------------------
# Timing
# ------------------------------------------------------------------

def timed_runs(fn, graph, source=SOURCE, repeats=3):
    """
    Run fn(graph, source) `repeats` times with time.perf_counter().
    The garbage collector is paused during the timed region (standard
    benchmarking practice, e.g. what timeit does) so that collection
    pauses over millions of tuples do not add noise. Applied equally to
    both implementations. Returns (list_of_times, distances_of_run_0).
    """
    times = []
    result = None
    gc.collect()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            out = fn(graph, source)
            end = time.perf_counter()
            times.append(end - start)
            if result is None:
                result = out
    finally:
        gc.enable()
    return times, result


# ------------------------------------------------------------------
# Operation counting (explains WHY the timings look the way they do)
# ------------------------------------------------------------------

def count_heap_ops(adj, source=SOURCE):
    """
    Same algorithm as dijkstra_heap (part b), but counting work done.
    Used only for operation counts; its distances are checked against
    dijkstra_heap's in the experiments.

      pushes      = successful relaxations (each pushes onto the heap)
      pops        = heap extractions (pushes + the initial entry)
      stale_pops  = extractions of outdated entries (skipped)
      edge_scans  = adjacency entries examined (= 2E for a full run)
    """
    V = len(adj)
    dist = [INF] * V
    dist[source] = 0
    heap = [(0, source)]
    pops = stale = pushes = scans = 0
    while heap:
        d, u = heapq.heappop(heap)
        pops += 1
        if d != dist[u]:
            stale += 1
            continue
        for v, w in adj[u]:
            scans += 1
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
                pushes += 1
    return dist, {
        "heap_pushes": pushes,
        "heap_pops": pops,
        "heap_stale_pops": stale,
        "edge_scans": scans,
    }


def matrix_steps(V):
    """
    Basic steps of part (a) on a connected graph: for each of the V
    iterations, find_min_vertex scans V entries and the relaxation loop
    scans a full matrix row of V entries -> exactly 2 * V^2, independent
    of the number of edges.
    """
    return 2 * V * V


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def scipy_distances(matrix, source=SOURCE):
    """Independent reference (compiled Dijkstra from SciPy)."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    arr = np.array(matrix, dtype=np.int64)
    return dijkstra(csr_matrix(arr), directed=False, indices=source).tolist()


def distances_agree(dist_matrix, dist_heap, dist_ref=None):
    """True if (a) == (b), all vertices reachable, and (optionally) == reference."""
    if dist_matrix != dist_heap:
        return False
    if max(dist_matrix) == INF:
        return False
    if dist_ref is not None:
        if [float(x) for x in dist_matrix] != [float(x) for x in dist_ref]:
            return False
    return True


def same_graph_as_part_b(matrix, V, density, seed):
    """
    Cross-check: the adjacency list we derived from part (a)'s matrix
    must equal what part (b)'s own generator produces for the same
    (V, density, seed) -- confirms both teammates really used one graph.
    """
    derived = matrix_to_list(matrix)
    theirs, _ = generate_list_graph(V, density, seed)
    return all(sorted(a) == sorted(b) for a, b in zip(derived, theirs))


# ------------------------------------------------------------------
# One measurement = both implementations on the same graph
# ------------------------------------------------------------------

def measure(matrix, adj, E, repeats=3, reference=True, count_ops=True):
    V = len(matrix)

    t_matrix, d_matrix = timed_runs(dijkstra_matrix, matrix, repeats=repeats)
    t_heap, d_heap = timed_runs(dijkstra_heap, adj, repeats=repeats)

    ref = scipy_distances(matrix) if reference else None
    ok = distances_agree(d_matrix, d_heap, ref)

    row = {
        "V": V,
        "E": E,
        "actual_density": E / (V * (V - 1) / 2),
        "matrix_mean_s": mean(t_matrix),
        "matrix_std_s": pstdev(t_matrix),
        "heap_mean_s": mean(t_heap),
        "heap_std_s": pstdev(t_heap),
        "speedup_matrix_over_heap": mean(t_matrix) / mean(t_heap),
        "matrix_steps": matrix_steps(V),
        "distances_match": ok,
    }

    if count_ops:
        d_count, ops = count_heap_ops(adj)
        if d_count != d_heap:
            row["distances_match"] = False
        row.update(ops)

    return row
