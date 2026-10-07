import heapq

INF = float("inf")


def dijkstra_heap(graph, source):
    """
    Dijkstra's algorithm using:
    - adjacency list for the graph
    - min-heap priority queue

    graph[u] contains (v, weight) pairs.
    """

    V = len(graph)

    dist = [INF] * V
    dist[source] = 0

    # Heap entries are (distance, vertex)
    min_heap = [(0, source)]

    while min_heap:

        # Extract vertex with the smallest tentative distance
        current_distance, u = heapq.heappop(min_heap)

        # Ignore stale heap entries
        if current_distance != dist[u]:
            continue

        # Only scan the actual neighbours of u
        for v, weight in graph[u]:

            new_distance = current_distance + weight

            # Relax edge
            if new_distance < dist[v]:
                dist[v] = new_distance
                heapq.heappush(
                    min_heap,
                    (new_distance, v)
                )

    return dist


# -------------------------
# Small correctness test
# -------------------------

if __name__ == "__main__":

    # Same graph as the correctness test in dijkstra_matrix.py,
    # represented as an adjacency list.
    graph = [
        [(1, 4), (2, 1)],
        [(0, 4), (2, 2), (3, 1)],
        [(0, 1), (1, 2), (3, 5)],
        [(1, 1), (2, 5), (4, 3)],
        [(3, 3)]
    ]

    source = 0

    distances = dijkstra_heap(graph, source)

    print("Shortest distances from vertex", source)

    for vertex, distance in enumerate(distances):
        print(f"Vertex {vertex}: {distance}")
