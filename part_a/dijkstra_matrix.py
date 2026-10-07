INF = float("inf")


def find_min_vertex(dist, visited):
    """
    Array-based priority queue:
    scan all vertices and return the unvisited vertex
    with the smallest current distance.
    """
    min_distance = INF
    min_vertex = -1

    for v in range(len(dist)):
        if not visited[v] and dist[v] < min_distance:
            min_distance = dist[v]
            min_vertex = v

    return min_vertex


def dijkstra_matrix(graph, source):
    """
    Dijkstra's algorithm using:
    - adjacency matrix for the graph
    - array-based priority queue

    graph[u][v] = edge weight
    graph[u][v] = 0 means no edge
    """

    V = len(graph)

    dist = [INF] * V
    visited = [False] * V

    dist[source] = 0

    for _ in range(V):

        # Pick unvisited vertex with smallest distance
        u = find_min_vertex(dist, visited)

        # No reachable vertices left
        if u == -1:
            break

        visited[u] = True

        # Scan entire row of adjacency matrix
        for v in range(V):

            weight = graph[u][v]

            # Check if an edge exists and v is unvisited
            if weight > 0 and not visited[v]:

                new_distance = dist[u] + weight

                # Relax edge
                if new_distance < dist[v]:
                    dist[v] = new_distance

    return dist


# -------------------------
# Small correctness test
# -------------------------

if __name__ == "__main__":

    graph = [
        [0, 4, 1, 0, 0],
        [4, 0, 2, 1, 0],
        [1, 2, 0, 5, 0],
        [0, 1, 5, 0, 3],
        [0, 0, 0, 3, 0]
    ]

    source = 0

    distances = dijkstra_matrix(graph, source)

    print("Shortest distances from vertex", source)

    for vertex, distance in enumerate(distances):
        print(f"Vertex {vertex}: {distance}")