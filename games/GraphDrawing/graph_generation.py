import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import random

# ---------- Random connected graph generator ----------
import random
from typing import List, Tuple, Dict

def generate_connected_random_graph(n: int, s: float = 0.3):
    """
    Generate:
        - n nodes (not n*n)
        - placed at n random positions of an n×n grid
        - a connected random graph with density controlled by s ∈ [0,1]

    Returns:
        n (grid size)
        pos_map: dict node_id -> (i,j)
        edges: list of (u,v)
    """

    # Step 1 — pick n random positions from grid
    all_positions = [(i, j) for i in range(n) for j in range(n)]
    chosen = random.sample(all_positions, n)

    node_ids = [i*n + j + 1 for (i, j) in chosen]  # flattened IDs
    pos_map = {node_ids[k]: chosen[k] for k in range(n)}

    # Step 2 — build a connected random graph on these n nodes
    edges = []
    remaining = node_ids[:]
    random.shuffle(remaining)

    # random spanning tree first
    for i in range(1, n):
        u = remaining[i]
        v = random.choice(remaining[:i])
        edges.append((u, v))

    # Step 3 — add more edges according to sparsity
    present = set((min(a,b), max(a,b)) for a,b in edges)

    # possible remaining pairs
    candidates = [
        (min(a, b), max(a, b))
        for i, a in enumerate(node_ids)
        for b in node_ids[i+1:]
        if (min(a,b), max(a,b)) not in present
    ]

    # s controls number of extra edges
    s = max(0.0, min(1.0, s))  # clamp
    max_extra = len(candidates)
    desired_extra = int(s * max_extra)

    extra_edges = random.sample(candidates, desired_extra)
    edges.extend(extra_edges)

    return n, pos_map, edges
