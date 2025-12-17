from typing import Dict, Tuple, List, Set, Iterable, Optional
import numpy as np

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class GraphDrawing:
    """
    A unified structure that stores a graph + its drawing on an n×n grid,
    together with cached information such as crossings, adjacency, local
    crossing contributions, and tensor representation.
    """

    def __init__(
        self,
        n: int,
        pos: Dict[int, Tuple[int, int]],
        edges: Iterable[Tuple[int, int]],
        build_tensor: bool = True,
    ):
        """Initialize graph-drawing structure."""
        self.n = n  # grid size
        self.pos = dict(pos)  # node_id -> (i, j)
        self.node_ids = list(self.pos.keys())  # FIX 1
        self.edges = self._canonical_edges(edges)
        self.adj = self._build_adj()  # node -> neighbors

        # Crossing caches
        self.crossing_pairs: set[frozenset] = set()
        self.crossings = self.count_crossings_full()

        # Tensor representation
        self.tensor: Optional[np.ndarray] = None
        if build_tensor:
            self.tensor = self.build_tensor()

    # -----------------------------------------------------------
    # Internal consistency helpers
    # -----------------------------------------------------------

    def _canonical_edges(
        self, edges: Iterable[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Return canonical undirected edges u < v with duplicates removed.
        This ensures that the internal representation is stable.
        """
        canon: Set[Tuple[int, int]] = set()
        for u, v in edges:
            if u == v:
                continue  # ignore self-loops
            u2, v2 = (u, v) if u < v else (v, u)
            canon.add((u2, v2))
        return sorted(canon)

    def _build_adj(self) -> Dict[int, Set[int]]:
        """
        Build adjacency list from the current set of canonical edges.
        Returns dict: node -> set(neighbors).
        """
        adj: Dict[int, Set[int]] = {nid: set() for nid in self.node_ids}

        for u, v in self.edges:  # edges are already canonical (u < v)
            adj[u].add(v)
            adj[v].add(u)

        return adj

    # -----------------------------------------------------------
    # Tensor builder
    # -----------------------------------------------------------

    def build_tensor(self) -> np.ndarray:
        """
        Build tensor T of shape (C, n, n) with your canonical encoding.

        Channels:
        - channel 0: node existence: T[0, i, j] = 1 if some node occupies (i,j)
        - for each canonical undirected edge (u, v) with u < v:
              let (i_u, j_u) be u's grid position
              then T[v, i_u, j_u] = 1

        Notes:
        - C = 1 + n*n (exactly like your format)
        - v is used as the channel index to encode the edge.
        """
        C = 1 + self.n * self.n
        T = np.zeros((C, self.n, self.n), dtype=int)

        # Mark node existence on channel 0
        for nid, (i, j) in self.pos.items():
            T[0, i, j] = 1

        # Encode edges in canonical form
        for u, v in self.edges:
            # u < v guaranteed by canonical form
            ui, uj = self.pos[u]
            T[v, ui, uj] = 1

        return T

    # -----------------------------------------------------------
    # Geometry + crossings
    # -----------------------------------------------------------

    def count_crossings_full(self) -> int:
        """
        Recompute ALL crossings from scratch.
        COMPLEXITY: O(E*E), we iterate twice over all the edges
        """
        self.crossing_pairs.clear()
        edges = self.edges
        pos = self.pos

        m = len(edges)

        for i in range(m):
            u1, v1 = edges[i]
            p1, p2 = pos[u1][::-1], pos[v1][::-1]

            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 in (u2, v2) or v1 in (u2, v2):
                    continue  # share endpoint → not valid crossing

                q1, q2 = pos[u2][::-1], pos[v2][::-1]

                if self._segments_intersect(p1, p2, q1, q2):
                    pair = frozenset({(u1, v1), (u2, v2)})
                    self.crossing_pairs.add(pair)

        self.crossings = len(self.crossing_pairs)
        return self.crossings

    def update_crossings_incremental(self, moved_node: int) -> int:
        """
        Update crossings only for edges incident to moved_node.
        Uses crossing_pairs set to maintain exact consistency.
        COMPLEXITY: O(D*E), we iterate over all the affected edges, then over all the edges
        """

        if self.crossings is None:
            return self.count_crossings_full()

        edges = self.edges
        pos = self.pos
        crossing_pairs = self.crossing_pairs

        # 1. Identify affected edges
        affected = [(u, v) for (u, v) in edges if moved_node in (u, v)]

        # 2. Remove all existing crossing pairs involving any affected edge
        to_remove = []
        for pair in crossing_pairs:
            # pair contains two edges as tuples
            if any(e in pair for e in affected):
                to_remove.append(pair)

        for p in to_remove:
            crossing_pairs.remove(p)

        # 3. Recompute crossings between affected edges and ALL other edges
        for u1, v1 in affected:
            p1, p2 = pos[u1][::-1], pos[v1][::-1]

            for u2, v2 in edges:
                if (u1, v1) == (u2, v2):
                    continue
                if u1 in (u2, v2) or v1 in (u2, v2):
                    continue  # share node → not a crossing
                if (u2, v2) in affected:
                    continue  # skip affected–affected (will be counted later if needed)

                q1, q2 = pos[u2][::-1], pos[v2][::-1]

                if self._segments_intersect(p1, p2, q1, q2):
                    crossing_pairs.add(frozenset({(u1, v1), (u2, v2)}))

        # 4. New total
        self.crossings = len(crossing_pairs)
        return self.crossings

    # -----------------------------------------------------------
    # Editing the drawing
    # -----------------------------------------------------------

    def move_node_to(self, node: int, new_pos: Tuple[int, int]):
        """Move a node to a new grid coordinate."""

        old_pos = self.pos[node]
        self.pos[node] = new_pos

        # Update tensor
        if self.tensor is not None:
            i_old, j_old = old_pos
            i_new, j_new = new_pos

            # Channel 0: node existence
            self.tensor[0, i_old, j_old] = 0
            self.tensor[0, i_new, j_new] = 1

            # Update edges where this node is the smaller one
            for u, v in self.edges:
                if u == node:
                    self.tensor[v, i_old, j_old] = 0
                    self.tensor[v, i_new, j_new] = 1

        # Incremental crossings update
        self.crossings = self.update_crossings_incremental(node)

    # -----------------------------------------------------------
    # Cloning (for MuZero)
    # -----------------------------------------------------------

    def clone(self) -> "GraphDrawing":
        """Return a deep copy of this graph-drawing."""
        # Create a new GraphDrawing instance with copies of positions and edges
        new_copy = GraphDrawing(
            n=self.n,
            pos=dict(self.pos),  # deep copy of positions
            edges=list(self.edges),  # copy of edges (already tuples)
            build_tensor=False,  # we'll copy the tensor manually
        )

        # Copy adjacency
        new_copy.adj = {nid: set(neighs) for nid, neighs in self.adj.items()}

        # Copy node_ids
        new_copy.node_ids = list(self.node_ids)

        # Copy crossing caches
        new_copy.crossings = self.crossings
        new_copy.crossing_pairs = self.crossing_pairs

        # Copy tensor if it exists
        if self.tensor is not None:
            new_copy.tensor = np.copy(self.tensor)

        return new_copy

    # -----------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------

    def plot(
        self, show_ids: bool = True, title: str = "", block: bool = False
    ) -> plt.Figure:
        """Plot the graph-drawing using current positions."""
        fig, ax = plt.subplots(figsize=(6, 6))

        # Draw faint grid lines
        for x in range(self.n):
            ax.axvline(x, color="lightgray", linewidth=0.7, zorder=0)
        for y in range(self.n):
            ax.axhline(y, color="lightgray", linewidth=0.7, zorder=0)

        # Draw edges (avoid duplicates)
        drawn_edges = set()
        for u, v in self.edges:
            key = (u, v) if u < v else (v, u)
            if key in drawn_edges:
                continue
            if u in self.pos and v in self.pos:
                drawn_edges.add(key)
                ux, uy = self.pos[u][1], self.pos[u][0]
                vx, vy = self.pos[v][1], self.pos[v][0]
                ax.plot([ux, vx], [uy, vy], linewidth=1.8, color="black", zorder=1)

        # Draw nodes
        xs = [self.pos[nid][1] for nid in sorted(self.pos.keys())]
        ys = [self.pos[nid][0] for nid in sorted(self.pos.keys())]
        ax.scatter(xs, ys, s=140, color="tab:orange", edgecolor="k", zorder=3)

        # Draw absent referenced nodes (hollow)
        referenced = set([u for (u, _) in self.edges] + [v for (_, v) in self.edges])
        absent = {rid for rid in referenced if rid not in self.pos}
        if absent:
            ax.scatter(
                [((i - 1) % self.n) for i in absent],
                [((i - 1) // self.n) for i in absent],
                s=120,
                facecolors="none",
                edgecolors="gray",
                linewidths=1.4,
                zorder=2,
            )

        # Node IDs (or flattened positions)
        if show_ids:
            for nid, (i, j) in self.pos.items():
                flat_pos = i * self.n + j + 1
                ax.text(
                    j,
                    i + 0.18,
                    str(nid),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="blue",
                    zorder=4,
                )

        # Optional: label grid coordinates for debugging
        for i in range(self.n):
            for j in range(self.n):
                ax.text(
                    j,
                    i - 0.15,
                    f"({i},{j})",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color="gray",
                    zorder=2,
                )

        ax.set_xlim(-0.5, self.n - 0.5)
        ax.set_ylim(-0.5, self.n - 0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.axis("off")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        plt.title(title, pad=12)

        if self.crossings is not None:
            fig.text(
                0.5,
                0.02,
                f"Edge crossings: {self.crossings}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.show(block=block)
        plt.pause(2)
        return fig

    # -----------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------

    @staticmethod
    def _segments_intersect(p1, p2, q1, q2):
        """Return True only if segments intersect in their interior (no endpoints)."""

        def orient(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        # Shared endpoints → NOT a crossing
        if p1 == q1 or p1 == q2 or p2 == q1 or p2 == q2:
            return False

        o1 = orient(p1, p2, q1)
        o2 = orient(p1, p2, q2)
        o3 = orient(q1, q2, p1)
        o4 = orient(q1, q2, p2)

        return (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (
            o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0
        )
