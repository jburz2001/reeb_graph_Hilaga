"""TTK-independent half of Path B (see mrg_comparison_path_b_ttk_hierarchy.py
for the full picture and design rationale).

Everything in this file operates purely on RNode/AttributeElement objects
(from python/reeb_graph) via plain attributes (`.attribute.a`, `.attribute.l`,
`.children`, `.parent`) -- no VTK or TTK import, so it can be exercised with
synthetic data and unit-tested without either installed. Only the pyramid
*construction* (mrg_comparison_path_b_ttk_hierarchy.py's
build_persistence_level/build_pyramid) needs TTK.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from reeb_graph.rnode import RNode  # noqa: E402


def sim(m: RNode, n: RNode, w: float) -> float:
    min_a = min(m.attribute.a, n.attribute.a)
    min_l = min(m.attribute.l, n.attribute.l)
    return w * min_a + (1 - w) * min_l


def link_levels(fine_nodes, fine_point_sets, coarse_nodes, coarse_point_sets):
    """Sets fine_nodes[i].parent (an RNode reference) and coarse_nodes[j].children
    (a list of fine-level indices), mirroring the parent/children convention
    RNode already uses elsewhere in this repo. A fine node's parent is
    whichever coarse node's point set overlaps it the most."""
    for coarse_node in coarse_nodes:
        coarse_node.children = []

    for fine_index, fine_points in enumerate(fine_point_sets):
        best_j, best_overlap = None, -1
        for coarse_index, coarse_points in enumerate(coarse_point_sets):
            overlap = len(fine_points & coarse_points)
            if overlap > best_overlap:
                best_overlap = overlap
                best_j = coarse_index
        if best_j is not None:
            coarse_nodes[best_j].children.append(fine_index)
            fine_nodes[fine_index].parent = coarse_nodes[best_j]


def match_children_greedily(children_a, children_b, w: float):
    """Simple greedy bipartite matching by descending sim(): repeatedly pick
    the best remaining pair, remove both, repeat. Used at each step of the
    top-down descent -- see mrg_comparison_path_b_ttk_hierarchy.py's module
    docstring ("Why not reuse CompareReebGraph's matching algorithm") for why
    this replaces CompareReebGraph's match-candidate-sequence search rather
    than reusing it as-is."""
    remaining_a = list(range(len(children_a)))
    remaining_b = list(range(len(children_b)))
    pairs = []

    while remaining_a and remaining_b:
        best = None
        for ia in remaining_a:
            for ib in remaining_b:
                score = sim(children_a[ia], children_b[ib], w)
                if best is None or score > best[0]:
                    best = (score, ia, ib)
        _, ia, ib = best
        pairs.append((children_a[ia], children_b[ib]))
        remaining_a.remove(ia)
        remaining_b.remove(ib)

    return pairs


def compare_pyramids(levels_a, levels_b, w: float) -> float:
    """Top-down greedy match, coarsest level first. levels_a/levels_b are
    each `[nodes_finest, ..., nodes_coarsest]` (list[list[RNode]]), with
    .children/.parent already linked by link_levels(). Both pyramids must
    have the same number of levels."""
    if len(levels_a) != len(levels_b):
        raise ValueError(f"pyramids have different numbers of levels: {len(levels_a)} vs {len(levels_b)}")

    coarsest_a, coarsest_b = levels_a[-1], levels_b[-1]
    if len(coarsest_a) != 1 or len(coarsest_b) != 1:
        print(
            f"NOTE: coarsest level did not simplify to a single node "
            f"({len(coarsest_a)} vs {len(coarsest_b)} nodes) -- raise the largest "
            "--thresholds value. Matching all pairs at the coarsest level greedily instead."
        )
        root_pairs = match_children_greedily(coarsest_a, coarsest_b, w)
    else:
        root_pairs = [(coarsest_a[0], coarsest_b[0])]

    total_sim = 0.0
    frontier = root_pairs
    for level_index in range(len(levels_a) - 2, -1, -1):
        next_frontier = []
        for node_a, node_b in frontier:
            total_sim += sim(node_a, node_b, w)

            children_a = [levels_a[level_index][i] for i in (node_a.children or [])]
            children_b = [levels_b[level_index][i] for i in (node_b.children or [])]
            next_frontier.extend(match_children_greedily(children_a, children_b, w))
        frontier = next_frontier

    for node_a, node_b in frontier:
        total_sim += sim(node_a, node_b, w)

    return total_sim
