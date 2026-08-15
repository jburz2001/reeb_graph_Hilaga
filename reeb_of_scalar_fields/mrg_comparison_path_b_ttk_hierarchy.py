#!/usr/bin/env python3
"""Path B (of the two paths discussed for adapting the Hilaga MRG codebase to
scalar-field Reeb graphs): build the MRG-style multiresolution pyramid from
TTK's OWN persistence-simplified Reeb graphs, instead of Hilaga's uniform
mu-range binning (see mrg_comparison_path_a_grid_native.py for that path).

***************************************************************************
* UNTESTED. `topologytoolkit` (TTK's Python bindings) is not on PyPI --   *
* it's normally installed via conda-forge or built from source, and      *
* could not be installed in the sandbox this was written in (VTK alone   *
* installs fine via pip; TTK does not). This file was written directly   *
* against the TTK/VTK API your exampleReebComparison_*.py files already  *
* use successfully, and reuses their helper functions rather than        *
* reimplementing them, but it has never actually been run against real  *
* TTK output. Please run it in your own environment; search this file    *
* for "ASSUMPTION" comments -- those mark the specific places (mainly    *
* TTK output array names/ports for the full-domain segmentation, which   *
* your existing code never needed) most likely to need adjustment for    *
* your TTK build.                                                        *
*                                                                         *
* What WAS tested here: persistence_pyramid_matching.py, which holds all *
* of this file's matching/comparison logic and has no VTK/TTK dependency *
* at all, was unit-tested directly against synthetic RNode pyramids      *
* (identical pyramids compare equal; mismatched ones score lower). Path  *
* A (mrg_comparison_path_a_grid_native.py) has no TTK dependency at all  *
* and was fully run end-to-end against the real data.                    *
***************************************************************************

Overview
--------
1. For a sequence of increasing persistence thresholds (finest = ~0,
   coarsest = large), run `ttkTopologicalSimplification` +
   `ttkReebGraph(..., WithSegmentation=True)` on the field, exactly the way
   `simplify_field_by_persistence` / `compute_reeb_graph` in
   exampleReebComparison_main.py already do.
2. WithSegmentation gives, for every point of the domain, which Reeb arc it
   belongs to -- this *is* a T-set, directly analogous to the T-sets
   MRGConstrLight builds from mu-range binning. Each arc's T-set is
   disjoint from every other arc's T-set at that level (unlike Hilaga's
   uniform bins, where boundary vertices deliberately belong to two
   adjacent T-sets -- see CompareReebGraph's MLIST/match-candidate
   machinery, which exists specifically to handle that ambiguity and
   therefore isn't needed here).
3. Node attributes a(m) (area) and l(m) (scalar range) are computed with
   the *same* triangle-area machinery python/reeb_graph already uses
   (AttributeCalculation.calculate_tset_area), applied to the T-set each
   arc's segmentation defines, on the same periodic mesh Path A builds --
   normalized the same way Hilaga's does: divided by the number of
   pyramid levels, so each level's a(m) values sum to about 1/num_levels.
4. `children`/`parent` links between consecutive levels are built from
   segmentation overlap: a fine-level arc's parent is whichever
   coarser-level arc contains the most of its points.
5. Comparison is a simplified, self-contained top-down greedy matcher
   (not a reuse of CompareReebGraph.look_for_matching_pair) -- see
   "Why not reuse CompareReebGraph's matching algorithm" below.

Why not reuse CompareReebGraph's matching algorithm
-----------------------------------------------------
CompareReebGraph's matching logic (NLIST/MPAIR bookkeeping, MLIST label
propagation, match-candidate sequences up to length 5) is built around a
specific problem: Hilaga's uniform mu-bins mean a boundary vertex sits in
*two* adjacent T-sets, so a "range" can correspond to more than one R-node,
and the algorithm has to search sequences of R-nodes to find the right
correspondence. TTK's segmentation is a clean partition (every point in
exactly one arc), so that ambiguity doesn't exist here, and reusing that
machinery unmodified would be solving a problem this data doesn't have.
What *is* reused is the underlying idea (see persistence_pyramid_matching.py):
match greedily top-down from the coarsest level, compare nodes with
sim(m,n) = w*min(a_m,a_n) + (1-w)*min(l_m,l_n), and only match children of
already-matched parents.

Note on SIM(M,M): unlike Hilaga's AttributeCalculation.normalize_attributes
(which rescales `a` so a perfect self-comparison is exactly 1.0), this file
does not apply an equivalent correction -- so SIM(sherwood, sherwood) is not
guaranteed to land on exactly 1.0 the way it does in Path A. Look at the
"cross / average(self)" ratio the script prints, not the raw SIM values, to
judge how close sherwood and sherwoodRolled are to each other.

Usage:
    python3 mrg_comparison_path_b_ttk_hierarchy.py [--resize N]
        [--thresholds T1,T2,T3,...] [--sim-weight W] [--orbit-index I]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import orbithunter as orb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

# compute_reeb_graph/simplify_field_by_persistence/field_to_vtk_dataset/
# iter_leaf_datasets are all defined in (or re-exported by, via its own
# `from exampleReebComparison_miscellaneous import *`) exampleReebComparison_main.py.
from exampleReebComparison_main import (  # noqa: E402  (needs sys.path set first)
    compute_reeb_graph,
    field_to_vtk_dataset,
    iter_leaf_datasets,
    simplify_field_by_persistence,
)

from reeb_graph.attribute_element import AttributeElement  # noqa: E402
from reeb_graph.extract_reeb_graph import calculate_whole_area  # noqa: E402
from reeb_graph.rnode import RNode  # noqa: E402
from reeb_graph.sparse_matrix import SparseMatrix  # noqa: E402

from mrg_comparison_path_a_grid_native import (  # noqa: E402
    REFERENCE_GRID,
    REFERENCE_ROLL_T,
    REFERENCE_ROLL_X,
    build_periodic_mesh,
    done,
    log,
)
from persistence_pyramid_matching import compare_pyramids, link_levels  # noqa: E402
from ttk_segmentation import calculate_tset_area, find_point_segmentation, tsets_from_segmentation  # noqa: E402

FPO_PATH = Path(__file__).parent / "fundamental_periodic_orbits.h5"


def load_sherwood_fields(orbit_index: int, resize: int, roll_t: int, roll_x: int):
    orbits = orb.io.read_h5(str(FPO_PATH))
    orbit = orbits[orbit_index]
    field_orbit = orbit.resize(resize, resize).transform(to="field")
    field_sherwood = np.asarray(field_orbit.state, dtype=np.float64)
    field_sherwood_rolled = np.roll(field_sherwood, shift=(roll_t, roll_x), axis=(0, 1))
    return field_sherwood, field_sherwood_rolled


def build_persistence_level(field, image, threshold, whole_area, points, sparse, num_levels, level_label):
    log(f"  [{level_label}] simplifying at persistence threshold {threshold:g}...")
    simplified = simplify_field_by_persistence(image, threshold) if threshold > 0 else image
    t0 = log(f"  [{level_label}] running ttkReebGraph (WithSegmentation=True)...")
    reeb_outputs = compute_reeb_graph(simplified, with_segmentation=True)
    done(t0)

    field_flat = field.ravel(order="C")
    segmentation = find_point_segmentation(reeb_outputs, len(points), iter_leaf_datasets)
    tsets_by_arc = tsets_from_segmentation(segmentation, len(points))

    nodes = []
    tset_point_sets = []
    raw_areas = []
    raw_ranges = []
    for arc_id, tset_points in sorted(tsets_by_arc.items()):
        raw_areas.append(calculate_tset_area(tset_points, points, sparse))
        values = [field_flat[p] for p in tset_points]
        raw_ranges.append((max(values) - min(values)) if values else 0.0)

        node = RNode()
        node.index = len(nodes)
        node.left_bound = min(values) if values else 0.0
        node.right_bound = max(values) if values else 0.0
        node.children = None  # filled in by link_levels()
        node.parent = None
        node.attribute = AttributeElement()
        node.Tsets = list(tset_points)
        nodes.append(node)
        tset_point_sets.append(set(tset_points))

    total_range = sum(raw_ranges)
    # normalized the way Hilaga's calculate_attributes() does: each
    # attribute is divided by num_levels (his "rnum" = number of MRG
    # resolutions), so a(m)/l(m) across one level sum to ~1/num_levels.
    for node, area, value_range in zip(nodes, raw_areas, raw_ranges):
        node.attribute.a = (area / whole_area / num_levels) if whole_area else 0.0
        node.attribute.l = (value_range / total_range / num_levels) if total_range else 0.0

    return nodes, tset_point_sets


def build_pyramid(field, thresholds, mesh_cache):
    """Returns levels = [nodes_finest, ..., nodes_coarsest], each element a
    list[RNode], with .children/.parent already linked."""
    points, sparse, whole_area = mesh_cache
    image = field_to_vtk_dataset(field, domain_mode="periodic")
    num_levels = len(thresholds)

    levels = []
    point_sets_by_level = []
    for level_index, threshold in enumerate(thresholds):
        nodes, point_sets = build_persistence_level(
            field, image, threshold, whole_area, points, sparse, num_levels, f"level {level_index}"
        )
        levels.append(nodes)
        point_sets_by_level.append(point_sets)

    for level_index in range(len(levels) - 1):
        link_levels(
            levels[level_index], point_sets_by_level[level_index],
            levels[level_index + 1], point_sets_by_level[level_index + 1],
        )

    return levels


def build_mesh(field):
    points, triangles = build_periodic_mesh(field)
    points, points_length, sparse = SparseMatrix().create_matrix(
        triangles, len(triangles), points, len(points)
    )
    whole_area = calculate_whole_area(triangles, points)
    return points, sparse, whole_area


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resize", type=int, default=32, help="grid resized to N x N (default: 32)")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0,0.02,0.06,0.15",
        help="comma-separated persistence thresholds as a FRACTION of the field's peak-to-peak "
        "range, finest (least simplified) first, coarsest last (default: 0,0.02,0.06,0.15). "
        "The last one should be large enough that only one Reeb arc survives.",
    )
    parser.add_argument("--sim-weight", type=float, default=0.5, help="w (default: 0.5)")
    parser.add_argument("--orbit-index", type=int, default=0, help="orbit index within the .h5 file (default: 0)")
    args = parser.parse_args()

    roll_t = round(REFERENCE_ROLL_T * args.resize / REFERENCE_GRID) or 1
    roll_x = round(REFERENCE_ROLL_X * args.resize / REFERENCE_GRID) or 1

    print(f"Loading orbit {args.orbit_index}, resizing to {args.resize}x{args.resize}, "
          f"roll=(t={roll_t}, x={roll_x})...")
    field_sherwood, field_sherwood_rolled = load_sherwood_fields(args.orbit_index, args.resize, roll_t, roll_x)

    thresholds_fraction = [float(t) for t in args.thresholds.split(",")]

    def thresholds_for(field):
        ptp = float(np.ptp(field))
        return [f * ptp for f in thresholds_fraction]

    print("\nBuilding pyramid for sherwood...")
    mesh_cache_a = build_mesh(field_sherwood)
    levels_a = build_pyramid(field_sherwood, thresholds_for(field_sherwood), mesh_cache_a)
    print("Level sizes (finest -> coarsest):", [len(level) for level in levels_a])

    print("\nBuilding pyramid for sherwoodRolled...")
    mesh_cache_b = build_mesh(field_sherwood_rolled)
    levels_b = build_pyramid(field_sherwood_rolled, thresholds_for(field_sherwood_rolled), mesh_cache_b)
    print("Level sizes (finest -> coarsest):", [len(level) for level in levels_b])

    print("\nComparing pyramids...")
    sim_cross = compare_pyramids(levels_a, levels_b, args.sim_weight)
    sim_self_a = compare_pyramids(levels_a, levels_a, args.sim_weight)
    sim_self_b = compare_pyramids(levels_b, levels_b, args.sim_weight)

    print("\n=== Results ===")
    print(f"SIM(sherwood, sherwood)             = {sim_self_a}")
    print(f"SIM(sherwoodRolled, sherwoodRolled) = {sim_self_b}")
    print(f"SIM(sherwood, sherwoodRolled)       = {sim_cross}")
    ratio = sim_cross / ((sim_self_a + sim_self_b) / 2.0) if (sim_self_a + sim_self_b) else float("nan")
    print(f"cross / average(self)               = {ratio}")


if __name__ == "__main__":
    main()
