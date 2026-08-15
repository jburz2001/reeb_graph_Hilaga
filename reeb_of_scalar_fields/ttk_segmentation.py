"""TTK-output-interpretation helpers for Path B
(mrg_comparison_path_b_ttk_hierarchy.py).

Like persistence_pyramid_matching.py, this file has no VTK/TTK import of its
own -- it only calls duck-typed methods (GetPointData(), GetTuple1(), ...) on
whatever dataset objects are handed to it, so it can be unit-tested against
plain mock objects without either installed. Only the actual TTK filter
*invocation* (simplify_field_by_persistence / compute_reeb_graph, called from
mrg_comparison_path_b_ttk_hierarchy.py) needs a real TTK install.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from reeb_graph.attribute_calculation import AttributeCalculation  # noqa: E402

SCALAR_NAME = "u"

# Candidate point-data array names TTK might use for per-point segmentation
# (which Reeb arc a domain point belongs to). ASSUMPTION: exact name/casing
# is a guess based on common TTK conventions ("SegmentationId" is the most
# common one for ttkReebGraph's WithSegmentation output); adjust if your TTK
# build uses something else -- run `dataset.GetPointData()` and print array
# names to check (inspect_ttk_outputs() in exampleReebComparison_main.py
# does this, just with its printing commented out).
SEGMENTATION_ARRAY_CANDIDATES = ("SegmentationId", "ArcId", "RegionId", "BranchId")


def find_point_segmentation(reeb_outputs, expected_point_count: int, iter_leaf_datasets):
    """Finds the point-data integer array that assigns each of the domain's
    `expected_point_count` points to a Reeb arc ID.

    ASSUMPTION: this scans every leaf dataset TTK returned for a point-data
    array (checked by name first, then by any integer-valued array whose
    length matches the full domain) rather than hardcoding an output port
    index, since WithSegmentation's exact port number/array name were not
    verifiable without a TTK install. Raises with a diagnostic listing of
    what *was* found if nothing matches, rather than silently returning
    wrong data.

    `iter_leaf_datasets` is passed in (rather than imported) so this module
    stays free of any VTK/TTK-importing dependency -- callers pass
    exampleReebComparison_main.iter_leaf_datasets.
    """
    seen = []
    for dataset in reeb_outputs:
        for leaf in iter_leaf_datasets(dataset):
            point_data = leaf.GetPointData()
            n_points = leaf.GetNumberOfPoints()
            for index in range(point_data.GetNumberOfArrays()):
                array = point_data.GetArray(index)
                if array is None:
                    continue
                name = array.GetName()
                seen.append((name, n_points))
                if n_points != expected_point_count:
                    continue
                if name in SEGMENTATION_ARRAY_CANDIDATES:
                    return array
            # fall back to *any* full-domain array (besides the scalar field
            # itself) if none of the named candidates matched on this leaf
            if n_points == expected_point_count:
                for index in range(point_data.GetNumberOfArrays()):
                    array = point_data.GetArray(index)
                    if array is not None and array.GetName() != SCALAR_NAME:
                        return array

    raise RuntimeError(
        "Could not find a per-point segmentation array (tried "
        f"{SEGMENTATION_ARRAY_CANDIDATES}) whose length matches the domain's "
        f"{expected_point_count} points. Arrays actually seen (name, length): "
        f"{seen}. Set with_segmentation=True was likely not honored, or this "
        "TTK build names/ports its segmentation output differently -- update "
        "SEGMENTATION_ARRAY_CANDIDATES or find_point_segmentation() to match."
    )


def tsets_from_segmentation(segmentation_array, point_count: int) -> dict:
    """Groups point indices by their arc ID. Returns {arc_id: [point_ids]}."""
    tsets = {}
    for point_index in range(point_count):
        arc_id = int(segmentation_array.GetTuple1(point_index))
        tsets.setdefault(arc_id, []).append(point_index)
    return tsets


def calculate_tset_area(tset_points, points, sparse) -> float:
    """Raw (not yet normalized) triangle area of a T-set, reusing
    AttributeCalculation.calculate_tset_area verbatim -- see that method's
    docstring for the order-dependent truncation quirk this inherits."""
    helper = AttributeCalculation()
    helper.points = points
    helper.sparse = sparse
    helper.points_length = len(points)
    helper.the_area = 0.0
    helper.count0 = helper.count1 = helper.count2 = helper.count3 = 0
    return helper.calculate_tset_area(tset_points)
