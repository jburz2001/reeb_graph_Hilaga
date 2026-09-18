#!/usr/bin/env python3
"""Bare-minimum demo of comparing two scalar fields with the Hilaga et al.
multiresolutional Reeb graph (MRG) pipeline in ``python/dolphin_comparison``
-- the Python port of the original Java code at
https://github.com/dbespalov/reeb_graph.

This is the scalar-field half of ``dolphin_validation.py``, split out on
its own: no CAD model files, no VTK, and no comparison against the
original Java repository's sample CAD models -- just
``compare_scalar_fields(field_a, field_b)``, which takes two 2D scalar
fields as numpy arrays and returns a similarity score. Each field is
treated as a height map on a flat triangulated grid; internally it's
converted to a mesh, turned into an MRG, and compared with the original
matching algorithm.

Requires:
  - numpy
  - the ``dolphin_comparison`` package, which ships in this repository's
    ``python/`` directory right next to this file -- to run this example
    elsewhere, copy ``python/dolphin_comparison/`` alongside this script
    (or keep it on ``PYTHONPATH``).

Usage:
    from scalar_field_example import compare_scalar_fields
    similarity = compare_scalar_fields(field_a, field_b)  # numpy arrays

Or, to see it run on a couple of small built-in synthetic examples:
    python3 scalar_field_example.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

from python.dolphin_comparison.attribute_calculation import AttributeCalculation
from python.dolphin_comparison.compare_reeb_graph import CompareReebGraph
from python.dolphin_comparison.extract_reeb_graph import calculate_whole_area
from python.dolphin_comparison.mrg_constr_light import MRGConstrLight
from python.dolphin_comparison.mrg_io import save_mrg
from python.dolphin_comparison.mu_normalization import MuNormalization
from python.dolphin_comparison.point import Point
from python.dolphin_comparison.sparse_matrix import SparseMatrix
from python.dolphin_comparison.triangle import Triangle


def make_grid_mesh(rows, cols, value_fn):
    """A flat rows x cols triangulated grid (2 triangles per cell), with a
    scalar field value at each vertex given by value_fn(row, col)."""
    points = [Point(float(c), float(r), 0.0) for r in range(rows) for c in range(cols)]
    values = [value_fn(r, c) for r in range(rows) for c in range(cols)]

    def point_id(r, c):
        return r * cols + c

    triangles = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            p00, p10 = point_id(r, c), point_id(r, c + 1)
            p01, p11 = point_id(r + 1, c), point_id(r + 1, c + 1)
            triangles.append(Triangle(p00, p10, p11))
            triangles.append(Triangle(p00, p11, p01))

    return points, values, triangles


def build_mrg_file(points, values, triangles, mrg_size, output_path):
    """Mutates points/values/triangles in place while constructing the MRG
    (mirroring the original Java's in-place array mutation) -- pass each
    mesh into this function at most once."""
    sparse_matrix = SparseMatrix()
    points, points_length, sparse = sparse_matrix.create_matrix(
        triangles, len(triangles), points, len(points)
    )

    shift = min(values)
    mu_values = MuNormalization().normalize([v - shift for v in values])
    whole_area = calculate_whole_area(triangles, points)

    mrg = MRGConstrLight()
    points, points_length, sparse, mu_values, all_tsets, MRG, reebs = mrg.do_process(
        mrg_size, points, points_length, sparse, mu_values, False
    )
    # AttributeCalculation's Tset-area computation is order-dependent (see
    # its docstring); sorting descending is what makes it correct.
    for tset in all_tsets:
        tset.sort(reverse=True)

    attributes = AttributeCalculation().do_process(
        points, points_length, sparse, mu_values, all_tsets, MRG, reebs, whole_area
    )
    save_mrg(str(output_path), MRG, reebs, attributes, mrg.FINEST_RESOLUTION)


def compare_meshes(mesh_a, mesh_b, mrg_size=8, sim_weight=0.5):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        path_a = tmp_dir / "mesh_a.wrl"
        path_b = tmp_dir / "mesh_b.wrl"
        build_mrg_file(*mesh_a, mrg_size, path_a)
        build_mrg_file(*mesh_b, mrg_size, path_b)

        comparer = CompareReebGraph()
        comparer.w = sim_weight
        comparer.main_one(str(path_a), str(path_b))
        return comparer.SIM_R_S


def compare_scalar_fields(field_a, field_b, mrg_size=8, sim_weight=0.5):
    """Compare two 2D scalar fields and return their MRG similarity score
    (1.0 = topologically identical, lower = less similar).

    field_a, field_b: 2D numpy arrays of numbers, same shape. Each field
    is treated as a height map on a flat triangulated grid.

    mrg_size: number of ranges in the finest MRG resolution (parameter K
    in Hilaga et al.); sim_weight: trade-off in [0, 1] between the area
    attribute `a` and the length attribute `l` in the similarity function.
    """
    field_a = np.asarray(field_a, dtype=np.float64)
    field_b = np.asarray(field_b, dtype=np.float64)
    if field_a.shape != field_b.shape:
        raise ValueError("field_a and field_b must have the same shape")

    def mesh_from_field(field):
        rows, cols = field.shape
        return make_grid_mesh(rows, cols, lambda r, c: float(field[r, c]))

    return compare_meshes(mesh_from_field(field_a), mesh_from_field(field_b), mrg_size, sim_weight)


def gaussian_bump(rows, cols, center_row, center_col, spread):
    r, c = np.mgrid[0:rows, 0:cols]
    return np.exp(-((c - center_col) ** 2 + (r - center_row) ** 2) / spread)


def main():
    rng = np.random.default_rng(0)
    rows = cols = 16

    field_bump = gaussian_bump(rows, cols, rows / 2, cols / 2, spread=30.0)
    field_noisy_bump = field_bump + rng.uniform(-0.01, 0.01, size=(rows, cols))
    field_two_bumps = gaussian_bump(rows, cols, rows / 4, cols / 4, spread=15.0) + gaussian_bump(
        rows, cols, 3 * rows / 4, 3 * cols / 4, spread=15.0
    )

    sim_same_shape = compare_scalar_fields(field_bump, field_noisy_bump)
    sim_different_shape = compare_scalar_fields(field_bump, field_two_bumps)

    print(f"Similarity(one bump, noisy one bump) = {sim_same_shape:.4f}  (same topology -> expect close to 1.0)")
    print(f"Similarity(one bump, two bumps)      = {sim_different_shape:.4f}  (different topology -> expect lower)")


if __name__ == "__main__":
    main()
