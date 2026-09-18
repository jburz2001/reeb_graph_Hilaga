#!/usr/bin/env python3
"""Minimal demo of the Hilaga et al. multiresolutional Reeb graph (MRG)
pipeline in ``python/dolphin_comparison`` -- the Python port of the
original Java code at https://github.com/dbespalov/reeb_graph.

Two demos are included:

1. ``compare_scalar_fields(field_a, field_b)``: pass it two 2D scalar
   fields as numpy arrays and get back a similarity score. Each field is
   treated as a height map on a flat triangulated grid (no CAD model
   files or VTK required); internally it's converted to a mesh, turned
   into an MRG, and compared with the original matching algorithm. Needs
   nothing beyond this file, numpy, and ``dolphin_comparison``.

2. ``run_original_cad_model_tests()``: replicates the test the original
   Java repository documents in its own README -- build an MRG for each
   sample CAD model and print every pairwise similarity score -- against
   the 16 real CAD models shipped in this repository's ``models/``
   directory. See that function's docstring for the full provenance of
   both the test and the models. Needs this repository's ``models/``
   directory in addition to the above (not just this one file).

Requires:
  - numpy
  - the ``dolphin_comparison`` package, which ships in this repository's
    ``python/`` directory right next to this file -- to run this example
    elsewhere, copy ``python/dolphin_comparison/`` alongside this script
    (or keep it on ``PYTHONPATH``).
  - for ``run_original_cad_model_tests()`` only: this repository's
    ``models/`` directory, also kept alongside this file.

Usage:
    from dolphin_quickstart_example import compare_scalar_fields
    similarity = compare_scalar_fields(field_a, field_b)  # numpy arrays

Or, to see both demos run end to end:
    python3 dolphin_quickstart_example.py
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

from dolphin_comparison.attribute_calculation import AttributeCalculation
from dolphin_comparison.compare_reeb_graph import CompareReebGraph
from dolphin_comparison.extract_reeb_graph import calculate_whole_area, main_one as extract_reeb_graph_one
from dolphin_comparison.java_fmt import java_double_str
from dolphin_comparison.mrg_constr_light import MRGConstrLight
from dolphin_comparison.mrg_io import save_mrg
from dolphin_comparison.mu_normalization import MuNormalization
from dolphin_comparison.point import Point
from dolphin_comparison.sparse_matrix import SparseMatrix
from dolphin_comparison.triangle import Triangle

MODELS_DIR = Path(__file__).resolve().parent / "models"


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


def run_original_cad_model_tests(
    num_pts=500, mu_coeff=0.0005, mrg_size=32, sim_weight=0.5, models_dir=MODELS_DIR
):
    """Replicate the original repository's own test on real CAD models.

    Where this test comes from: the original Java repository
    (https://github.com/dbespalov/reeb_graph) documents, in its own
    README's "Sample Usage" section, running its two programs back to
    back on the sample CAD models it ships: ``ExtractReebGraph`` builds
    an MRG for every model, then ``CompareReebGraph`` prints a
    similarity score for every pair. This function is that same
    pipeline, calling this port's ``dolphin_comparison.extract_reeb_graph``
    (port of ``ExtractReebGraph.java``) and
    ``dolphin_comparison.compare_reeb_graph`` (port of
    ``CompareReebGraph.java``) -- the same code path already verified
    bit-for-bit identical to the original Java's similarity scores (see
    ``scripts/validate_parity.sh`` and this project's top-level
    README.md).

    Where the CAD models come from: the 16 face-vertex VRML meshes in
    this repository's ``models/`` directory, shipped unmodified from the
    original Java repository above (its own ``models/`` directory). Per
    this project's top-level README.md, they were used to obtain the
    experimental results reported in:
        Dmitriy Bespalov, William C. Regli, Ali Shokoufandeh. "Reeb
        graph based shape retrieval for CAD." ASME IDETC, 2003.
        Dmitriy Bespalov, Cheuk Yiu Ip, William C. Regli, Joshua
        Shaffer. "Benchmarking search techniques for CAD." ACM SPM, 2005.

    num_pts/mu_coeff/mrg_size/sim_weight default to a smaller num_pts
    than the original README's own example invocation
    (``4000 0.0005 128 0.5``) so this finishes in about a minute --
    dolphin_comparison is pure Python, unlike the original Java. Pass
    num_pts=4000, mrg_size=128 for a byte-for-byte faithful (but much
    slower) reproduction of that exact invocation.

    Returns a dict of {(model_a_name, model_b_name): similarity}.
    """
    wrl_paths = sorted(Path(models_dir).glob("*.wrl"))
    if not wrl_paths:
        raise FileNotFoundError(
            f"no *.wrl CAD models found in {models_dir} -- run_original_cad_model_tests() "
            "needs this repository's models/ directory, not just this one file"
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        # Work on copies so this never writes .mrg files into the repo's
        # own models/ directory.
        work_paths = []
        for src in wrl_paths:
            dst = tmp_dir / src.name
            dst.write_bytes(src.read_bytes())
            work_paths.append(dst)

        print(
            f"Building MRGs for {len(work_paths)} CAD models from {models_dir} "
            f"(num_pts={num_pts}, mu_coeff={mu_coeff}, mrg_size={mrg_size}) ..."
        )
        for path in work_paths:
            extract_reeb_graph_one(str(path), num_pts, mu_coeff, mrg_size)

        print("\nPairwise similarity scores (original repo's CompareReebGraph algorithm):")
        comparer = CompareReebGraph()
        comparer.w = sim_weight
        results = {}
        for path_i in work_paths:
            for path_j in work_paths:
                # CompareReebGraph.main_one() also prints "Similarity
                # between <tmp path> and <tmp path> is: ..." itself
                # (matching the original Java program's own console
                # output) -- silenced here in favor of the cleaner,
                # basename-only summary line below.
                with contextlib.redirect_stdout(io.StringIO()):
                    comparer.main_one(str(path_i), str(path_j))
                results[(path_i.name, path_j.name)] = comparer.SIM_R_S
                print(f"  Similarity between {path_i.name} and {path_j.name} is {java_double_str(comparer.SIM_R_S)}")

        return results


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

    print("=== Demo 1: compare_scalar_fields() on synthetic scalar fields ===")
    print(f"Similarity(one bump, noisy one bump) = {sim_same_shape:.4f}  (same topology -> expect close to 1.0)")
    print(f"Similarity(one bump, two bumps)      = {sim_different_shape:.4f}  (different topology -> expect lower)")

    print()
    print("=== Demo 2: run_original_cad_model_tests() on the shipped CAD models ===")
    run_original_cad_model_tests()


if __name__ == "__main__":
    main()

