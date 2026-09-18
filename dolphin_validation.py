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
   Java repository documents in its own README, at the exact parameters
   from its own example invocation (``4000 0.0005 128 0.5``) -- build an
   MRG for each sample CAD model, then compute the full NxN pairwise
   similarity matrix -- against the 16 real CAD models shipped in this
   repository's ``models/`` directory. ``print_top_k_matches()`` then
   replicates the original repository's "Retrieval Results" section
   (ranking each model's most similar matches); ``plot_similarity_matrix()``
   displays the full matrix as a heatmap with ``plt.imshow()``. See
   ``run_original_cad_model_tests()``'s docstring for the full
   provenance of both the test and the models, and an important note on
   why the printed scores won't exactly match the specific historical
   numbers quoted in the original README (inherited, genuine run-to-run
   randomness -- confirmed present in the original Java code too, not a
   porting bug). Needs this repository's ``models/`` directory in
   addition to the above (not just this one file).

Requires:
  - numpy, matplotlib
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

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

from dolphin_comparison.attribute_calculation import AttributeCalculation
from dolphin_comparison.compare_reeb_graph import CompareReebGraph
from dolphin_comparison.extract_reeb_graph import calculate_whole_area, main_one as extract_reeb_graph_one
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
    num_pts=4000, mu_coeff=0.0005, mrg_size=128, sim_weight=0.5, models_dir=MODELS_DIR
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

    num_pts/mu_coeff/mrg_size/sim_weight default to the exact values used
    in the original README's own example invocation
    (``4000 0.0005 128 0.5``) so this is a faithful replication of that
    test, not an abbreviated one -- expect several minutes for all 16
    models, since dolphin_comparison is pure Python and this mrg_size
    builds a much deeper MRG pyramid than a quick smoke test would need.
    Pass a smaller num_pts/mrg_size (e.g. 500/32) for a much faster, but
    less faithful, sanity check.

    IMPORTANT -- even at these exact parameters, do not expect the
    printed scores to exactly match the specific historical numbers
    quoted in the original README (e.g. "Similarity between
    models/goodpart_2.wrl and models/fork_3.wrl is 0.7661396393161327").
    ``SparseMatrix.makeRandom`` shuffles each model's triangle list with
    an unseeded ``Math.random()``/``random.random()`` call before mesh
    resampling, so the resulting MRG (and every score derived from it)
    varies from run to run -- in the *original, unmodified Java code
    too*. Running the original Java `ExtractReebGraph`+`CompareReebGraph`
    twice in a row at these exact parameters was directly verified to
    give two different sets of scores, both in the same ballpark as the
    README's historical numbers but not identical to them or to each
    other (e.g. goodpart_2/fork_3 landed at 0.7655 and 0.7642 across the
    two runs, vs. the README's 0.7661). CompareReebGraph's matching
    algorithm itself has no randomness and is exactly reproduced here
    (see scripts/validate_parity.sh); the "test" being replicated is the
    methodology, not a specific set of numbers.

    Prints the full NxN similarity matrix (see print_similarity_matrix())
    instead of one line per pair, and returns (names, matrix): `names` is
    the list of model filenames in matrix row/column order, and `matrix`
    is an NxN numpy array with matrix[i, j] = similarity between
    names[i] and names[j].
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

        print("\nComputing pairwise similarity matrix (original repo's CompareReebGraph algorithm)...")
        comparer = CompareReebGraph()
        comparer.w = sim_weight
        names = [path.name for path in work_paths]
        matrix = np.empty((len(work_paths), len(work_paths)))
        for i, path_i in enumerate(work_paths):
            for j, path_j in enumerate(work_paths):
                # CompareReebGraph.main_one() also prints "Similarity
                # between <tmp path> and <tmp path> is: ..." itself
                # (matching the original Java program's own console
                # output) -- silenced in favor of the matrix printed below.
                with contextlib.redirect_stdout(io.StringIO()):
                    comparer.main_one(str(path_i), str(path_j))
                matrix[i, j] = comparer.SIM_R_S

        print_similarity_matrix(names, matrix)
        return names, matrix


def print_similarity_matrix(names, matrix):
    """Print an NxN similarity matrix as an index-legend + aligned table
    (model names are too long to use as column headers directly)."""
    print("\nModel index legend:")
    for i, name in enumerate(names):
        print(f"  [{i:2d}] {name}")

    print()
    header = "      " + "".join(f"{i:>7d}" for i in range(len(names)))
    print(header)
    for i, name in enumerate(names):
        row = "".join(f"{value:7.3f}" for value in matrix[i])
        print(f"[{i:2d}] {row}")


def print_top_k_matches(names, matrix, k=5):
    """Print each model's top-k most similar matches, replicating the
    original Java repository's "Retrieval Results" section
    (https://github.com/dbespalov/reeb_graph#retrieval-results, mirrored
    in this project's top-level README.md): "Pairwise similarity values
    can be used to rank 3D models in terms of their relevance to a query
    model", illustrated there with a figure of the five top-ranked
    models for a few query models (`figs/sample_matches.pdf`). This
    reproduces that ranking as text, for every model, from the
    similarity matrix computed by run_original_cad_model_tests() --
    each model excludes itself from its own ranking.
    """
    print(f"\nTop-{k} matches per model (original repo's 'Retrieval Results'):")
    n = len(names)
    for i, name in enumerate(names):
        ranked = sorted((j for j in range(n) if j != i), key=lambda j: matrix[i, j], reverse=True)
        matches = ", ".join(f"{names[j]} ({matrix[i, j]:.3f})" for j in ranked[:k])
        print(f"  {name}: {matches}")


def plot_similarity_matrix(names, matrix, output_path="similarity_matrix.png"):
    """Display the NxN similarity matrix as a heatmap with plt.imshow()."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    fig.colorbar(im, ax=ax, label="similarity")

    labels = [name.removesuffix(".wrl") for name in names]
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("CAD model MRG similarity matrix")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        print(f"\nSaved similarity matrix heatmap to {output_path}")
    plt.show()


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
    print("(exact original README parameters: 4000 0.0005 128 0.5 -- this takes several minutes)")
    names, matrix = run_original_cad_model_tests()
    print_top_k_matches(names, matrix)
    plot_similarity_matrix(names, matrix)


if __name__ == "__main__":
    main()



# Model index legend:
#   [ 0] bracket_1.wrl
#   [ 1] bracket_2.wrl
#   [ 2] bracket_3.wrl
#   [ 3] fork_1.wrl
#   [ 4] fork_2.wrl
#   [ 5] fork_3.wrl
#   [ 6] goodpart_1.wrl
#   [ 7] goodpart_2.wrl
#   [ 8] housing_1.wrl
#   [ 9] housing_2.wrl
#   [10] linkage_arm_1.wrl
#   [11] linkage_arm_2.wrl
#   [12] socket_1.wrl
#   [13] socket_2.wrl
#   [14] spring_1.wrl
#   [15] spring_2.wrl

#             0      1      2      3      4      5      6      7      8      9     10     11     12     13     14     15
# [ 0]   1.000  0.824  0.824  0.799  0.707  0.770  0.705  0.714  0.758  0.720  0.825  0.772  0.673  0.675  0.815  0.748
# [ 1]   0.824  1.000  0.786  0.800  0.710  0.810  0.703  0.704  0.761  0.767  0.834  0.809  0.672  0.676  0.842  0.830
# [ 2]   0.824  0.786  1.000  0.735  0.805  0.745  0.745  0.761  0.686  0.675  0.756  0.780  0.713  0.706  0.731  0.705
# [ 3]   0.799  0.800  0.735  1.000  0.718  0.804  0.657  0.703  0.751  0.732  0.804  0.806  0.686  0.708  0.806  0.779
# [ 4]   0.707  0.710  0.805  0.718  1.000  0.740  0.751  0.775  0.651  0.641  0.709  0.705  0.776  0.746  0.627  0.612
# [ 5]   0.770  0.810  0.745  0.804  0.740  1.000  0.715  0.766  0.812  0.809  0.840  0.811  0.725  0.722  0.756  0.737
# [ 6]   0.705  0.703  0.745  0.657  0.751  0.715  1.000  0.732  0.636  0.618  0.649  0.679  0.768  0.822  0.658  0.659
# [ 7]   0.714  0.704  0.761  0.703  0.775  0.766  0.732  1.000  0.760  0.755  0.629  0.648  0.779  0.771  0.663  0.648
# [ 8]   0.758  0.761  0.686  0.751  0.651  0.812  0.636  0.760  1.000  0.905  0.806  0.752  0.687  0.686  0.822  0.811
# [ 9]   0.720  0.767  0.675  0.732  0.641  0.809  0.618  0.755  0.905  1.000  0.779  0.751  0.669  0.694  0.777  0.773
# [10]   0.825  0.834  0.756  0.804  0.709  0.840  0.649  0.629  0.806  0.779  1.000  0.868  0.651  0.652  0.904  0.861
# [11]   0.772  0.809  0.780  0.806  0.705  0.811  0.679  0.648  0.752  0.751  0.868  1.000  0.649  0.653  0.850  0.816
# [12]   0.673  0.672  0.713  0.686  0.776  0.725  0.768  0.779  0.687  0.669  0.651  0.649  1.000  0.901  0.623  0.623
# [13]   0.675  0.676  0.706  0.708  0.746  0.722  0.822  0.771  0.686  0.694  0.652  0.653  0.901  1.000  0.634  0.621
# [14]   0.815  0.842  0.731  0.806  0.627  0.756  0.658  0.663  0.822  0.777  0.904  0.850  0.623  0.634  1.000  0.940
# [15]   0.748  0.830  0.705  0.779  0.612  0.737  0.659  0.648  0.811  0.773  0.861  0.816  0.623  0.621  0.940  1.000

# Top-5 matches per model (original repo's 'Retrieval Results'):
#   bracket_1.wrl: linkage_arm_1.wrl (0.825), bracket_2.wrl (0.824), bracket_3.wrl (0.824), spring_1.wrl (0.815), fork_1.wrl (0.799)
#   bracket_2.wrl: spring_1.wrl (0.842), linkage_arm_1.wrl (0.834), spring_2.wrl (0.830), bracket_1.wrl (0.824), fork_3.wrl (0.810)
#   bracket_3.wrl: bracket_1.wrl (0.824), fork_2.wrl (0.805), bracket_2.wrl (0.786), linkage_arm_2.wrl (0.780), goodpart_2.wrl (0.761)
#   fork_1.wrl: linkage_arm_2.wrl (0.806), spring_1.wrl (0.806), fork_3.wrl (0.804), linkage_arm_1.wrl (0.804), bracket_2.wrl (0.800)
#   fork_2.wrl: bracket_3.wrl (0.805), socket_1.wrl (0.776), goodpart_2.wrl (0.775), goodpart_1.wrl (0.751), socket_2.wrl (0.746)
#   fork_3.wrl: linkage_arm_1.wrl (0.840), housing_1.wrl (0.812), linkage_arm_2.wrl (0.811), bracket_2.wrl (0.810), housing_2.wrl (0.809)
#   goodpart_1.wrl: socket_2.wrl (0.822), socket_1.wrl (0.768), fork_2.wrl (0.751), bracket_3.wrl (0.745), goodpart_2.wrl (0.732)
#   goodpart_2.wrl: socket_1.wrl (0.779), fork_2.wrl (0.775), socket_2.wrl (0.771), fork_3.wrl (0.766), bracket_3.wrl (0.761)
#   housing_1.wrl: housing_2.wrl (0.905), spring_1.wrl (0.822), fork_3.wrl (0.812), spring_2.wrl (0.811), linkage_arm_1.wrl (0.806)
#   housing_2.wrl: housing_1.wrl (0.905), fork_3.wrl (0.809), linkage_arm_1.wrl (0.779), spring_1.wrl (0.777), spring_2.wrl (0.773)
#   linkage_arm_1.wrl: spring_1.wrl (0.904), linkage_arm_2.wrl (0.868), spring_2.wrl (0.861), fork_3.wrl (0.840), bracket_2.wrl (0.834)
#   linkage_arm_2.wrl: linkage_arm_1.wrl (0.868), spring_1.wrl (0.850), spring_2.wrl (0.816), fork_3.wrl (0.811), bracket_2.wrl (0.809)
#   socket_1.wrl: socket_2.wrl (0.901), goodpart_2.wrl (0.779), fork_2.wrl (0.776), goodpart_1.wrl (0.768), fork_3.wrl (0.725)
#   socket_2.wrl: socket_1.wrl (0.901), goodpart_1.wrl (0.822), goodpart_2.wrl (0.771), fork_2.wrl (0.746), fork_3.wrl (0.722)
#   spring_1.wrl: spring_2.wrl (0.940), linkage_arm_1.wrl (0.904), linkage_arm_2.wrl (0.850), bracket_2.wrl (0.842), housing_1.wrl (0.822)
#   spring_2.wrl: spring_1.wrl (0.940), linkage_arm_1.wrl (0.861), bracket_2.wrl (0.830), linkage_arm_2.wrl (0.816), housing_1.wrl (0.811)



  