#!/usr/bin/env python3
"""Path A (of the two paths discussed for adapting the Hilaga MRG codebase to
scalar-field Reeb graphs): rebuild the MRG pyramid directly from the scalar
field on its own periodic grid, reusing python/reeb_graph UNCHANGED.

This bypasses TTK's Reeb graph computation entirely. Hilaga's own machinery
(MRGConstrLight's uniform mu-range binning + multiresolution merge,
AttributeCalculation's triangle-area math) builds the T-set hierarchy
directly from the grid's connectivity and the field's values -- the same
way it would from a 3D mesh's connectivity and geodesic-distance-based mu,
just substituting:

  * mesh triangle adjacency        -> periodic grid triangle adjacency
    (2 triangles per grid cell, wrapping at both boundaries)
  * Hilaga's own MuApprox mu(v)    -> the scalar field's value at v, used
    (geodesic-distance based)         directly as "mu" (MuApprox is skipped
                                       entirely -- we already have the
                                       function we want the Reeb graph of)
  * mesh triangle area              -> grid cell area (still literal
                                       triangle area, just on a flat unit
                                       grid instead of an arbitrary 3D
                                       mesh -- AttributeCalculation doesn't
                                       care which)

Everything downstream (MRGConstrLight, AttributeCalculation, mrg_io,
CompareReebGraph) is the exact, already-validated python/reeb_graph code --
no modifications. See mrg_comparison_path_b_ttk_hierarchy.py for the other
path (building the pyramid from TTK's own persistence-simplified Reeb
graphs instead).

IMPORTANT -- what similarity score to expect. Rolling a periodic field is
an exact relabeling of its domain (np.roll is a bijection on grid indices;
every value moves to a new point ID but no value changes), so the two
Reeb graphs are truly isomorphic and one might expect SIM(sherwood,
sherwoodRolled) very close to SIM(sherwood, sherwood). In testing, the
self-vs-self noise floor from SparseMatrix's random shuffle alone (two
*independent* builds of the identical field) was ~1e-6 -- negligible. But
sherwood-vs-rolled measured a real, repeatable gap on the order of 4-7%,
well above that noise floor. This traces to AttributeCalculation's area
computation (ported verbatim from the original Java, see its docstring):
the a(m) area attribute is provably dependent on the *insertion order* of
points within a T-set, which comes from a flood fill seeded by scanning
point IDs in ascending order. A roll changes *which* point ID is "first"
at every value, so topologically-corresponding T-sets get built in a
different order between the two fields, and can end up with systematically
different computed areas even though the underlying region is the same
shape. This is a property of Hilaga's original algorithm (faithfully
preserved by the port), not a bug introduced here -- see
python/README.md's "Notable porting decisions" section.

IMPORTANT -- performance. python/reeb_graph is a line-for-line port of the
original Java, which was written for meshes with thousands of points, not
hundreds of thousands. MRGConstrLight's construction is *not* linear in
point count (`unify_two_ranges` restarts its scan on every merge; T-set
flood fill does O(n) "in" checks against a growing list) -- doubling the
grid side (4x the points) measured roughly 15-20x the runtime during
prototyping. The full field_sherwood grid is 512x512 = 262144 points; at
that scale this would take on the order of many hours to a day in pure
Python. This script therefore resizes the field down to a small grid by
default (RESIZE below) -- large enough to show real Reeb-graph structure,
small enough to run in about a minute. Raise --resize if you have the time
budget for it; runtime grows roughly with (points)^2 or worse, so scale up
gradually.

Usage:
    python3 mrg_comparison_path_a_grid_native.py [--resize N] [--mrg-size K]
        [--sim-weight W] [--orbit-index I]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import orbithunter as orb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from reeb_graph.attribute_calculation import AttributeCalculation
from reeb_graph.compare_reeb_graph import CompareReebGraph
from reeb_graph.extract_reeb_graph import calculate_whole_area
from reeb_graph.mrg_constr_light import MRGConstrLight
from reeb_graph.mrg_io import save_mrg
from reeb_graph.mu_normalization import MuNormalization
from reeb_graph.point import Point
from reeb_graph.sparse_matrix import SparseMatrix
from reeb_graph.triangle import Triangle

FPO_PATH = Path(__file__).parent / "fundamental_periodic_orbits.h5"
OUTPUT_DIR = Path(__file__).parent / "results_path_a"

# Original run() in exampleReebComparison_main.py used a 512x512 field with
# this shift; kept as the *reference* shift for that resolution and scaled
# proportionally to whatever --resize is used, so the roll always covers
# the same fraction of the domain.
REFERENCE_GRID = 512
REFERENCE_ROLL_T = 128
REFERENCE_ROLL_X = 170


def log(message: str) -> float:
    print(message, flush=True)
    return time.time()


def done(start: float) -> None:
    print(f"  done in {time.time() - start:.2f}s", flush=True)


def load_sherwood_fields(orbit_index: int, resize: int, roll_t: int, roll_x: int):
    """Loads field_sherwood and field_sherwoodRolled exactly as
    exampleReebComparison_main.run() does, except resized to `resize` x
    `resize` instead of hardcoded 512x512 (see module docstring)."""
    orbits = orb.io.read_h5(str(FPO_PATH))
    orbit = orbits[orbit_index]
    field_orbit = orbit.resize(resize, resize).transform(to="field")
    field_sherwood = np.asarray(field_orbit.state, dtype=np.float64)

    field_sherwood_rolled = np.roll(field_sherwood, shift=(roll_t, roll_x), axis=(0, 1))

    return field_sherwood, field_sherwood_rolled


def point_id(t_index: int, x_index: int, x_count: int) -> int:
    return t_index * x_count + x_index


def build_periodic_mesh(field: np.ndarray):
    """Builds a flat (Z=0) periodic triangulated mesh over the field's
    (t, x) grid -- domain geometry only, no field values in it. Mirrors
    exampleReebComparison_miscellaneous.numpy_field_to_periodic_vtk_grid /
    _periodic_triangles, but producing our own Point/Triangle objects
    instead of a VTK grid."""
    t_count, x_count = field.shape

    points = [Point(float(x), float(t), 0.0) for t in range(t_count) for x in range(x_count)]

    triangles = []
    for t_index in range(t_count):
        t_next = (t_index + 1) % t_count
        for x_index in range(x_count):
            x_next = (x_index + 1) % x_count
            p00 = point_id(t_index, x_index, x_count)
            p10 = point_id(t_index, x_next, x_count)
            p01 = point_id(t_next, x_index, x_count)
            p11 = point_id(t_next, x_next, x_count)
            triangles.append(Triangle(p00, p10, p11))
            triangles.append(Triangle(p00, p11, p01))

    return points, triangles


def field_to_mu_values(field: np.ndarray) -> list:
    """The scalar field itself becomes "mu", instead of Hilaga's own
    MuApprox-computed geodesic-distance-based mu -- see module docstring.

    NOTE: MuNormalization.normalize() is ported verbatim from the original
    Java, which computes (v - min) / max rather than (v - min) / (max -
    min). That only lands in [0, 1] when all values are non-negative,
    which is always true of Hilaga's own mu (a sum of distances x areas)
    but not of an arbitrary signed scalar field. So we shift the field to
    start at 0 first; MuNormalization itself is used completely unchanged.
    """
    shifted = field - field.min()
    return MuNormalization().normalize(list(shifted.ravel(order="C")))


def build_mrg(field: np.ndarray, mrg_size: int, label: str):
    log(f"[{label}] building periodic mesh ({field.shape[0]}x{field.shape[1]} grid)...")
    t0 = time.time()
    points, triangles = build_periodic_mesh(field)
    done(t0)

    t0 = log(f"[{label}] building point connectivity...")
    sparse_matrix = SparseMatrix()
    points, points_length, sparse = sparse_matrix.create_matrix(
        triangles, len(triangles), points, len(points)
    )
    done(t0)

    mu_values = field_to_mu_values(field)
    whole_area = calculate_whole_area(triangles, points)

    t0 = log(f"[{label}] constructing MRG (K={mrg_size})...")
    mrg = MRGConstrLight()
    points, points_length, sparse, mu_values, all_tsets, MRG, reebs = mrg.do_process(
        mrg_size, points, points_length, sparse, mu_values, False
    )
    done(t0)
    print(f"  MRG resolutions: {[len(r) for r in reebs]} nodes (finest -> coarsest)")

    t0 = log(f"[{label}] computing node attributes...")
    attribute_calc = AttributeCalculation()
    attributes = attribute_calc.do_process(
        points, points_length, sparse, mu_values, all_tsets, MRG, reebs, whole_area
    )
    done(t0)

    return MRG, reebs, attributes, mrg.FINEST_RESOLUTION


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resize", type=int, default=32, help="grid resized to N x N (default: 32; see module docstring on performance)")
    parser.add_argument("--mrg-size", type=int, default=8, help="K, number of ranges in the finest MRG resolution (default: 8)")
    parser.add_argument("--sim-weight", type=float, default=0.5, help="w, trade-off between area and range attributes (default: 0.5)")
    parser.add_argument("--orbit-index", type=int, default=0, help="orbit index within the .h5 file (default: 0)")
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="also build 'sherwood' a second, independent time and compare it to the first build, "
        "to report the construction's own random-shuffle noise floor for context (see module docstring)",
    )
    args = parser.parse_args()

    roll_t = round(REFERENCE_ROLL_T * args.resize / REFERENCE_GRID) or 1
    roll_x = round(REFERENCE_ROLL_X * args.resize / REFERENCE_GRID) or 1

    print(f"Loading orbit {args.orbit_index}, resizing to {args.resize}x{args.resize}, "
          f"roll=(t={roll_t}, x={roll_x})...")
    field_sherwood, field_sherwood_rolled = load_sherwood_fields(
        args.orbit_index, args.resize, roll_t, roll_x
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    MRG_a, reebs_a, attributes_a, finest_a = build_mrg(field_sherwood, args.mrg_size, "sherwood")
    save_mrg("field_sherwood.wrl", MRG_a, reebs_a, attributes_a, finest_a)
    (Path("field_sherwood.mrg")).replace(OUTPUT_DIR / "field_sherwood.mrg")

    MRG_b, reebs_b, attributes_b, finest_b = build_mrg(field_sherwood_rolled, args.mrg_size, "sherwoodRolled")
    save_mrg("field_sherwoodRolled.wrl", MRG_b, reebs_b, attributes_b, finest_b)
    (Path("field_sherwoodRolled.mrg")).replace(OUTPUT_DIR / "field_sherwoodRolled.mrg")

    sanity_sim = None
    if args.sanity_check:
        MRG_a2, reebs_a2, attributes_a2, finest_a2 = build_mrg(field_sherwood, args.mrg_size, "sherwood (2nd build)")
        save_mrg("field_sherwood_2.wrl", MRG_a2, reebs_a2, attributes_a2, finest_a2)
        (Path("field_sherwood_2.mrg")).replace(OUTPUT_DIR / "field_sherwood_2.mrg")

    print("\nComparing MRGs (CompareReebGraph, unmodified)...")
    comparer = CompareReebGraph()
    comparer.w = args.sim_weight

    sherwood_path = str(OUTPUT_DIR / "field_sherwood.wrl")
    rolled_path = str(OUTPUT_DIR / "field_sherwoodRolled.wrl")

    comparer.main_one(sherwood_path, sherwood_path)
    sim_self_a = comparer.SIM_R_S

    comparer.main_one(rolled_path, rolled_path)
    sim_self_b = comparer.SIM_R_S

    comparer.main_one(sherwood_path, rolled_path)
    sim_cross = comparer.SIM_R_S

    if args.sanity_check:
        sherwood_2_path = str(OUTPUT_DIR / "field_sherwood_2.wrl")
        comparer.main_one(sherwood_path, sherwood_2_path)
        sanity_sim = comparer.SIM_R_S

    print("\n=== Results ===")
    print(f"SIM(sherwood, sherwood)             = {sim_self_a}")
    print(f"SIM(sherwoodRolled, sherwoodRolled) = {sim_self_b}")
    print(f"SIM(sherwood, sherwoodRolled)       = {sim_cross}")
    ratio = sim_cross / ((sim_self_a + sim_self_b) / 2.0)
    print(f"cross / average(self)               = {ratio}")
    if sanity_sim is not None:
        print(f"\nSIM(sherwood, sherwood REBUILT independently) = {sanity_sim}")
        print("(this is the construction's own random-shuffle noise floor -- compare it to the")
        print(" sherwood-vs-rolled ratio above: if the rolled gap is much bigger than this, it's")
        print(" a real effect, not shuffle noise -- see the module docstring.)")
    if ratio > 0.99:
        print("\n=> Reeb graphs are (numerically) the same: the periodic roll did not change the MRG similarity.")
    else:
        print("\n=> Reeb graphs differ more than construction noise alone would explain -- see the module")
        print("   docstring's \"IMPORTANT -- what similarity score to expect\" section for why.")


if __name__ == "__main__":
    main()
