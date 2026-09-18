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

IMPORTANT -- what similarity score to expect, and three bugs found getting
there. Rolling a periodic field is an exact relabeling of its domain
(np.roll is a bijection on grid indices; every value moves to a new point
ID but no value changes), so the two Reeb graphs are truly isomorphic, and
SIM(sherwood, sherwoodRolled) should come out close to SIM(sherwood,
sherwood) -- not just "similar", genuinely close to the construction's own
noise floor. Getting there took three separate fixes, all in *this* file
(python/reeb_graph itself was never touched -- it must stay byte-for-byte
faithful to the original Java, see python/README.md):

1. T-set area is order-dependent (see calculate_tset_area()'s docstring in
   attribute_calculation.py): a triangle-counting loop only scans
   all-but-the-last-two positions of a T-set's point list, which is only
   correct if that list is sorted descending by point ID. MRGConstrLight's
   flood fill produces T-sets in arbitrary (BFS queue) order, so raw area
   was provably wrong by construction-order alone -- confirmed with an
   isolated test: the same 9-unit-area patch measured as 7, 8, or 9
   depending purely on point list order. Fixed below by sorting each T-set
   descending before AttributeCalculation sees it.

2. The periodic mesh was embedded with *flat*, unwrapped (x, t, 0)
   coordinates. A triangle that wraps across the periodic seam (e.g.
   x=x_count-1 to x=0) has vertices that are geometrically far apart in
   that flat embedding despite being topologically adjacent -- measured
   directly, a should-be-0.5-area triangle came out as 3.5. Which physical
   region sits on that seam depends on the field, so rolling silently
   moved the corruption to a different set of T-sets each time. Fixed by
   build_periodic_mesh() embedding points on an actual 3D torus instead
   (see its docstring for why this is an approximation, not exact --
   a flat torus needs 4D for a truly isometric embedding).

3. Even with (1) and (2), MRGConstrLight processes points in ascending
   point-ID order (do_resampling's cascading edge-subdivision, create_tsets'
   flood-fill seed order), and point IDs are assigned by grid position. A
   roll changes *which* physical location holds a given value without
   touching point IDs at all, so the same value gets processed in a
   different relative order between the two fields -- a real effect,
   verified by proving that after relabeling both fields' points by
   ascending mu value (so point ID depends only on value, canceling the
   roll out exactly: same value always gets the same ID, by construction),
   their mu arrays and triangle sets become *provably, exactly* identical.
   Fixed below by relabel_points_by_mu_ascending().

With all three fixes, SIM(sherwood, sherwoodRolled) reaches ~0.999 (vs.
~0.96 before, and a construction noise floor of ~1.0 measured via
--sanity-check) -- the small remaining gap is expected residual curvature
from fix (2)'s embedding not being perfectly isometric (a fundamental
limitation, not a bug -- see build_periodic_mesh()'s docstring).

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
import math
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
PITCHFORK10_PATH = Path(__file__).parent / "pitchfork_10_iterations.h5"
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


def load_clipped10_field(orbit_index: int, resize: int):
    """Loads field_clipped10 the same way as the (commented-out) block in
    exampleReebComparison_main.py's run() -- a genuinely different dataset
    (pitchfork_10_iterations.h5, not a roll of fundamental_periodic_orbits.h5),
    so comparing against it is a real dissimilarity check rather than a
    roll-invariance check."""
    orbits = orb.io.read_h5(str(PITCHFORK10_PATH))
    orbit = orbits[orbit_index]
    field_orbit = orbit.resize(resize, resize).transform(to="field")
    return np.asarray(field_orbit.state, dtype=np.float64)


def point_id(t_index: int, x_index: int, x_count: int) -> int:
    return t_index * x_count + x_index


#: torus radii for build_periodic_mesh's embedding (see its docstring).
#: A flat torus (zero curvature everywhere) cannot be *exactly* isometrically
#: embedded in 3D at all -- only in 4D (Point is 3D: X, Y, Z). A standard
#: R=2,r=1 donut fixes the catastrophic seam-wraparound distortion (a
#: should-be-0.5-area triangle measured as 3.5 with the old flat embedding),
#: but still has real, smooth curvature: the tube's local radius is
#: R + r*cos(theta), so measured triangle area varies by a factor of
#: (R+r)/(R-r) = 3x across the domain depending on x-position alone. Taking
#: the minor radius much smaller than the major radius shrinks that
#: variation toward 0 (it scales as ~r/R), at the cost of no longer
#: resembling a "unit-cell-sized" grid -- irrelevant here since every area
#: is used only as a fraction of the whole (AttributeCalculation.a
#: normalizes by whole_area). Verified empirically: this ratio keeps
#: max/min triangle area within 1.0002x (see module tests before trusting
#: a different ratio).
_TORUS_MAJOR_RADIUS = 10000.0
_TORUS_MINOR_RADIUS = 1.0


def build_periodic_mesh(field: np.ndarray):
    """Builds a periodic triangulated mesh over the field's (t, x) grid --
    domain geometry only, no field values in it. Mirrors
    exampleReebComparison_miscellaneous.numpy_field_to_periodic_vtk_grid /
    _periodic_triangles for the *triangulation*, but embeds points on an
    actual 3D torus rather than a flat (x, t, 0) plane.

    A flat embedding is wrong here: AttributeCalculation.calculate_tset_area
    computes triangle area from raw Euclidean coordinates, and a triangle
    that wraps across the periodic seam (e.g. x=x_count-1 to x=0) would have
    vertices that are geometrically far apart in a flat embedding even
    though they're topologically adjacent -- measured directly, a
    should-be-0.5-area wraparound triangle came out as 3.5. Which physical
    region happens to sit on that (fixed, coordinate-system) seam depends on
    the field, so rolling the field silently corrupted a different set of
    T-set areas each time -- exactly the bug this function fixes.

    On a torus embedding, grid translation (which is exactly what
    np.roll(field, ...) is) becomes a rigid rotation of the embedded torus,
    which is an isometry -- it preserves every pairwise distance, and so
    every triangle's area, exactly. There is no seam discontinuity to catch
    a triangle on."""
    t_count, x_count = field.shape

    points = []
    for t_index in range(t_count):
        phi = 2.0 * math.pi * t_index / t_count
        for x_index in range(x_count):
            theta = 2.0 * math.pi * x_index / x_count
            tube_radius = _TORUS_MAJOR_RADIUS + _TORUS_MINOR_RADIUS * math.cos(theta)
            points.append(
                Point(
                    tube_radius * math.cos(phi),
                    tube_radius * math.sin(phi),
                    _TORUS_MINOR_RADIUS * math.sin(theta),
                )
            )

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


def relabel_points_by_mu_ascending(points, triangles, mu_values):
    """Relabels point IDs by ascending mu value (stable tie-break on the
    old ID) instead of grid position -- see fix (3) in the module
    docstring. Returns (new_points, new_triangles, new_mu_values); does not
    mutate its inputs.

    Verified directly: applying this to both `field` and
    `np.roll(field, shift)` produces bit-for-bit identical mu arrays and
    identical triangle-vertex-ID sets between the two, since a point's new
    ID depends only on its value (shared by both fields, just at different
    original locations) and never on which grid location originally held
    it."""
    n = len(points)
    order = sorted(range(n), key=lambda old_id: (mu_values[old_id], old_id))

    new_id_of_old = [0] * n
    for new_id, old_id in enumerate(order):
        new_id_of_old[old_id] = new_id

    new_points = [None] * n
    new_mu_values = [0.0] * n
    for old_id, new_id in enumerate(new_id_of_old):
        new_points[new_id] = points[old_id]
        new_mu_values[new_id] = mu_values[old_id]

    new_triangles = [
        Triangle(new_id_of_old[t.a], new_id_of_old[t.b], new_id_of_old[t.c]) for t in triangles
    ]

    return new_points, new_triangles, new_mu_values


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
    mu_values = field_to_mu_values(field)
    # fix (3) in the module docstring -- must happen before SparseMatrix
    # builds connectivity, since it renumbers point IDs (and therefore
    # every triangle's vertex references) to depend only on mu value.
    points, triangles, mu_values = relabel_points_by_mu_ascending(points, triangles, mu_values)
    done(t0)

    t0 = log(f"[{label}] building point connectivity...")
    sparse_matrix = SparseMatrix()
    points, points_length, sparse = sparse_matrix.create_matrix(
        triangles, len(triangles), points, len(points)
    )
    done(t0)

    whole_area = calculate_whole_area(triangles, points)

    t0 = log(f"[{label}] constructing MRG (K={mrg_size})...")
    mrg = MRGConstrLight()
    points, points_length, sparse, mu_values, all_tsets, MRG, reebs = mrg.do_process(
        mrg_size, points, points_length, sparse, mu_values, False
    )
    done(t0)
    print(f"  MRG resolutions: {[len(r) for r in reebs]} nodes (finest -> coarsest)")

    # AttributeCalculation.calculate_tset_area's triangle-counting loop only
    # scans all-but-the-last-two positions of each T-set, which is only
    # correct if a T-set's points are in descending order by point ID (then
    # any triangle's largest-index vertex provably can't fall in the
    # skipped last two slots -- see calculate_tset_area()'s docstring).
    # MRGConstrLight's T-sets come out in flood-fill (BFS queue) order,
    # which isn't sorted, so the raw area is provably order-dependent --
    # confirmed empirically: rolling the field (a pure relabeling of point
    # IDs, changing flood-fill order but not the underlying shape) should
    # not change the computed Reeb graph at all, but without this sort it
    # measurably did. Sorting here restores that invariance without
    # touching python/reeb_graph itself (which intentionally preserves this
    # exact behavior for byte-for-byte parity with the original Java --
    # see python/README.md).
    for tset in all_tsets:
        tset.sort(reverse=True)

    t0 = log(f"[{label}] computing node attributes...")
    attribute_calc = AttributeCalculation()
    attributes = attribute_calc.do_process(
        points, points_length, sparse, mu_values, all_tsets, MRG, reebs, whole_area
    )
    done(t0)

    return MRG, reebs, attributes, mrg.FINEST_RESOLUTION


def run_all(args):
    """--against all: builds sherwood, sherwoodRolled, and clipped10, and
    prints the full pairwise SIM table between all three (six comparisons:
    three self-comparisons + three cross-comparisons), including
    SIM(sherwoodRolled, clipped10), which the two-way --against modes
    don't compute."""
    roll_t = round(REFERENCE_ROLL_T * args.resize / REFERENCE_GRID) or 1
    roll_x = round(REFERENCE_ROLL_X * args.resize / REFERENCE_GRID) or 1

    print(f"Loading sherwood + sherwoodRolled: orbit {args.orbit_index}, resized to "
          f"{args.resize}x{args.resize}, roll=(t={roll_t}, x={roll_x})...")
    field_sherwood, field_rolled = load_sherwood_fields(args.orbit_index, args.resize, roll_t, roll_x)

    print(f"Loading clipped10: orbit {args.orbit_index} from {PITCHFORK10_PATH.name}, "
          f"resized to {args.resize}x{args.resize}...")
    field_clipped10 = load_clipped10_field(args.orbit_index, args.resize)

    fields = [
        ("sherwood", "field_sherwood", field_sherwood),
        ("sherwoodRolled", "field_sherwoodRolled", field_rolled),
        ("clipped10", "field_clipped10", field_clipped10),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {}
    for label, file_stem, field in fields:
        MRG, reebs, attributes, finest = build_mrg(field, args.mrg_size, label)
        save_mrg(file_stem + ".wrl", MRG, reebs, attributes, finest)
        Path(file_stem + ".mrg").replace(OUTPUT_DIR / (file_stem + ".mrg"))
        paths[label] = str(OUTPUT_DIR / (file_stem + ".wrl"))

    print("\nComparing all pairs (CompareReebGraph, unmodified)...")
    comparer = CompareReebGraph()
    comparer.w = args.sim_weight

    labels = [label for label, _, _ in fields]
    sims = {}
    for i, label_i in enumerate(labels):
        for label_j in labels[i:]:
            comparer.main_one(paths[label_i], paths[label_j])
            sims[(label_i, label_j)] = comparer.SIM_R_S

    print("\n=== Full pairwise SIM table ===")
    col_width = max(len(l) for l in labels) + 2
    header = " " * col_width + "".join(f"{l:>{col_width}}" for l in labels)
    print(header)
    for label_i in labels:
        row = f"{label_i:<{col_width}}"
        for label_j in labels:
            key = (label_i, label_j) if (label_i, label_j) in sims else (label_j, label_i)
            row += f"{sims[key]:>{col_width}.6f}"
        print(row)

    print("\n=== Key values ===")
    print(f"SIM(sherwood, sherwood)                 = {sims[('sherwood', 'sherwood')]}")
    print(f"SIM(sherwoodRolled, sherwoodRolled)      = {sims[('sherwoodRolled', 'sherwoodRolled')]}")
    print(f"SIM(clipped10, clipped10)                = {sims[('clipped10', 'clipped10')]}")
    print(f"SIM(sherwood, sherwoodRolled)            = {sims[('sherwood', 'sherwoodRolled')]}")
    print(f"SIM(sherwood, clipped10)                 = {sims[('sherwood', 'clipped10')]}")
    print(f"SIM(sherwoodRolled, clipped10)           = {sims[('sherwoodRolled', 'clipped10')]}")

    print("\n=> sherwood and sherwoodRolled are the same field (up to a periodic shift), so they should")
    print("   (and do) compare close to self-similarity. clipped10 is a genuinely different dataset, so")
    print("   its comparisons against both sherwood and sherwoodRolled should land meaningfully lower --")
    print("   and, since sherwoodRolled is just a relabeling of sherwood, SIM(sherwoodRolled, clipped10)")
    print("   should be close to SIM(sherwood, clipped10), not some unrelated third value.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resize", type=int, default=32, help="grid resized to N x N (default: 32; see module docstring on performance)")
    parser.add_argument("--mrg-size", type=int, default=8, help="K, number of ranges in the finest MRG resolution (default: 8)")
    parser.add_argument("--sim-weight", type=float, default=0.5, help="w, trade-off between area and range attributes (default: 0.5)")
    parser.add_argument("--orbit-index", type=int, default=0, help="orbit index within the .h5 file (default: 0)")
    parser.add_argument(
        "--against",
        choices=("rolled", "clipped10", "all"),
        default="rolled",
        help="what to compare field_sherwood against: 'rolled' (default) is field_sherwoodRolled, "
        "a periodic roll of the same field -- a roll-invariance check, expect SIM close to self-similarity. "
        "'clipped10' is field_clipped10, loaded from a genuinely different dataset "
        "(pitchfork_10_iterations.h5) -- a real dissimilarity check, expect SIM to actually be lower. "
        "'all' builds sherwood, sherwoodRolled, AND clipped10, and prints the full pairwise SIM table "
        "(including SIM(sherwoodRolled, clipped10), which the two-way modes don't compute).",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="also build 'sherwood' a second, independent time and compare it to the first build, "
        "to report the construction's own random-shuffle noise floor for context (see module docstring)",
    )
    args = parser.parse_args()

    if args.against == "all":
        run_all(args)
        return

    print(f"Loading sherwood: orbit {args.orbit_index}, resized to {args.resize}x{args.resize}...")
    if args.against == "rolled":
        roll_t = round(REFERENCE_ROLL_T * args.resize / REFERENCE_GRID) or 1
        roll_x = round(REFERENCE_ROLL_X * args.resize / REFERENCE_GRID) or 1
        print(f"Loading sherwoodRolled: same field, roll=(t={roll_t}, x={roll_x})...")
        field_sherwood, field_b = load_sherwood_fields(args.orbit_index, args.resize, roll_t, roll_x)
        label_b, file_b = "sherwoodRolled", "field_sherwoodRolled"
    else:
        field_sherwood, _ = load_sherwood_fields(args.orbit_index, args.resize, 0, 0)
        print(f"Loading clipped10: orbit {args.orbit_index} from {PITCHFORK10_PATH.name}, "
              f"resized to {args.resize}x{args.resize}...")
        field_b = load_clipped10_field(args.orbit_index, args.resize)
        label_b, file_b = "clipped10", "field_clipped10"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    MRG_a, reebs_a, attributes_a, finest_a = build_mrg(field_sherwood, args.mrg_size, "sherwood")
    save_mrg("field_sherwood.wrl", MRG_a, reebs_a, attributes_a, finest_a)
    (Path("field_sherwood.mrg")).replace(OUTPUT_DIR / "field_sherwood.mrg")

    MRG_b, reebs_b, attributes_b, finest_b = build_mrg(field_b, args.mrg_size, label_b)
    save_mrg(file_b + ".wrl", MRG_b, reebs_b, attributes_b, finest_b)
    (Path(file_b + ".mrg")).replace(OUTPUT_DIR / (file_b + ".mrg"))

    sanity_sim = None
    if args.sanity_check:
        MRG_a2, reebs_a2, attributes_a2, finest_a2 = build_mrg(field_sherwood, args.mrg_size, "sherwood (2nd build)")
        save_mrg("field_sherwood_2.wrl", MRG_a2, reebs_a2, attributes_a2, finest_a2)
        (Path("field_sherwood_2.mrg")).replace(OUTPUT_DIR / "field_sherwood_2.mrg")

    print("\nComparing MRGs (CompareReebGraph, unmodified)...")
    comparer = CompareReebGraph()
    comparer.w = args.sim_weight

    sherwood_path = str(OUTPUT_DIR / "field_sherwood.wrl")
    b_path = str(OUTPUT_DIR / (file_b + ".wrl"))

    comparer.main_one(sherwood_path, sherwood_path)
    sim_self_a = comparer.SIM_R_S

    comparer.main_one(b_path, b_path)
    sim_self_b = comparer.SIM_R_S

    comparer.main_one(sherwood_path, b_path)
    sim_cross = comparer.SIM_R_S

    if args.sanity_check:
        sherwood_2_path = str(OUTPUT_DIR / "field_sherwood_2.wrl")
        comparer.main_one(sherwood_path, sherwood_2_path)
        sanity_sim = comparer.SIM_R_S

    print("\n=== Results ===")
    print(f"SIM(sherwood, sherwood)         = {sim_self_a}")
    print(f"SIM({label_b}, {label_b}) = {sim_self_b}")
    print(f"SIM(sherwood, {label_b})       = {sim_cross}")
    ratio = sim_cross / ((sim_self_a + sim_self_b) / 2.0)
    print(f"cross / average(self)           = {ratio}")
    if sanity_sim is not None:
        print(f"\nSIM(sherwood, sherwood REBUILT independently) = {sanity_sim}")
        print("(this is the construction's own random-shuffle noise floor, for context.)")

    if args.against == "rolled":
        if ratio > 0.99:
            print("\n=> Reeb graphs are (numerically) the same: the periodic roll did not change the MRG similarity.")
        else:
            print("\n=> Reeb graphs differ more than construction noise alone would explain -- see the module")
            print("   docstring's \"IMPORTANT -- what similarity score to expect\" section for why.")
    else:
        print(f"\n=> sherwood vs {label_b} is a genuine dissimilarity check (different dataset, not a roll of")
        print("   the same field) -- unlike the 'rolled' case, SIM well below self-similarity here is the")
        print("   *expected*, correct result, not a bug to chase.")


if __name__ == "__main__":
    main()
