# MRG comparison of scalar-field Reeb graphs

Two implementations of comparing `field_sherwood` and `field_sherwoodRolled`
(defined in `exampleReebComparison_main.py`'s `run()`) using the Hilaga
et al. MRG codebase in [`../python/reeb_graph`](../python/reeb_graph),
instead of the `networkx` isomorphism/WL-hash check `exampleReebComparison_main.py`
already does. Both compute `SIM(sherwood, sherwood)`, `SIM(sherwoodRolled,
sherwoodRolled)`, and `SIM(sherwood, sherwoodRolled)` and print the ratio of
the last to the average of the first two.

## Path A -- `mrg_comparison_path_a_grid_native.py`

**Fully implemented, run, and validated** against the real data. Rebuilds
the MRG pyramid directly from the field on its own periodic grid, reusing
`python/reeb_graph` completely unchanged -- Hilaga's own uniform mu-range
binning stands in for a Reeb graph computation, using the field's values
directly as "mu" instead of Hilaga's geodesic-distance-based one.

```bash
python3 mrg_comparison_path_a_grid_native.py --resize 64 --mrg-size 8 --sanity-check
```

`--against` picks what `sherwood` is compared to: `rolled` (default, a
periodic roll of the same field -- roll-invariance check), `clipped10`
(field_clipped10 from `pitchfork_10_iterations.h5`, a genuinely different
dataset -- real dissimilarity check), or `all` (builds all three and prints
the full pairwise SIM table, e.g. to get `SIM(sherwoodRolled, clipped10)`):

```bash
python3 mrg_comparison_path_a_grid_native.py --against all --resize 64 --mrg-size 8
```

Only depends on `numpy`, `orbithunter`, and the dependency-free
`python/reeb_graph` package -- no VTK/TTK needed.

`SIM(sherwood, sherwoodRolled)` now lands within `~0.0001-0.02%` of
`SIM(sherwood, sherwood)` (verified at multiple grid sizes) -- close to the
construction's own random-shuffle noise floor. Getting there took three
separate bug fixes, all contained in this file (`python/reeb_graph` itself
was never touched, and must stay byte-for-byte faithful to the original
Java -- see `../python/README.md`): an order-dependent triangle-area
computation that needed T-sets pre-sorted; a flat (unwrapped) coordinate
embedding that badly distorted the area of triangles crossing the periodic
seam, fixed by embedding the grid on an actual 3D torus instead; and
MRGConstrLight's construction being sensitive to point-ID processing order
in a way a roll disturbs (since IDs are tied to grid position, not value),
fixed by relabeling points by ascending mu value before construction. See
the module docstring's "IMPORTANT" section for the full writeup of each.
Also still true: it only scales to a downsampled grid, not the full
512x512 field (pure-Python `MRGConstrLight` is not linear in point count).

## Path B -- `mrg_comparison_path_b_ttk_hierarchy.py`

**Best-effort, UNTESTED against real TTK** -- `topologytoolkit` isn't on
PyPI and could not be installed in the environment this was written in.
Builds the pyramid from TTK's own persistence-simplified Reeb graphs
instead of Hilaga's binning, reusing `compute_reeb_graph` /
`simplify_field_by_persistence` from `exampleReebComparison_main.py`. Uses
a simplified, self-contained top-down matcher rather than
`CompareReebGraph`'s algorithm (see the module docstring for why). Split
across three files so the non-TTK-dependent logic could still be unit
tested without TTK installed:

- `persistence_pyramid_matching.py` -- the matching/comparison algorithm.
  No VTK/TTK dependency. **Unit-tested** against synthetic pyramids.
- `ttk_segmentation.py` -- interprets TTK's segmentation output into
  T-sets. No VTK/TTK import (duck-types whatever dataset objects it's
  given). **Unit-tested** against mock VTK-shaped objects.
- `mrg_comparison_path_b_ttk_hierarchy.py` -- everything that actually
  invokes TTK filters. Search it for "ASSUMPTION" comments before trusting
  its output; those mark guesses (mainly TTK output array names for the
  full-domain segmentation) made without being able to verify against a
  real TTK install.

```bash
python3 mrg_comparison_path_b_ttk_hierarchy.py --resize 64 --thresholds 0,0.02,0.06,0.15
```

Please run this in your own TTK environment and treat it as a draft to
debug, not a validated result, until you've confirmed
`SEGMENTATION_ARRAY_CANDIDATES` in `ttk_segmentation.py` actually matches
what your TTK build names its segmentation array.
