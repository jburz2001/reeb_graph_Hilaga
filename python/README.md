# reeb_graph (Python port)

A Python 3 (standard library only) port of the original Java implementation
in [`../src`](../src). This is a line-for-line translation, not a rewrite:
every class and method in the Java sources has a corresponding module/
function here, and the algorithms (including their loop structures, data
layouts, and edge-case behavior) were kept as close to the original as
Python allows. See the top-level [`../README.md`](../README.md) for the
algorithm background and CLI usage.

## Module map

| Java file | Python module |
|---|---|
| `Point.java` | `point.py` |
| `Triangle.java` | `triangle.py` |
| `ecomp.java` | `ecomp.py` (module-level `epsilon`/`epsilon2`, mirroring the Java `static` fields) |
| `AttributeElement.java` | `attribute_element.py` |
| `ReebGraphElement.java` | `reeb_graph_element.py` |
| `RNode.java` | `rnode.py` |
| `MPairElement.java` | `mpair_element.py` |
| `MatchCandidate.java` | `match_candidate.py` |
| `MinHeapElement.java` | `min_heap_element.py` |
| `MinHeap2.java` | `min_heap.py` |
| `GValueStorage.java` | `gvalue_storage.py` |
| `SparseMatrix.java` | `sparse_matrix.py` |
| `Resample.java` | `resample.py` |
| `MuApprox.java` | `mu_approx.py` |
| `MuNormalization.java` | `mu_normalization.py` |
| `MRGConstrLight.java` | `mrg_constr_light.py` |
| `AttributeCalculation.java` | `attribute_calculation.py` |
| `SaveInText.java` + the `readOneFile` half of `CompareReebGraph.java` | `mrg_io.py` (`save_mrg` / `load_mrg`) |
| `ExtractReebGraph.java` (incl. its VRML parser) | `extract_reeb_graph.py` + `vrml_parser.py` |
| `CompareReebGraph.java` | `compare_reeb_graph.py` |
| `aman.java` | `aman.py` (only `trim_sparse` survives -- see below) |
| n/a | `java_fmt.py` (formats floats the way `Double.toString` would, only used for the log filename / log lines) |

## Running it

No dependencies beyond the Python 3 standard library.

```bash
cd python
python3 -m reeb_graph.extract_reeb_graph 4000 0.0005 128 ../models/*.wrl
python3 -m reeb_graph.compare_reeb_graph 4000 0.0005 128 0.5 ../models/*.wrl
```

`.mrg` files are written next to each `.wrl` model, in the same text format
the Java version uses -- the two are cross-readable (a Python-built `.mrg`
can be fed to the Java `CompareReebGraph`, and a Java-built `.mrg` can be
fed to `compare_reeb_graph.py`); this was verified during porting.

## Validating against Java yourself

`../scripts/validate_parity.sh <num_pts> <mu_coeff> <mrg_size> <sim_weight> <model.wrl> [...]`
compiles the Java sources, runs both implementations, and checks the one
thing that's actually guaranteed to match: given the same `.mrg` input,
`CompareReebGraph` and `compare_reeb_graph.py` must produce byte-identical
similarity scores (it also prints the MRG structure from each side for
comparison, but -- per the non-determinism note below -- a mismatch there
is expected and not treated as a failure).

## Notable porting decisions

- **`aman.java` mostly disappears.** It was a grab-bag of static helpers
  for growing/trimming fixed-capacity Java arrays (`expand`,
  `expand_sparse`, `expand_tsets`, `cut_ends`, ...). Python lists grow
  dynamically via `.append()`, so every Java "array" in this port is just
  a Python list and none of that capacity bookkeeping is needed. The one
  function with real algorithmic effect -- `trimSparse`, which removes
  tombstoned (`-1`) adjacency entries and *reorders* the remaining ones --
  is kept, as `aman.trim_sparse`.

- **Adjacency-row convention preserved.** Point connectivity ("the sparse
  matrix") and R-node adjacency in the MRG are still represented as rows
  where `row[0]` is a used-slot counter and `row[1:row[0]]` holds the
  actual neighbor indices, exactly mirroring the Java array layout. This
  was kept (rather than switching to plain Python lists/sets) because a
  few algorithms are sensitive to *insertion order* within these rows --
  most importantly `AttributeCalculation.calculate_tset_area`, whose
  triangle-counting loop only scans all-but-the-last-two elements of a
  Tset, which means area attribute a(m) is provably order-dependent. That
  quirk is reproduced faithfully rather than "fixed".

- **Java array reallocation quirks are not reproduced.** The Java version
  periodically reallocates a row into a bigger fixed-size array once its
  preallocated capacity is exceeded, which can (rarely, for
  high-degree vertices) leave a stale local reference pointing at an
  orphaned array. Python list rows never change identity, so a local
  alias always sees live updates -- this makes the port behave like the
  *intended* algorithm rather than replicating an allocation artifact.

- **One genuine source of run-to-run non-determinism, inherited from the
  original.** `SparseMatrix.makeRandom` (`sparse_matrix.py`) shuffles the
  triangle list with `Math.random()` (`random.random()` here) before
  building point connectivity. This changes adjacency ordering, which
  changes the order `Resample` visits edges, which can change
  `points_length` and downstream MRG *structure* (node/edge counts per
  resolution) in addition to perturbing the `a` attribute. **This is not
  a porting bug** -- running the original Java `ExtractReebGraph` twice on
  the same input produces different `points_length` and structure too
  (verified while validating this port, see `scripts/validate_parity.sh`).
  Do not expect a Java-built and Python-built `.mrg` for the same model to
  line up node-for-node; that is only true by coincidence on small inputs.

- **`CompareReebGraph`'s matching algorithm is fully deterministic** and
  was verified to reproduce Java's similarity scores bit-for-bit given the
  same `.mrg` inputs (the only upstream randomness is in MRG
  *construction*, not comparison). Its one recursive method,
  `lookForMatchingPair` (a pure tail call), is written here as an explicit
  loop to avoid relying on Python's shallower default recursion limit.

- **Confirmed-dead code was dropped**, matching the Java source's own
  disabled/unreachable paths: `MRGConstrLight.reduceNumberNodes()` (never
  called -- the Java code takes a `System.exit(1)` branch instead, which
  this port reproduces), `ExtractReebGraph.calculateSparseArea`,
  `CompareReebGraph.allAlone`/`getNodeName`, and the unused
  `Point.IsBetween*`/`dotProduct` helpers were ported for completeness
  but are (as in Java) never called.
