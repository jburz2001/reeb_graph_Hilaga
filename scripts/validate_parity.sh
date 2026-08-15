#!/usr/bin/env bash
#
# Validates the Python port (python/reeb_graph) against the original Java
# implementation (src/*.java) on a set of models.
#
# Two-tier check (see python/README.md for why it's two-tier, not one):
#
#   Tier 1 -- Extraction (ExtractReebGraph / extract_reeb_graph.py):
#     SparseMatrix's triangle shuffle uses Math.random()/random.random(),
#     so even two runs of the *original* Java code produce slightly
#     different `a` attribute values. We only check that MRG *structure*
#     (node counts per resolution) matches between Java and Python.
#
#   Tier 2 -- Comparison (CompareReebGraph / compare_reeb_graph.py):
#     No randomness here. Feeding the SAME .mrg files (Java-built, in
#     this script) to both comparers must produce byte-identical
#     similarity scores. This is the real pass/fail check.
#
# Usage:
#   scripts/validate_parity.sh <num_pts> <mu_coeff> <mrg_size> <sim_weight> <model.wrl> [<model2.wrl> ...]
#
# Example:
#   scripts/validate_parity.sh 500 0.0005 32 0.5 models/goodpart_2.wrl models/fork_3.wrl

set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <num_pts> <mu_coeff> <mrg_size> <sim_weight> <model.wrl> [<model2.wrl> ...]" >&2
    exit 1
fi

NUM_PTS="$1"; MU_COEFF="$2"; MRG_SIZE="$3"; SIM_WEIGHT="$4"; shift 4
MODELS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(mktemp -d)"
trap 'echo "(work dir kept at $WORK_DIR for inspection)"' EXIT

JAVA_BUILD_DIR="$WORK_DIR/javabuild"
JAVA_RUN_DIR="$WORK_DIR/java_run"
PYTHON_RUN_DIR="$WORK_DIR/python_run"
COMPARE_DIR="$WORK_DIR/compare_run"
mkdir -p "$JAVA_BUILD_DIR" "$JAVA_RUN_DIR" "$PYTHON_RUN_DIR" "$COMPARE_DIR"

MODEL_NAMES=()
for m in "${MODELS[@]}"; do
    MODEL_NAMES+=("$(basename "$m")")
    cp "$REPO_ROOT/$m" "$JAVA_RUN_DIR/" 2>/dev/null || cp "$m" "$JAVA_RUN_DIR/"
    cp "$REPO_ROOT/$m" "$PYTHON_RUN_DIR/" 2>/dev/null || cp "$m" "$PYTHON_RUN_DIR/"
done

echo "==> Compiling Java sources..."
javac -d "$JAVA_BUILD_DIR" "$REPO_ROOT"/src/*.java

echo "==> Extracting with Java..."
( cd "$JAVA_RUN_DIR" && java -cp "$JAVA_BUILD_DIR" ExtractReebGraph "$NUM_PTS" "$MU_COEFF" "$MRG_SIZE" "${MODEL_NAMES[@]}" )

echo "==> Extracting with Python..."
( cd "$PYTHON_RUN_DIR" && PYTHONPATH="$REPO_ROOT/python" python3 -m reeb_graph.extract_reeb_graph "$NUM_PTS" "$MU_COEFF" "$MRG_SIZE" "${MODEL_NAMES[@]}" )

echo ""
echo "==> Tier 1 (informational only -- NOT a pass/fail check): comparing MRG structure..."
echo "    SparseMatrix's Math.random()/random.random() triangle shuffle affects Resample's"
echo "    edge-subdivision order, which can change points_length and MRG structure between"
echo "    ANY two runs -- Java-vs-Java included. A mismatch here is expected, not a bug."
for name in "${MODEL_NAMES[@]}"; do
    base="${name%.wrl}"
    java_struct="$(grep -A1 'elements{' "$JAVA_RUN_DIR/$base.mrg" | grep -v 'elements{\|--' | tr '\n' '/')"
    py_struct="$(grep -A1 'elements{' "$PYTHON_RUN_DIR/$base.mrg" | grep -v 'elements{\|--' | tr '\n' '/')"
    if [ "$java_struct" == "$py_struct" ]; then
        echo "  [match]    $base: node counts per resolution: $java_struct"
    else
        echo "  [differs]  $base: java=$java_struct python=$py_struct"
    fi
done

echo ""
echo "==> Tier 2: comparing similarity scores (using Java's .mrg files for both -- this is"
echo "    the check that actually proves the matching algorithm was ported correctly)..."
COMPARE_JAVA_DIR="$COMPARE_DIR/java"
COMPARE_PYTHON_DIR="$COMPARE_DIR/python"
mkdir -p "$COMPARE_JAVA_DIR" "$COMPARE_PYTHON_DIR"
for name in "${MODEL_NAMES[@]}"; do
    base="${name%.wrl}"
    cp "$JAVA_RUN_DIR/$name" "$JAVA_RUN_DIR/$base.mrg" "$COMPARE_JAVA_DIR/"
    cp "$JAVA_RUN_DIR/$name" "$JAVA_RUN_DIR/$base.mrg" "$COMPARE_PYTHON_DIR/"
done

( cd "$COMPARE_JAVA_DIR" && java -cp "$JAVA_BUILD_DIR" CompareReebGraph "$NUM_PTS" "$MU_COEFF" "$MRG_SIZE" "$SIM_WEIGHT" "${MODEL_NAMES[@]}" )
( cd "$COMPARE_PYTHON_DIR" && PYTHONPATH="$REPO_ROOT/python" python3 -m reeb_graph.compare_reeb_graph "$NUM_PTS" "$MU_COEFF" "$MRG_SIZE" "$SIM_WEIGHT" "${MODEL_NAMES[@]}" )

JAVA_LOG=$(ls "$COMPARE_JAVA_DIR"/log_*)
PYTHON_LOG=$(ls "$COMPARE_PYTHON_DIR"/log_*)

if diff -u "$JAVA_LOG" "$PYTHON_LOG"; then
    TIER2_OK=1
    echo "  [OK]   similarity scores are byte-identical"
else
    TIER2_OK=0
    echo "  [FAIL] similarity scores differ (see diff above)"
fi

echo ""
if [ "$TIER2_OK" -eq 1 ]; then
    echo "PASS: Python's comparison algorithm reproduces Java's similarity scores exactly."
    exit 0
else
    echo "FAIL: see Tier 2 diff above."
    exit 1
fi
