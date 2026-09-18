"""Port of src/ecomp.java -- epsilon-based floating point comparisons.

``epsilon`` and ``epsilon2`` are module-level (mirroring the Java
``public static double`` fields) and are assigned at runtime by
ExtractReebGraph / MRGConstrLight, exactly as in the original code.
"""

#: default value matches an uninitialized Java "double" static field
epsilon = 0.0
epsilon2 = 0.0


def eq(r, q):
    return abs(r - q) < epsilon


def gr(r, q):
    return (r - epsilon) > q


def le(r, q):
    return (r + epsilon) < q


def ge(r, q):
    return (r + epsilon) > q


def lq(r, q):
    return (r - epsilon) < q
