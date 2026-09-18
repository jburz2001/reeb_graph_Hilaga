"""Port of (the still-relevant part of) src/aman.java.

The original ``aman`` class was mostly a collection of static helpers for
growing/trimming fixed-size Java arrays (``expand``, ``expand_sparse``,
``expand_tsets``, ``cut_ends``, ...). Python lists grow dynamically via
``append``, so none of that capacity-management machinery is needed here
-- every other Java "array" in this port is simply a Python list. The one
piece of ``aman`` that has real algorithmic effect (it removes tombstoned
connections and *reorders* each adjacency row) is ``trimSparse``, ported
below as ``trim_sparse``.
"""


def trim_sparse(sparse, sparse_size):
    """Compacts each row ``sparse[i]`` for ``i`` in ``[0, sparse_size)`` by
    removing tombstoned (``-1``) entries, preserving the relative order of
    the remaining ones. Row ``[0]`` keeps holding the "number of used
    slots" counter, matching the Java array convention used throughout
    this port (see resample.py's module docstring for details)."""
    for i in range(sparse_size):
        array = sparse[i]
        new_array = [1]
        for j in range(1, array[0]):
            if array[j] > -1:
                new_array.append(array[j])
                new_array[0] += 1
        sparse[i] = new_array

    return sparse
