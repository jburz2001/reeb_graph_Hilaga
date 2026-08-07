"""Port of src/SparseMatrix.java.

We use an adjacency-list representation for 3D models (triangle meshes),
stored as a "sparse matrix": ``sparse[i]`` is a list where ``sparse[i][0]``
is the number of used slots (mirroring the Java convention of a counter
stored at index 0, including any tombstoned/-1 entries) and
``sparse[i][1:sparse[i][0]]`` holds the indices of points connected to
point ``i`` (a ``-1`` entry marks a removed connection that has not yet
been compacted out -- see ``aman.trim_sparse``).

NOTE: ``makeRandom`` in the original Java code shuffles the triangle list
using ``Math.random()`` before building the adjacency list. This makes the
original program's output for a given input non-deterministic across runs
(different triangle order can affect the order neighbors appear in each
adjacency list, which can affect tie-breaking later in the pipeline). This
port preserves that same amount of randomization using Python's ``random``
module, but two runs (Java vs. Python, or Python vs. Python) are therefore
not guaranteed to produce bit-identical output -- only statistically
equivalent results, exactly as re-running the original Java code twice
would not reproduce identical output either.
"""

import random


class SparseMatrix:
    def __init__(self):
        self.sparse = None

    def create_matrix(self, triangles, triangles_length, points, points_length):
        """Main function of the class."""
        triangles = self.make_random(triangles, triangles_length)

        sparse = [None] * len(points)
        for i in range(points_length):
            sparse[i] = [1]
        self.sparse = sparse

        # add each triangle with add_connection
        for i in range(triangles_length):
            t = triangles[i]
            a, b, c = t.a, t.b, t.c

            if a != b and not self.is_connected(a, b):
                self.add_connection(a, b)

            if b != c and not self.is_connected(b, c):
                self.add_connection(b, c)

            if a != c and not self.is_connected(a, c):
                self.add_connection(a, c)

        return points, points_length, self.sparse

    def check_algorithm(self, points_length):
        for i in range(points_length):
            for j in range(1, self.sparse[i][0]):
                index = self.sparse[i][j]
                found = False
                for k in range(1, self.sparse[index][0]):
                    if self.sparse[index][k] == i:
                        found = True

                if not found:
                    print("WARNING in SparseMatrix: can not find point i=" + str(i))

    def add_connection(self, index1, index2):
        sparse = self.sparse

        length = sparse[index1][0]
        sparse[index1].append(index2)
        sparse[index1][0] = length + 1

        length = sparse[index2][0]
        sparse[index2].append(index1)
        sparse[index2][0] = length + 1

    def is_connected(self, index1, index2):
        adj = self.sparse[index1]
        length = adj[0]

        for i in range(1, length):
            if adj[i] == index2:
                return True

        return False

    def make_random(self, triangles, length):
        for i in range(length):
            random1 = random.random() * (length - 1)
            random2 = random.random() * (length - 1)

            n1 = int(random1)
            n2 = int(random2)

            triangles[n1], triangles[n2] = triangles[n2], triangles[n1]

        return triangles
