"""Port of src/Resample.java.

Resamples the vertices before construction of the MRG until the specified
number of points (``points_number``) is reached, by repeatedly splitting
edges longer than a shrinking ``THRESHOLD`` in half.

Implementation note on the "sparse matrix" representation: each row
``sparse[i]`` is a Python list where ``sparse[i][0]`` is the number of used
slots (mirroring the Java array's counter-at-index-0 convention) and
``sparse[i][1:sparse[i][0]]`` are neighbor point indices (``-1`` marks a
removed/tombstoned connection). Unlike the Java version -- which
periodically reallocates a row into a bigger fixed-size array once its
preallocated capacity is exceeded -- Python lists grow in place via
``append``, so a row object's identity never changes. This makes the port
behave exactly like the *intended* algorithm (a local reference to
``sparse[i]`` always sees live updates), removing the need to track array
capacity at all; it has no effect on the resulting connectivity.
"""

import math

from .aman import trim_sparse
from .point import Point


class Resample:
    def __init__(self):
        self.points = None
        self.sparse = None
        self.THRESHOLD = 0.0  # threshold for resampling vertices
        self.points_length = 0

    def do_process(self, points_number, pts, points_l, sprs):
        self.points = pts
        self.points_length = points_l
        self.sparse = sprs

        self.THRESHOLD = self.approximate_edge_length() * 2

        while self.points_length < points_number - (points_number * 0.0333):
            self.resample_once(points_number)
            self.THRESHOLD = self.THRESHOLD - (self.THRESHOLD * 0.05)
            trim_sparse(self.sparse, self.points_length)

        self.check_algorithm()

        return self.points, self.points_length, self.sparse

    def resample_once(self, points_number):
        stopper = False
        i = 0
        while not stopper:
            adj = self.sparse[i]
            j = 1
            while j < adj[0]:
                index = adj[j]
                if index > -1 and self.distance(i, index) > self.THRESHOLD:
                    A = self.points[i]
                    B = self.points[index]

                    # create new point
                    P = Point()
                    P.X = (A.X + B.X) / 2.0
                    P.Y = (A.Y + B.Y) / 2.0
                    P.Z = (A.Z + B.Z) / 2.0

                    self.points.append(P)
                    self.points_length += 1

                    # removes old connections
                    adj[j] = -1

                    k = 1
                    stop = False
                    while not stop:
                        if self.sparse[index][k] == i:
                            self.sparse[index][k] = -1
                            stop = True
                        k += 1

                    # adds new ones
                    adj.append(self.points_length - 1)
                    adj[0] += 1

                    adj2 = self.sparse[index]
                    adj2.append(self.points_length - 1)
                    adj2[0] += 1

                    # create new element in sparse matrix
                    self.sparse.append([3, i, index])

                    # look for extra connections
                    for k2 in range(1, adj[0]):
                        i1 = adj[k2]
                        if i1 > -1:
                            for h in range(1, adj2[0]):
                                i2 = adj2[h]
                                if i2 > -1 and i1 == i2:
                                    row_i1 = self.sparse[i1]
                                    row_i1.append(self.points_length - 1)
                                    row_i1[0] += 1

                                    row_new = self.sparse[self.points_length - 1]
                                    row_new.append(i1)
                                    row_new[0] += 1

                j += 1

            i += 1

            if i >= self.points_length or self.points_length >= points_number * 2:
                stopper = True

    def approximate_edge_length(self):
        max_len = 0.0

        for i in range(self.points_length):
            adj = self.sparse[i]
            length = adj[0]

            for j in range(1, length):
                if i > adj[j]:
                    AB = self.distance(i, adj[j])

                    if max_len < AB:
                        max_len = AB

        return max_len

    def distance(self, a, b):
        A = self.points[a]
        B = self.points[b]

        return math.sqrt((A.X - B.X) ** 2 + (A.Y - B.Y) ** 2 + (A.Z - B.Z) ** 2)

    def place_in_points(self, P):
        from . import ecomp

        for i in range(self.points_length):
            P1 = self.points[i]

            if ecomp.eq(P.X, P1.X) and ecomp.eq(P.Y, P1.Y) and ecomp.eq(P.Z, P1.Z):
                return i

        return self.points_length

    def is_connected(self, index1, index2):
        adj = self.sparse[index1]
        length = adj[0]

        for i in range(1, length):
            if adj[i] == index2:
                return True

        return False

    def check_algorithm(self):
        for i in range(self.points_length):
            for j in range(1, self.sparse[i][0]):
                index = self.sparse[i][j]
                found = False
                for k in range(1, self.sparse[index][0]):
                    if self.sparse[index][k] == i:
                        found = True

                if not found:
                    print("WARNING in Resample: can not find point i = " + str(i))
