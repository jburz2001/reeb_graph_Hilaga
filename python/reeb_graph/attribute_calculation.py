"""Port of src/AttributeCalculation.java -- computes attributes a(m) and
l(m) for the finest-resolution Reeb graph nodes (Section 4.3 in Hilaga et
al.).

NOTE on ``calculate_tset_area``: the original Java loop only scans Tset
positions ``[0, len(points_in_area) - 2)`` while looking for the
largest-index vertex of each triangle -- meaning the *insertion order* of
points within a Tset genuinely affects which triangles get counted (and
therefore the computed area). This port preserves that behavior exactly by
keeping Tsets as ordinary Python lists in the same order they were built
by MRGConstrLight (a FIFO/queue traversal order), rather than "fixing" it
into an order-independent triangle count.
"""

import math

from .attribute_element import AttributeElement


class AttributeCalculation:
    def __init__(self):
        self.whole_area = 0.0  # the area that is taken up by the object

        self.points = None
        self.sparse = None
        self.mu_values = None
        self.points_length = 0

        self.MRG = None
        self.reebs = None
        self.all_Tsets = None

        self.attributes = None

        self.the_area = 0.0

        self.count0 = 0
        self.count1 = 0
        self.count2 = 0
        self.count3 = 0

        self.INFINITY = 1000000000

    def do_process(self, pts, pts_size, sprs, mu, tsets, mr_graph, rbs, s):
        self.count0 = 0
        self.count1 = 0
        self.count2 = 0
        self.count3 = 0
        self.the_area = 0.0

        self.points = pts
        self.all_Tsets = tsets
        self.MRG = mr_graph
        self.mu_values = mu
        self.whole_area = s
        self.sparse = sprs
        self.reebs = rbs
        self.points_length = pts_size

        self.attributes = [None] * len(self.reebs[0])

        self.calculate_attributes()

        self.normalize_attributes()

        return self.attributes

    def normalize_attributes(self):
        """Multiplies all the values of function a(n) by a coefficient
        that sets the value SIM(M, M) = 1."""
        coefficient = self.whole_area / self.the_area
        for i in range(len(self.attributes)):
            self.attributes[i].a = self.attributes[i].a * coefficient

    def calculate_attributes(self):
        rnum = float(len(self.MRG))

        lens = [0.0] * len(self.all_Tsets)
        sum_len = 0.0

        for i in range(len(self.all_Tsets)):
            tset = self.all_Tsets[i]

            # computes value of a(m) for each tset
            temp = AttributeElement()
            temp.a = (1 / rnum) * (self.calculate_tset_area(tset) / self.whole_area)

            self.attributes[i] = temp

            min_m = self.mu_values[tset[0]]
            max_m = self.mu_values[tset[0]]

            # looks for min and maximum values of mu among all points that lie in one tset
            for j in range(1, len(tset)):
                temp_double = self.mu_values[tset[j]]

                if min_m > temp_double:
                    min_m = temp_double

                if max_m < temp_double:
                    max_m = temp_double

            length = max_m - min_m
            sum_len = sum_len + length

            lens[i] = length

        # calculates l(m) for each tset
        for i in range(len(self.all_Tsets)):
            self.attributes[i].l = (1 / rnum) * (lens[i] / sum_len)

    def calculate_tset_area(self, points_in_area):
        """Computes the area that is taken by one tset."""
        reference = [0] * self.points_length
        for idx in points_in_area:
            reference[idx] = 1

        area = 0.0

        for i in range(len(points_in_area) - 2):
            index_a = points_in_area[i]
            adj = self.sparse[index_a]
            length = adj[0]

            for j in range(1, length):
                index_b = adj[j]

                if index_a > index_b and reference[index_b] == 1:
                    adj2 = self.sparse[index_b]
                    len2 = adj2[0]
                    for k in range(1, len2):
                        index_c = adj2[k]

                        if index_b > index_c and reference[index_c] == 1 and self.is_connected(index_a, index_c):
                            area = area + self.calculate_trig_area(
                                self.points[index_a], self.points[index_b], self.points[index_c]
                            )

        self.the_area = self.the_area + area

        if len(points_in_area) == 0:
            self.count0 += 1
            return 0.0

        if len(points_in_area) == 1:
            self.count1 += 1
            return 0.0

        if len(points_in_area) == 2:
            self.count2 += 1
            return 0.0

        if area == 0.0:
            self.count3 += 1

        return area

    def is_connected(self, index1, index2):
        adj = self.sparse[index1]
        length = adj[0]

        for i in range(1, length):
            if adj[i] == index2:
                return True

        return False

    def calculate_trig_area(self, A, B, C):
        """Computes area of a triangle defined by 3 points A, B, C."""
        a = math.sqrt((C.X - B.X) ** 2 + (C.Y - B.Y) ** 2 + (C.Z - B.Z) ** 2)
        b = math.sqrt((A.X - C.X) ** 2 + (A.Y - C.Y) ** 2 + (A.Z - C.Z) ** 2)
        c = math.sqrt((A.X - B.X) ** 2 + (A.Y - B.Y) ** 2 + (A.Z - B.Z) ** 2)

        p = (a + b + c) / 2

        area = p * (p - a) * (p - b) * (p - c)

        if area < 0:
            return 0.0

        area = math.sqrt(area)

        return area

    def calculate_distance(self, A, B):
        return math.sqrt((A.X - B.X) ** 2 + (A.Y - B.Y) ** 2 + (A.Z - B.Z) ** 2)
