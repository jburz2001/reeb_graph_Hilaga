"""Port of src/MuApprox.java -- computes approximate values for the mu
function (Hilaga et al., Section 4.2) via multi-source Dijkstra from a set
of "base points" that greedily cover the mesh with radius-r neighborhoods.
"""

import math

from .gvalue_storage import GValueStorage
from .min_heap import MinHeap2
from .min_heap_element import MinHeapElement

INFINITY = 100000000000000.0


class MuApprox:
    def __init__(self):
        self.coefficient = 0.0
        self.r = 0.0
        self.S = 0.0

        self.sparse = None
        self.points = None
        self.points_length = 0

        self.g_values = None  # list[list[GValueStorage]] indexed [base_index][point_index]
        self.base_points_length = 0

        self.base_points = None
        self.base_areas = None

        self.mu_values = None

        self.unvisited_vertices = None

        self.the_area = 0.0

        self.found = 0
        self.not_found = 0

        self.base_counter = 0
        self.count0 = 0
        self.count1 = 0
        self.count2 = 0

    def do_process(self, coeff, s, pts, points_size, sprs):
        self.coefficient = coeff
        self.S = s
        self.the_area = 0.0

        self.found = 0
        self.not_found = 0
        self.base_counter = 5
        self.count0 = self.count1 = self.count2 = 0

        self.sparse = sprs
        self.points = pts
        self.points_length = points_size

        self.calculate_threshold()
        print("Threshold r = " + repr(self.r))

        self.calculate_paths()

        print("Total number of base points: ")
        print(self.base_points_length)

        for i in range(self.points_length):
            self.mu_values[i] = self.mu(i)

        self.the_area = 0.0
        for i in range(self.base_points_length):
            self.the_area = self.the_area + self.base_areas[i]
            if self.base_areas[i] < 0:
                print("WARNING in MuApprox: base_areas area is negative for i=" + str(i))

        return self.mu_values

    def calculate_paths(self):
        # fill unvisited_vertices with all vertices
        self.unvisited_vertices = list(self.points[: self.points_length])

        self.base_points_length = 0
        self.base_points = []
        self.base_areas = []

        self.g_values = []
        self.mu_values = [0.0] * self.points_length

        last_index = 0
        stopper = False
        while not stopper:
            # takes one vertex from unvisited_vertices and puts it into
            # base_points, then calls calculate_shortest_path
            j = last_index
            temp = None
            while j < len(self.unvisited_vertices):
                if self.unvisited_vertices[j] is not None:
                    temp = self.unvisited_vertices[j]
                    last_index = j
                    break
                j += 1

            if temp is not None:
                temp_int = j

                self.base_points.append(temp)
                self.base_areas.append(0.0)

                arr = [None] * (self.points_length + 1)
                for i in range(1, self.points_length + 1):
                    temp_el = MinHeapElement()
                    temp_el.key = INFINITY
                    temp_el.index = i - 1
                    arr[i] = temp_el
                vlist = MinHeap2(arr, self.points_length)

                self.g_values.append(self.calculate_shortest_path(temp_int, vlist))
                self.base_points_length += 1
            else:
                stopper = True

    def calculate_shortest_path(self, base_vertex_index, vlist):
        """Returns a list of GValueStorage holding the geodesic path length
        from base_vertex_index to every single point."""
        result = [None] * self.points_length
        for z in range(self.points_length):
            result[z] = GValueStorage(index=z, value=INFINITY)

        vlist.decrease_key(base_vertex_index, 0.0)

        while vlist.heap_size != 0:
            smallest = vlist.extract_min()
            g_v = smallest.key

            # saves the values of g for each vertex removed from VLIST
            if result[smallest.index].value > smallest.key:
                result[smallest.index].value = smallest.key

            # for each vertex from adj_vers checks g(Va) > g(V) + length(V,Va)
            adj = self.sparse[smallest.index]
            length = adj[0]
            for j in range(1, length):
                index = adj[j]

                length_vva = self.calculate_distance(self.points[smallest.index], self.points[index])

                g_va = result[index].value

                # Main Check
                if g_va > g_v + length_vva:
                    result[index].value = g_v + length_vva
                    vlist.decrease_key(index, g_v + length_vva)

        # creating the vector of vertices that are in the area
        points_in_area = []

        for i in range(len(result)):
            temp_storage = result[i]
            # checks if g(V) <= r (i.e., V point is in the area)
            if temp_storage.value <= self.r:
                # checks if V was not included in the area,
                # then adds it to the area vector
                if self.unvisited_vertices[temp_storage.index] is not None and temp_storage.index != base_vertex_index:
                    points_in_area.append(temp_storage.index)
                self.unvisited_vertices[temp_storage.index] = None

        points_in_area.append(base_vertex_index)
        self.unvisited_vertices[base_vertex_index] = None

        # calculates the area around the base point
        self.calculate_base_area(points_in_area, len(points_in_area))

        return result

    def mu(self, point_index):
        if self.base_points is None:
            print("ERROR in MuApprox: base_points are not initialized!\n Exiting...")
            raise SystemExit(1)

        value = 0.0
        for i in range(self.base_points_length):
            value = value + (self.g(point_index, i) * self.base_areas[i])
        return value

    def g(self, point_index, base_index):
        temp = self.g_values[base_index][point_index]

        if temp.value != INFINITY:
            self.found += 1
            return temp.value
        else:
            self.not_found += 1
            return 0.0

    def calculate_base_area(self, points_in_area, points_in_area_length):
        area = 0.0

        for i in range(points_in_area_length):
            index_a = points_in_area[i]

            adj1 = self.sparse[index_a]
            len1 = adj1[0]
            for j in range(1, len1):
                index_b = adj1[j]

                if index_a > index_b:
                    adj2 = self.sparse[index_b]
                    len2 = adj2[0]
                    for k in range(1, len2):
                        index_c = adj2[k]

                        if index_b > index_c and self.is_connected(index_c, index_a):
                            area = area + self.calculate_trig_area(
                                self.points[index_a], self.points[index_b], self.points[index_c]
                            )

        if points_in_area_length == 0:
            area = 0.0
            self.count0 += 1

        if points_in_area_length == 1:
            area = 1.0
            self.count1 += 1

        if points_in_area_length == 2:
            area = 2.0
            self.count2 += 1

        self.the_area = self.the_area + area

        self.base_areas[self.base_points_length] = area

    def calculate_trig_area(self, A, B, C):
        a = math.sqrt((C.X - B.X) ** 2 + (C.Y - B.Y) ** 2 + (C.Z - B.Z) ** 2)
        b = math.sqrt((A.X - C.X) ** 2 + (A.Y - C.Y) ** 2 + (A.Z - C.Z) ** 2)
        c = math.sqrt((A.X - B.X) ** 2 + (A.Y - B.Y) ** 2 + (A.Z - B.Z) ** 2)

        p = (a + b + c) / 2

        area = p * (p - a) * (p - b) * (p - c)

        if area < 0:
            area = 1.0

        area = math.sqrt(area)

        return area

    def calculate_threshold(self):
        self.r = math.sqrt(self.coefficient * self.S)

    def calculate_distance(self, A, B):
        return math.sqrt((A.X - B.X) ** 2 + (A.Y - B.Y) ** 2 + (A.Z - B.Z) ** 2)

    def is_connected(self, index1, index2):
        adj = self.sparse[index1]
        length = adj[0]

        for i in range(1, length):
            if adj[i] == index2:
                return True

        return False
