"""Port of src/MRGConstrLight.java -- constructs the multiresolutional Reeb
graph (MRG) for a mesh model given its (normalized) mu values.

Construction proceeds as follows:

1. Creation of a Reeb graph at the finest resolution with
   ``create_finest_resolution_reeb_graph()``.
2. Reeb graphs at coarser levels are created by ``create_mrg()``, which
   recursively merges neighboring ranges of the normalized mu values.
3. Data for each R-node in the MRG is maintained by a ReebGraphElement
   object, stored in ``self.reebs[res_idx][rnode_index]``, where
   ``res_idx`` is the resolution index (0 = finest), and ``rnode_index``
   is the R-node index.
4. R-edges between R-nodes at the same resolution are stored via
   adjacency lists in ``self.MRG[res_idx][rnode_index]``, where
   ``MRG[res_idx][rnode_index][0]`` stores the length of the adjacency
   list (mirroring the Java array's counter-at-index-0 convention -- see
   resample.py for more on this).
5. R-edges that connect R-nodes from different resolutions are stored in
   ``ReebGraphElement.parents``: for each R-node at resolution
   ``res_idx``, ``parents`` holds indices to R-nodes at the previous
   (finer) resolution ``res_idx - 1``.
"""

import math

from . import ecomp
from .aman import trim_sparse
from .point import Point
from .reeb_graph_element import ReebGraphElement


class MRGConstrLight:
    def __init__(self):
        self.percentage_threshold = 0.01

        self.points = None
        self.points_length = 0
        self.sparse = None
        self.mu_values = None

        self.all_Tsets = None  # all_Tsets[range] -> list of Tsets (each a list[int])
        self.tsets = None  # flattened list of all Tsets, in the same order as R-node indices

        self.ranges = None

        self.FINEST_RESOLUTION = 0
        self.MRG = None  # MRG[res][node] -> adjacency row (list, [0]=count)
        self.reebs = None  # reebs[res][node] -> ReebGraphElement or None
        self.Rnumber = 0

        self.info = None
        self.stacker = None

        self.cons = None

    def do_process(self, mrg_num, pts, points_size, sprs, mu, chooser):
        mrg_num_d = float(mrg_num)
        ecomp.epsilon2 = 1 / (mrg_num_d * mrg_num_d)

        self.FINEST_RESOLUTION = mrg_num
        self.points = pts
        self.mu_values = mu
        self.points_length = points_size
        self.sparse = sprs

        self.divide_ranges()

        self.do_resampling()
        trim_sparse(self.sparse, self.points_length)
        if not self.checker():
            print("WARNING in MRGConstrLight: Resampling wasn't successful!")

        self.create_tsets()

        self.create_finest_resolution_reeb_graph()

        if chooser:
            print(
                "ERROR in MRGConstrLight: this function is not implemented "
                "(reduceNumberNodes is not ported -- it is dead code in the "
                "original Java too)\n Exiting..."
            )
            raise SystemExit(1)

        self.create_mrg()

        print("Done creating MRG of size: " + str(len(self.MRG)))

        return (self.points, self.points_length, self.sparse, self.mu_values, self.tsets, self.MRG, self.reebs)

    def divide_ranges(self):
        finest = self.FINEST_RESOLUTION
        ranges = [0.0] * (finest + 1)

        ranges[0] = 0.0
        for i in range(1, finest):
            ranges[i] = float(i) / float(finest)
        ranges[finest] = 1.0

        self.ranges = ranges

        self.mu_values = list(self.mu_values[: self.points_length])

    def do_resampling(self):
        i = 0
        while i < self.points_length:
            adj = self.sparse[i]
            length = adj[0]
            range1 = self.what_range(i)

            for j in range(1, length):
                index = adj[j]
                if index > -1:
                    range2 = self.what_range(index)

                    if not self.is_in_one_range(i, index):
                        mu1 = self.mu_values[i]
                        mu2 = self.mu_values[index]

                        if range1 > range2:
                            new_mu = self.ranges[range2 + 1]
                        else:
                            new_mu = self.ranges[range1 + 1]

                        self.create_new_vertex(i, index, mu1, mu2, new_mu)

            i += 1

    def checker(self):
        for i in range(self.points_length):
            adj = self.sparse[i]
            for j in range(1, adj[0]):
                if not self.is_in_one_range(i, adj[j]):
                    return False
        return True

    def is_in_one_range(self, index1, index2):
        mu_v1 = self.mu_values[index1]
        mu_v2 = self.mu_values[index2]

        for i in range(len(self.ranges) - 1):
            left_range = self.ranges[i]
            right_range = self.ranges[i + 1]

            if left_range <= mu_v1 <= right_range:
                if left_range <= mu_v2 <= right_range:
                    return True

        return False

    def create_tsets(self):
        finest = self.FINEST_RESOLUTION
        self.all_Tsets = [[] for _ in range(finest)]

        self.info = [0] * self.points_length

        for i in range(len(self.info)):
            rng_idx = self.what_range(i)
            rng = self.ranges[rng_idx]
            mu_v = self.mu_values[i]

            if rng == mu_v and rng_idx != 0:
                self.info[i] = 2
            else:
                self.info[i] = 1

        i = 0
        while i < len(self.info):
            if self.info[i] == 0:
                i += 1
            elif self.info[i] == 1:
                rang = self.what_range(i)
                one_set = self.create_one_tset(i, rang)
                self.all_Tsets[rang].append(one_set)
            elif self.info[i] == 2 or self.info[i] == 3:
                rang = self.what_range(i) - 1
                one_set = self.create_one_tset(i, rang)
                self.all_Tsets[rang].append(one_set)
            # NOTE: intentionally no "else" / no i += 1 in the elif branches --
            # matches the original Java control flow exactly. A point with
            # info == 2 gets folded into two Tsets (its own range, and the
            # range below it), and only advances once info[i] finally becomes 0.

    def create_one_tset(self, point_index, range_):
        tset = []

        self.stacker = []

        index = point_index

        if self.info[point_index] > 0:
            self.stacker.append(index)

        while len(self.stacker) != 0:
            index = self.stacker.pop(0)

            if self.create_one_point(index, range_):
                tset.append(index)

        return tset

    def create_one_point(self, point_index, range_):
        result = False
        real_range = self.what_range(point_index)

        if self.info[point_index] > 0:
            result = True

            if self.info[point_index] == 1:
                self.info[point_index] = 0

                if real_range != range_:
                    print("WARNING in MRGConstrLight: error code 0")

            elif self.info[point_index] == 2:
                if real_range == range_:
                    self.info[point_index] = 3
                elif real_range == range_ + 1:
                    self.info[point_index] = 1
                else:
                    print("WARNING in MRGConstrLight: error code 1")

            elif self.info[point_index] == 3:
                self.info[point_index] = 0

                if real_range != range_ + 1:
                    print("WARNING in MRGConstrLight: error code 2")

            adj = self.sparse[point_index]
            length = adj[0]
            for i in range(1, length):
                index = adj[i]
                if self.info[index] > 0:
                    real_range = self.what_range(index)

                    if self.info[index] == 1 and real_range == range_:
                        if index not in self.stacker:
                            self.stacker.append(index)

                    elif self.info[index] == 3 and real_range == range_ + 1:
                        if index not in self.stacker:
                            self.stacker.append(index)

                    elif self.info[index] == 2:
                        if real_range == range_ + 1 or real_range == range_:
                            if index not in self.stacker:
                                self.stacker.append(index)

        return result

    def create_finest_resolution_reeb_graph(self):
        mrg_size = 0
        tm = len(self.ranges) - 1
        while tm != 1:
            mrg_size += 1
            tm = tm // 2
        mrg_size += 1

        r_number = 0
        for i in range(len(self.all_Tsets)):
            r_number += len(self.all_Tsets[i])

        self.Rnumber = r_number

        self.MRG = [None] * mrg_size
        self.MRG[0] = [[1] for _ in range(r_number)]

        self.reebs = [None] * mrg_size
        self.reebs[0] = [None] * r_number

        for i in range(r_number):
            rel = ReebGraphElement()
            rel.index = i

            temp_int = self.calculate_range(i)
            rel.left_bound = self.ranges[temp_int]
            rel.right_bound = self.ranges[temp_int + 1]

            rel.parents = None

            rel.Tsets = [i]

            self.reebs[0][i] = rel

        main_index = 0

        for i in range(len(self.all_Tsets) - 1):
            current = self.all_Tsets[i]
            following = self.all_Tsets[i + 1]

            for j in range(len(current)):
                tset1 = current[j]

                for k in range(len(following)):
                    tset2 = following[k]

                    if self.is_connected_to_tset(tset1, tset2):
                        index2 = main_index + len(current) + k
                        index1 = main_index + j

                        self.MRG[0][index1].append(index2)
                        self.MRG[0][index1][0] += 1

                        self.MRG[0][index2].append(index1)
                        self.MRG[0][index2][0] += 1

            main_index += len(current)

        self.put_tsets_in_vector()

    def calculate_range(self, index):
        temp = 0
        for i in range(len(self.all_Tsets)):
            temp += len(self.all_Tsets[i])
            if temp > index:
                return i

        print("WARNING in MRGConstrLight: problem with calculating range. Returning -1")
        return -1

    def put_tsets_in_vector(self):
        tsets = []
        for i in range(len(self.all_Tsets)):
            for tset in self.all_Tsets[i]:
                tsets.append(tset)
        self.tsets = tsets

    def create_mrg(self):
        prev_mrg_count = 0

        while len(self.ranges) > 2:
            curr_mrg_count = prev_mrg_count + 1

            # create a clone of the RG
            prev_reebs = self.reebs[prev_mrg_count]
            self.reebs[curr_mrg_count] = [None] * len(prev_reebs)

            for i in range(len(self.reebs[curr_mrg_count])):
                element = prev_reebs[i]
                copy_element = ReebGraphElement()

                copy_element.index = element.index
                copy_element.left_bound = element.left_bound
                copy_element.right_bound = element.right_bound

                copy_element.Tsets = list(element.Tsets)

                copy_element.parents = [element.index]

                self.reebs[curr_mrg_count][i] = copy_element

            # holds info on which nodes to be unified
            self.cons = [None] * len(prev_reebs)
            for i in range(len(self.cons)):
                vec = self.MRG[prev_mrg_count][i]
                self.cons[i] = list(vec[: vec[0]])

            i = 0
            while i < len(self.ranges) - 2:
                self.unify_two_ranges(i, i + 1, i + 2, prev_mrg_count)
                i += 2

            # removes every other element from ranges
            temp_ranges = self.ranges
            new_ranges = [0.0] * ((len(temp_ranges) // 2) + 1)

            j = 0
            for i in range(0, len(temp_ranges), 2):
                new_ranges[j] = temp_ranges[i]
                j += 1
            new_ranges[0] = temp_ranges[0]
            new_ranges[-1] = temp_ranges[-1]
            self.ranges = new_ranges

            self.update_reebs(prev_mrg_count)
            self.update_mrg(prev_mrg_count)

            prev_mrg_count += 1

    def update_reebs(self, prev_mrg_index):
        curr_mrg_index = prev_mrg_index + 1
        temp_reeb = self.reebs[curr_mrg_index]

        null_count = 0
        for el in temp_reeb:
            if el is None:
                null_count += 1

        new_reeb = [None] * (len(temp_reeb) - null_count)
        idx = 0
        for el in temp_reeb:
            if el is not None:
                el.index = idx
                new_reeb[idx] = el
                idx += 1

        self.reebs[curr_mrg_index] = new_reeb

    def update_mrg(self, prev_mrg_index):
        curr_mrg_index = prev_mrg_index + 1
        size_reeb = len(self.reebs[curr_mrg_index])

        parents = [None] * size_reeb
        for i in range(size_reeb):
            parents[i] = list(self.reebs[curr_mrg_index][i].parents)

        self.MRG[curr_mrg_index] = [[1] for _ in range(size_reeb)]

        mrg_curr = self.MRG[curr_mrg_index]

        for i in range(size_reeb):
            j = i + 1
            while j < size_reeb:
                for k1 in range(len(parents[i])):
                    for k2 in range(len(parents[j])):
                        if self.are_nodes_connected(
                            prev_mrg_index, parents[i][k1], parents[j][k2]
                        ) and not self.are_nodes_connected(curr_mrg_index, i, j):

                            mrg_curr[i].append(j)
                            mrg_curr[i][0] += 1

                            mrg_curr[j].append(i)
                            mrg_curr[j][0] += 1

                j += 1

    def are_nodes_connected(self, mrg_index, index1, index2):
        adj = self.MRG[mrg_index][index1]
        length = adj[0]

        for i in range(1, length):
            if adj[i] == index2:
                return True

        return False

    def unify_two_ranges(self, left_range_index, mid_range_index, right_range_index, prev_mrg_index):
        curr_mrg_index = prev_mrg_index + 1

        left_range = self.ranges[left_range_index]
        mid_range = self.ranges[mid_range_index]
        right_range = self.ranges[right_range_index]

        reebs_curr = self.reebs[curr_mrg_index]

        i = 0
        while i < len(reebs_curr):
            if reebs_curr[i] is not None:
                element = reebs_curr[i]
                if (
                    (element.left_bound == left_range and element.right_bound == mid_range)
                    or (element.left_bound == mid_range and element.right_bound == right_range)
                    or (element.left_bound == left_range and element.right_bound == right_range)
                ):

                    j = 1
                    temp_i = i
                    while j < self.cons[element.index][0]:
                        index = self.cons[element.index][j]
                        if reebs_curr[index] is not None:
                            element2 = reebs_curr[index]

                            if (
                                (element2.left_bound == left_range and element2.right_bound == mid_range)
                                or (element2.left_bound == mid_range and element2.right_bound == right_range)
                                or (element2.left_bound == left_range and element2.right_bound == right_range)
                            ):
                                self.unify_two_nodes(
                                    element.index, element2.index, left_range, right_range, prev_mrg_index
                                )
                                temp_i = -1
                        j += 1
                    i = temp_i
            i += 1

        for i in range(len(reebs_curr)):
            if reebs_curr[i] is not None:
                elt = reebs_curr[i]

                if (elt.left_bound == left_range and elt.right_bound == mid_range) or (
                    elt.left_bound == mid_range and elt.right_bound == right_range
                ):
                    elt.left_bound = left_range
                    elt.right_bound = right_range

    def unify_two_nodes(self, node_index1, node_index2, left_b, right_b, prev_mrg_index):
        curr_mrg_index = prev_mrg_index + 1

        if node_index1 == node_index2:
            print("WARNING in MRGConstrLight: node_index1 and node_index2 are the same!")
            return

        reebs_curr = self.reebs[curr_mrg_index]
        el1 = reebs_curr[node_index1]
        el2 = reebs_curr[node_index2]

        for temp_integer in el2.Tsets:
            if temp_integer not in el1.Tsets:
                el1.Tsets.append(temp_integer)

        for int_t in el2.parents:
            if int_t not in el1.parents:
                el1.parents.append(int_t)

        el1.left_bound = left_b
        el1.right_bound = right_b

        el2.left_bound = left_b
        el2.right_bound = right_b

        el2.index = -1

        reebs_curr[node_index2] = None

        adj2 = self.cons[node_index2]
        len2 = adj2[0]

        for i in range(1, len2):
            index2 = adj2[i]

            adj1 = self.cons[node_index1]
            len1 = adj1[0]

            found = False
            for j in range(1, len1):
                if adj1[j] == index2:
                    found = True

            if not found and index2 != node_index1:
                adj1.append(index2)
                adj1[0] += 1

    def is_connected_to_tset(self, tset1, tset2):
        for temp in tset1:
            if self.is_connected_to_tset_single(temp, tset2):
                return True
        return False

    def is_connected_to_tset_single(self, ptr_index, tset):
        if tset is None or len(tset) == 0:
            return True

        adj = self.sparse[ptr_index]
        length = adj[0]

        for index in tset:
            for j in range(1, length):
                if adj[j] == index:
                    return True

        return False

    def what_range(self, ptr_index):
        mu_v = self.mu_values[ptr_index]

        if abs(mu_v - 1.0) < ecomp.epsilon2:
            return len(self.ranges) - 2

        for i in range(len(self.ranges) - 1):
            left_range = self.ranges[i]
            right_range = self.ranges[i + 1]

            if left_range <= mu_v < right_range:
                return i

        print("WARNING in MRGConstrLight: algorithm in whatRange is wrong! Returning -1")
        return -1

    def create_new_vertex(self, point1, point2, mu1, mu2, new_mu):
        """Creates new vertex with the value new_mu, puts it in the points
        list and mu_values, and re-wires the sparse connectivity around it."""
        i = point1
        index = point2

        p1 = self.points[point1]
        p2 = self.points[point2]

        p = Point()

        if mu1 > mu2:
            p.X = (p1.X * (new_mu - mu2) + p2.X * (mu1 - new_mu)) / (mu1 - mu2)
            p.Y = (p1.Y * (new_mu - mu2) + p2.Y * (mu1 - new_mu)) / (mu1 - mu2)
            p.Z = (p1.Z * (new_mu - mu2) + p2.Z * (mu1 - new_mu)) / (mu1 - mu2)
        else:
            p.X = (p1.X * (mu2 - new_mu) + p2.X * (new_mu - mu1)) / (mu2 - mu1)
            p.Y = (p1.Y * (mu2 - new_mu) + p2.Y * (new_mu - mu1)) / (mu2 - mu1)
            p.Z = (p1.Z * (mu2 - new_mu) + p2.Z * (new_mu - mu1)) / (mu2 - mu1)

        self.points.append(p)
        self.mu_values.append(new_mu)

        self.points_length += 1

        # removes old connections
        row_i = self.sparse[i]
        k = 1
        stop = False
        while not stop:
            if row_i[k] == index:
                row_i[k] = -1
                stop = True
            k += 1

        row_index = self.sparse[index]
        k = 1
        stop = False
        while not stop:
            if row_index[k] == i:
                row_index[k] = -1
                stop = True
            k += 1

        # adds new ones
        row_i.append(self.points_length - 1)
        row_i[0] += 1

        row_index.append(self.points_length - 1)
        row_index[0] += 1

        # create new element in sparse matrix
        self.sparse.append([3, i, index])

        # looks for extra connections
        adj = row_i
        adj2 = row_index
        for k in range(1, adj[0]):
            i1 = adj[k]
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

    def calculate_distance(self, A, B):
        return math.sqrt((A.X - B.X) ** 2 + (A.Y - B.Y) ** 2 + (A.Z - B.Z) ** 2)

    def is_connected(self, index1, index2):
        adj = self.sparse[index1]
        length = adj[0]

        for i in range(1, length):
            if adj[i] == index2:
                return True

        return False
