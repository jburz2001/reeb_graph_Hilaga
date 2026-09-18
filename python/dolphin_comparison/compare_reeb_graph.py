#!/usr/bin/env python3
"""Port of src/CompareReebGraph.java.

The method for estimating similarity of 3D mesh models based on
multiresolutional Reeb graphs (MRG) was proposed by Hilaga et al. [1].

``extract_reeb_graph.py`` implements construction of MRGs for 3D mesh
models (see Section 4 in [1]) stored in (pseudo-) VRML format.

``compare_reeb_graph.py`` implements the matching algorithm for a pair of
MRGs (see Section 5 in [1]).

Matching a pair of MRGs proceeds as follows:

1. Two MRGs are stored in ``self.MRG1`` and ``self.MRG2``: for each
   resolution, ``MRG{1,2}[res][node_index]`` is a list whose first element
   is the RNode itself and whose remaining elements are its R-edge
   neighbors at the same resolution (mirroring the nested-Vector structure
   of the original).

2. Matching is performed starting with the coarsest resolution. Each RNode
   maintains ``RNode.children`` to store R-edges from a coarser to a finer
   resolution of the MRG (this is the same data as
   ``ReebGraphElement.parents`` written by ExtractReebGraph -- see
   mrg_io.py).

References:
    [1] Topology Matching for Fully Automatic Similarity Estimation of 3D
        Shapes. Masaki Hilaga, Yoshihisa Shinagawa, Taku Kohmura and
        Tosiyasu L. Kunii. SIGGRAPH, 2001.

Usage::

    python3 -m reeb_graph.compare_reeb_graph <num_pts> <mu_coeff> <mrg_size> <sim_weight> \\
        <model_1>.wrl <model_2>.wrl ... <model_N>.wrl

It is assumed each VRML model was already processed with
``extract_reeb_graph.py`` and that ``<model_i>.mrg`` exists next to
``<model_i>.wrl``.
"""

import math
import sys

from .attribute_element import AttributeElement
from .java_fmt import java_double_str
from .match_candidate import MatchCandidate, add_integer_vector
from .mpair_element import MPairElement
from .mrg_io import load_mrg
from .rnode import RNode

_NEG_INFINITY = -1000000000.0


class CompareReebGraph:
    def __init__(self):
        self.attributes1 = None
        self.attributes2 = None

        self.MRG1 = None
        self.MRG2 = None

        self.w = 0.5

        self.NLIST = None
        self.MPAIR = None
        self.SIM_R_S = 0.0

    # ------------------------------------------------------------------
    # top-level driving methods
    # ------------------------------------------------------------------

    def main_one(self, arg0, arg1):
        """Compares a pair of MRGs read from ``<arg0/arg1 minus '.wrl'>.mrg``."""
        self.attributes1 = None
        self.attributes2 = None

        self.MRG1 = None
        self.MRG2 = None

        self.read_one_file(arg0)
        self.read_one_file(arg1)

        if len(self.MRG1) != len(self.MRG2):
            print("ERROR: number of resolutions must match! Exiting...")
            raise SystemExit(1)

        self.calculate_rest_attributes(self.MRG1, self.attributes1)
        self.calculate_rest_attributes(self.MRG2, self.attributes2)

        self.compute_parents()

        self.do_comparison(arg0, arg1)

        self.attributes1 = None
        self.attributes2 = None

    def read_one_file(self, filename):
        attributes, mrg = load_mrg(filename)

        if self.attributes1 is None and self.MRG1 is None:
            self.attributes1 = attributes
            self.MRG1 = mrg
        elif self.attributes2 is None and self.MRG2 is None:
            self.attributes2 = attributes
            self.MRG2 = mrg
        else:
            print("WARNING: can only compare two MRGs at a time!")

    def do_comparison(self, str0, str1):
        list1 = []
        list2 = []
        self.MPAIR = []
        self.SIM_R_S = 0.0

        reeb_graph = self.MRG1[-1]
        for i in range(len(reeb_graph)):
            list1.append(reeb_graph[i][0])

        reeb_graph = self.MRG2[-1]
        for i in range(len(reeb_graph)):
            list2.append(reeb_graph[i][0])

        self.NLIST = [list1, list2]

        while len(list1) != 0 and len(list2) != 0:
            self.look_for_matching_pair()

        for el in self.MPAIR:
            self.SIM_R_S = self.SIM_R_S + self.sim(el.node1, el.node2)

        print("Similarity  between " + str0 + " and " + str1 + " is: ")
        print(self.SIM_R_S)

    # ------------------------------------------------------------------
    # matching algorithm
    # ------------------------------------------------------------------

    def look_for_matching_pair(self):
        """Takes a node from NLIST that has the largest value of sim(m,m)
        and finds a matching pair for it. Ported from the recursive Java
        method of the same name; the "no match found -> drop the seed node
        and retry" recursion is rewritten as an explicit loop (it was a
        pure tail call in the original)."""
        while True:
            list1 = self.NLIST[0]
            list2 = self.NLIST[1]

            if len(list1) == 0 or len(list2) == 0:
                return

            select_nlist2 = -1
            index2 = -1

            pair, select_nlist, index = self.find_maximum_sim()

            m = n = None
            max_mat = _NEG_INFINITY
            temp_vector = None

            if select_nlist == 2:
                n_list = self.NLIST[0]
                node = pair.node2
            else:
                n_list = self.NLIST[1]
                node = pair.node1

            for i in range(len(n_list)):
                temp_node = n_list[i]

                # both nodes have to be from the same range
                if temp_node.left_bound == node.left_bound and temp_node.right_bound == node.right_bound:

                    if select_nlist == 2:
                        m = temp_node
                        n = node
                    else:
                        m = node
                        n = temp_node

                    range_diff = m.right_bound - m.left_bound
                    what_res_m = self.calculate_index_in_mrg1(m)
                    what_res_n = self.calculate_index_in_mrg2(n)

                    vec_m = self.MRG1[what_res_m][m.index]
                    vec_n = self.MRG2[what_res_n][n.index]

                    # parents of those nodes have to match
                    if self.parents_are_matched(m, n):

                        # MLISTs for two nodes must be the same
                        if n.MLIST == m.MLIST:

                            candidates_count = 1
                            match_candidates = []
                            the_node = temp_node

                            if select_nlist == 1:
                                self.how_many_same_range_nodes_in_nlist(the_node, 2)
                            else:
                                self.how_many_same_range_nodes_in_nlist(the_node, 1)

                            # select_nlist=1 indicates that MRG2 has the largest sim(m,m) value
                            # select_nlist=0 indicates that MRG1 has the largest sim(m,m) value
                            while self.create_match_candidates(
                                the_node, n_list, match_candidates, candidates_count, select_nlist
                            ):

                                for h in range(len(match_candidates)):
                                    candidate_element = match_candidates[h]

                                    if candidates_count != 1:
                                        temp_node = RNode()
                                        temp_node.index = the_node.index
                                        temp_node.left_bound = the_node.left_bound
                                        temp_node.right_bound = the_node.right_bound
                                        temp_node.parent = the_node.parent
                                        temp_node.children = add_integer_vector(None, the_node.children)
                                        temp_node.MLIST = list(the_node.MLIST)
                                        temp_node.attribute = AttributeElement()
                                        temp_node.attribute.a = the_node.attribute.a
                                        temp_node.attribute.l = the_node.attribute.l

                                        for g in range(1, len(candidate_element.vector)):
                                            tmp_in = candidate_element.vector[g]
                                            temp_node4 = n_list[tmp_in]
                                            temp_node.children = add_integer_vector(
                                                temp_node.children, temp_node4.children
                                            )

                                            temp_node.attribute.a = temp_node.attribute.a + temp_node4.attribute.a
                                            temp_node.attribute.l = temp_node.attribute.l + temp_node4.attribute.l

                                        vector_nodes = candidate_element.vector
                                    else:
                                        vector_nodes = None
                                        temp_node = the_node

                                    if select_nlist == 2:
                                        m = temp_node
                                        n = node
                                    else:
                                        m = node
                                        n = temp_node

                                    temp_double = self.mat(m, n)

                                    if max_mat < temp_double:
                                        max_mat = temp_double

                                        if select_nlist == 2:
                                            pair.node1 = temp_node
                                            temp_vector = None if vector_nodes is None else list(vector_nodes)
                                            select_nlist2 = 1
                                            index2 = i
                                        else:
                                            pair.node2 = temp_node
                                            temp_vector = None if vector_nodes is None else list(vector_nodes)
                                            select_nlist2 = 2
                                            index2 = i

                                    if select_nlist == select_nlist2:
                                        print(
                                            "WARNING in CompareReebGraph.look_for_matching_pair(): "
                                            "select_nlist and select_nlist2 point to the same MRG!"
                                        )

                                    temp_int_m = m.MLIST[-1]
                                    temp_int_n = n.MLIST[-1]

                                    # propagate labels
                                    if vector_nodes is None:
                                        for j in range(1, len(vec_m)):
                                            temp_node2 = vec_m[j]

                                            if (
                                                temp_node2.left_bound == m.right_bound
                                                and temp_node2.right_bound == m.right_bound + range_diff
                                                and temp_int_m > 0
                                            ):
                                                temp_node2.MLIST.append(temp_int_m + 1)

                                            if (
                                                temp_node2.right_bound == m.left_bound
                                                and temp_node2.left_bound == m.left_bound - range_diff
                                                and temp_int_m < 0
                                            ):
                                                temp_node2.MLIST.append(temp_int_m - 1)

                                        for j in range(1, len(vec_n)):
                                            temp_node2 = vec_n[j]

                                            if (
                                                temp_node2.left_bound == n.right_bound
                                                and temp_node2.right_bound == n.right_bound + range_diff
                                                and temp_int_n > 0
                                            ):
                                                temp_node2.MLIST.append(temp_int_n + 1)

                                            if (
                                                temp_node2.right_bound == n.left_bound
                                                and temp_node2.left_bound == n.left_bound - range_diff
                                                and temp_int_n < 0
                                            ):
                                                temp_node2.MLIST.append(temp_int_n - 1)

                                match_candidates = []
                                candidates_count += 1

            # if a matching pair can not be found, the R-node is removed
            # from NLIST and we retry (was: recursive call)
            if pair.node1 is None or pair.node2 is None:
                if select_nlist == 1:
                    del list1[index]
                else:
                    del list2[index]

                pair = None
                continue

            # removes matching nodes from NLISTs and adds this pair to MPAIR
            new_vec = None
            if temp_vector is not None:
                new_vec = [n_list[ti] for ti in temp_vector]

            if select_nlist == 1:
                pair.vector_nodes1 = None
                pair.vector_nodes2 = None if temp_vector is None else new_vec
            else:
                pair.vector_nodes2 = None
                pair.vector_nodes1 = None if temp_vector is None else new_vec

            self.MPAIR.append(pair)

            m = pair.node1
            n = pair.node2

            if select_nlist == 1:
                del list1[index]
            else:
                del list2[index]

            if temp_vector is None:
                if select_nlist2 == 1:
                    del list1[index2]
                else:
                    del list2[index2]
            else:
                for ti in temp_vector:
                    n_list[ti] = None

                n_list[:] = [x for x in n_list if x is not None]

            if select_nlist == select_nlist2:
                print(
                    "WARNING in CompareReebGraph.look_for_matching_pair(): "
                    "select_nlist and select_nlist2 point to the same MRG!"
                )

            what_res_m = self.calculate_index_in_mrg1(m)
            what_res_n = self.calculate_index_in_mrg2(n)

            if m.children is not None and what_res_m > 0:
                vec_m = self.MRG1[what_res_m - 1]
                for z in range(len(m.children)):
                    temp_int = m.children[z]
                    tv = vec_m[temp_int]
                    temp_node = tv[0]

                    if temp_node not in list1:
                        list1.append(temp_node)

            if n.children is not None and what_res_n > 0:
                vec_n = self.MRG2[what_res_n - 1]
                for z in range(len(n.children)):
                    temp_int = n.children[z]
                    tv = vec_n[temp_int]
                    temp_node = tv[0]

                    if temp_node not in list2:
                        list2.append(temp_node)

            return

    def how_many_same_range_nodes_in_nlist(self, node, select_nlist):
        """Calculates the number of nodes that are connected to R-node
        node. NOTE: in the original Java, the returned counter is computed
        but its result is never actually used by the caller (the code that
        would have consumed it is commented out) -- this port keeps the
        call (and its side-effect-free computation) for fidelity, matching
        the original's dead computation."""
        counter = 1
        nlist = self.NLIST[0] if select_nlist == 1 else self.NLIST[1]

        vector_nodes = []
        for i in range(len(nlist)):
            temp_node = nlist[i]

            vector_nodes.append(i)

            if (
                temp_node.left_bound == node.left_bound
                and temp_node.right_bound == node.right_bound
                and self.is_connected_to_one_node(node, vector_nodes, nlist, select_nlist, 5)
            ):
                counter += 1
            else:
                vector_nodes.pop()

        return counter

    def create_match_candidates(self, node, n_list, match_candidates, candidates_count, select_nlist):
        """select_nlist=1 indicates that MRG2 has the largest sim(m,m)
        value; select_nlist=0 indicates that MRG1 has the largest sim(m,m)
        value."""
        # only takes a look at sequences that are no longer than 5 nodes
        if candidates_count > 6:
            return False

        # this is used so that a sequence with the maximum length can be taken into account
        if candidates_count == 6:
            node_index_in_nlist = -1
            element = MatchCandidate()

            for i in range(len(n_list)):
                temp_node = n_list[i]

                if (
                    temp_node.index == node.index
                    and temp_node.left_bound == node.left_bound
                    and temp_node.right_bound == node.right_bound
                    and temp_node.attribute.a == node.attribute.a
                    and temp_node.attribute.l == node.attribute.l
                ):
                    node_index_in_nlist = i
                    element.vector = [node_index_in_nlist]

            if node_index_in_nlist == -1:
                print(
                    "ERROR in CompareReebGraph.create_match_candidates(): can not find R-node "
                    "in n_list that match ranges of R-node node. Exiting... "
                )
                raise SystemExit(1)

            for i in range(len(n_list)):
                temp_node = n_list[i]

                if (
                    node_index_in_nlist != i
                    and temp_node.left_bound == node.left_bound
                    and temp_node.right_bound == node.right_bound
                ):
                    mrg_num = 2 if select_nlist == 1 else 1

                    if i not in element.vector and self.is_connected_to_one_node(
                        temp_node, element.vector, n_list, mrg_num, candidates_count
                    ):
                        element.vector.append(i)

            match_candidates.append(element)

            return len(element.vector) > 5

        # computes sequences that are 1 to 5 nodes long
        counter = 0
        stopper = False
        node_index_in_nlist = -1

        element = MatchCandidate()

        for i in range(len(n_list)):
            temp_node = n_list[i]

            if (
                temp_node.index == node.index
                and temp_node.left_bound == node.left_bound
                and temp_node.right_bound == node.right_bound
                and temp_node.attribute.a == node.attribute.a
                and temp_node.attribute.l == node.attribute.l
            ):
                node_index_in_nlist = i
                element.vector = [node_index_in_nlist]

        if node_index_in_nlist == -1:
            print(
                "ERROR in CompareReebGraph.create_match_candidates(): can not find R-node "
                "in n_list that match ranges of R-node node. Exiting... "
            )
            raise SystemExit(1)

        counter += 1
        match_candidates.append(element)

        if counter == candidates_count:
            return True

        found_one = False

        while not stopper:
            match_candidates.append(None)
            element = match_candidates.pop(0)

            found_one = False
            while element is not None:
                for i in range(len(n_list)):
                    temp_node = n_list[i]

                    tmp_int = element.vector[-1] if len(element.vector) > 1 else -1

                    if (
                        tmp_int < i
                        and node_index_in_nlist != i
                        and temp_node.left_bound == node.left_bound
                        and temp_node.right_bound == node.right_bound
                    ):
                        mrg_num = 2 if select_nlist == 1 else 1

                        if i not in element.vector and self.is_connected_to_one_node(
                            temp_node, element.vector, n_list, mrg_num, candidates_count
                        ):
                            new_el = MatchCandidate()
                            new_el.clone_element(element)
                            new_el.vector.append(i)

                            found_one = True

                            match_candidates.append(new_el)

                element = match_candidates.pop(0)

            if found_one:
                counter += 1
            else:
                stopper = True

            if counter == candidates_count:
                return True

        return False

    def is_connected_to_one_node(self, node, vector_nodes, n_list, what_mrg, candidates_count):
        """Checks that node and all R-nodes whose indices are stored in
        vector_nodes are connected to one R-node."""
        if vector_nodes is None:
            return False

        main_node = n_list[vector_nodes[0]]
        is_connected = True

        if what_mrg == 1:
            reeb_graph = self.MRG1[self.calculate_index_in_mrg1(main_node)]
        else:
            reeb_graph = self.MRG2[self.calculate_index_in_mrg2(main_node)]

        vec = reeb_graph[main_node.index]

        for i in range(1, len(vec)):
            temp_node = vec[i]

            for j in range(1, len(vector_nodes)):
                temp_node2 = n_list[vector_nodes[j]]
                temp_vec = reeb_graph[temp_node2.index]

                if temp_node not in temp_vec:
                    is_connected = False
                    break

            if is_connected:
                tmp = reeb_graph[node.index]
                if temp_node in tmp and node is not temp_node:
                    return True

            is_connected = True

        return False

    def find_maximum_sim(self):
        nlist1 = self.NLIST[0]
        nlist2 = self.NLIST[1]

        temp_node = nlist1[0]
        maximum_sim = self.sim(temp_node, temp_node)
        maximum_node = temp_node
        index = 0
        mrg_no = 1

        for i in range(1, len(nlist1)):
            temp_node = nlist1[i]
            temp = self.sim(temp_node, temp_node)

            if temp > maximum_sim:
                maximum_sim = temp
                maximum_node = temp_node
                mrg_no = 1
                index = i

        for i in range(len(nlist2)):
            temp_node = nlist2[i]
            temp = self.sim(temp_node, temp_node)

            if temp > maximum_sim:
                maximum_sim = temp
                maximum_node = temp_node
                mrg_no = 2
                index = i

        element = MPairElement()

        if mrg_no == 1:
            element.node1 = maximum_node
            element.node2 = None
        else:
            element.node1 = None
            element.node2 = maximum_node

        select_nlist = mrg_no

        return element, select_nlist, index

    # ------------------------------------------------------------------
    # similarity / loss functions (Section 5.2 in Hilaga et al.)
    # ------------------------------------------------------------------

    def parents_are_matched(self, m, n):
        """R-node m must be from MRG1, and R-node n is from MRG2."""
        if m.parent is None or n.parent is None:
            return True

        for el in self.MPAIR:
            if el.vector_nodes1 is None and el.vector_nodes2 is None:
                if m.parent is el.node1 and n.parent is el.node2:
                    return True
            else:
                if el.vector_nodes1 is not None:
                    if m.parent in el.vector_nodes1 and n.parent is el.node2:
                        return True
                else:
                    if n.parent in el.vector_nodes2 and m.parent is el.node1:
                        return True

        return False

    def compute_parents(self):
        # calculating parent for each RNode in MRG1
        for i in range(1, len(self.MRG1)):
            reeb_graph = self.MRG1[i]

            for j in range(len(reeb_graph)):
                element = reeb_graph[j][0]

                if i == len(self.MRG1) - 1:
                    element.parent = None

                if element.children is not None:
                    for k in range(len(element.children)):
                        temp_int = element.children[k]
                        next_reeb_graph = self.MRG1[i - 1]
                        element2 = next_reeb_graph[temp_int][0]
                        element2.parent = element

        # calculating parent for each RNode in MRG2
        for i in range(1, len(self.MRG2)):
            reeb_graph = self.MRG2[i]

            for j in range(len(reeb_graph)):
                element = reeb_graph[j][0]

                if element.children is not None:
                    for k in range(len(element.children)):
                        temp_int = element.children[k]
                        next_reeb_graph = self.MRG2[i - 1]
                        element2 = next_reeb_graph[temp_int][0]
                        element2.parent = element

    def mat(self, m, n):
        loss_m_n = -self.loss(m, n)

        range_length_m = m.right_bound - m.left_bound
        range_length_n = n.right_bound - n.left_bound

        adj_m1 = self.adj(m, 1, m.right_bound, m.right_bound + range_length_m)
        adj_m2 = self.adj(m, 1, m.left_bound - range_length_m, m.left_bound)

        adj_n1 = self.adj(n, 2, n.right_bound, n.right_bound + range_length_n)
        adj_n2 = self.adj(n, 2, n.left_bound - range_length_n, n.left_bound)

        loss_adj1 = self.loss(adj_m1, adj_n1)
        loss_adj2 = self.loss(adj_m2, adj_n2)

        return loss_m_n - (loss_adj1 + loss_adj2)

    def loss(self, m, n):
        return (0.5 * (self.sim(m, m) + self.sim(n, n))) - self.sim(m, n)

    def adj(self, m, what_mrg, left_b, right_b):
        if what_mrg == 1:
            reeb_graph = self.MRG1[self.calculate_index_in_mrg1(m)]
        elif what_mrg == 2:
            reeb_graph = self.MRG2[self.calculate_index_in_mrg2(m)]
        else:
            print("ERROR in CompareReebGraph.adj(): wrong MRG index! Exiting...")
            raise SystemExit(1)

        adj_vers = reeb_graph[m.index]

        result = RNode()
        result.attribute = AttributeElement()
        result.attribute.a = 0.0
        result.attribute.l = 0.0

        for i in range(1, len(adj_vers)):
            node = adj_vers[i]

            if node.left_bound == left_b or node.right_bound == right_b:
                result.attribute.a = result.attribute.a + node.attribute.a
                result.attribute.l = result.attribute.l + node.attribute.l

        return result

    def calculate_index_in_mrg1(self, m):
        range_length = m.right_bound - m.left_bound
        n = -(math.log(range_length)) / (math.log(2))

        result = int(n)
        return len(self.MRG1) - 1 - result

    def calculate_index_in_mrg2(self, m):
        range_length = m.right_bound - m.left_bound
        n = -(math.log(range_length)) / (math.log(2))

        result = int(n)
        return len(self.MRG2) - 1 - result

    def sim(self, m, n):
        min_a = n.attribute.a if m.attribute.a > n.attribute.a else m.attribute.a
        min_l = n.attribute.l if m.attribute.l > n.attribute.l else m.attribute.l

        return self.w * min_a + (1 - self.w) * min_l

    def calculate_rest_attributes(self, mrg, attributes):
        reeb_graph = mrg[0]

        for i in range(len(reeb_graph)):
            node = reeb_graph[i][0]

            node.attribute = AttributeElement()
            temp = attributes[i]
            node.attribute.a = temp.a
            node.attribute.l = temp.l

        for i in range(1, len(mrg)):
            reeb_graph = mrg[i]

            for j in range(len(reeb_graph)):
                node = reeb_graph[j][0]

                node.attribute = AttributeElement()
                node.attribute.a = 0.0
                node.attribute.l = 0.0
                prev_graph = mrg[i - 1]

                for k in range(len(node.children)):
                    temp_int = node.children[k]
                    adj_vec = prev_graph[temp_int]
                    temp_node = adj_vec[0]

                    node.attribute.a = node.attribute.a + temp_node.attribute.a
                    node.attribute.l = node.attribute.l + temp_node.attribute.l


def main(argv):
    points_number = int(argv[0])
    mu_coeff = float(argv[1])
    mrg_number = int(argv[2])
    w = float(argv[3])

    log_filename = "log_{}_{}_{}_{}".format(
        points_number, java_double_str(mu_coeff), mrg_number, java_double_str(w)
    )

    comparer = CompareReebGraph()
    comparer.w = w

    models = argv[4:]

    with open(log_filename, "w") as out:
        for arg_i in models:
            for arg_j in models:
                comparer.main_one(arg_i, arg_j)
                out.write(
                    "Similarity between " + arg_i + " and " + arg_j + " is "
                    + java_double_str(comparer.SIM_R_S) + "\n"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
