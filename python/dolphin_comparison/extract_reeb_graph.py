#!/usr/bin/env python3
"""Port of src/ExtractReebGraph.java.

The method for estimating similarity of 3D mesh models based on
multiresolutional Reeb graphs (MRG) was proposed by Hilaga et al. [1].

``extract_reeb_graph.py`` implements construction of MRGs for 3D mesh
models (see Section 4 in [1]) stored in (pseudo-) VRML format.

``compare_reeb_graph.py`` implements the matching algorithm for a pair of
MRGs (see Section 5 in [1]).

References:
    [1] Topology Matching for Fully Automatic Similarity Estimation of 3D
        Shapes. Masaki Hilaga, Yoshihisa Shinagawa, Taku Kohmura and
        Tosiyasu L. Kunii. SIGGRAPH, 2001.

Usage::

    python3 -m reeb_graph.extract_reeb_graph <num_pts> <mu_coeff> <mrg_size> \\
        <model_1>.wrl <model_2>.wrl ... <model_N>.wrl

    <num_pts>     -- target number of vertices before the MRG is extracted
                     (triangle meshes in each 3D model are resampled to
                     match <num_pts>)
    <mu_coeff>    -- coefficient for calculating threshold parameter
                     r=sqrt(mu_coeff * area(S)), used to approximate mu
                     values
    <mrg_size>    -- number of ranges in the finest resolution of MRG
                     (parameter K in [1])
    <model_i>.wrl -- i-th VRML model to process, for i=[1,N] (MRG for this
                     model is stored in text file <model_i>.mrg)
"""

import math
import sys

from . import ecomp
from .attribute_calculation import AttributeCalculation
from .mrg_io import save_mrg
from .mrg_constr_light import MRGConstrLight
from .mu_approx import MuApprox
from .mu_normalization import MuNormalization
from .resample import Resample
from .sparse_matrix import SparseMatrix
from .vrml_parser import parse_vrml


def calculate_trig_area(A, B, C):
    a = math.sqrt((C.X - B.X) ** 2 + (C.Y - B.Y) ** 2 + (C.Z - B.Z) ** 2)
    b = math.sqrt((A.X - C.X) ** 2 + (A.Y - C.Y) ** 2 + (A.Z - C.Z) ** 2)
    c = math.sqrt((A.X - B.X) ** 2 + (A.Y - B.Y) ** 2 + (A.Z - B.Z) ** 2)

    p = (a + b + c) / 2

    area = p * (p - a) * (p - b) * (p - c)

    if area < 0:
        return 0.0

    return math.sqrt(area)


def calculate_whole_area(triangles, points):
    area = 0.0
    for t in triangles:
        area = area + calculate_trig_area(points[t.a], points[t.b], points[t.c])
    return area


def main_one(filename, points_number, mu_coeff, mrg_number):
    """Computes the MRG for a single VRML model and saves it to
    ``<filename minus '.wrl'>.mrg``."""
    print("Working on " + filename + " ...")

    points, triangles = parse_vrml(filename)
    points_length = len(points)
    triangles_length = len(triangles)

    # Calculate epsilon
    eps = 1000000000.0
    for i in range(points_length):
        a_pt = points[i]

        if eps > abs(a_pt.X) and abs(a_pt.X) != 0.0:
            eps = abs(a_pt.X)
        if eps > abs(a_pt.Y) and abs(a_pt.Y) != 0.0:
            eps = abs(a_pt.Y)
        if eps > abs(a_pt.Z) and abs(a_pt.Z) != 0.0:
            eps = abs(a_pt.Z)

    ecomp.epsilon = abs(eps / 1000.0)

    # calculates area of the model
    whole_area = calculate_whole_area(triangles, points)

    # create connectivity matrix (i.e., adjacency graph) for the mesh model
    sparsem = SparseMatrix()
    points, points_length, sparse = sparsem.create_matrix(triangles, triangles_length, points, points_length)

    # resample faces until the needed number of vertices is reached
    print("Resampling faces...")
    resample = Resample()
    points, points_length, sparse = resample.do_process(points_number, points, points_length, sparse)
    print("After resampling, points_length = " + str(points_length))

    # calculates mu values
    print("Calculating mu values...")
    mu_val = MuApprox()
    mu_values = mu_val.do_process(mu_coeff, whole_area, points, points_length, sparse)

    # normalize mu values
    mu_norm = MuNormalization()
    mu_values = mu_norm.normalize(mu_values)

    # construct MRG
    print("Constructing MRG...")
    mrg = MRGConstrLight()
    points, points_length, sparse, mu_values, all_tsets, MRG, reebs = mrg.do_process(
        mrg_number, points, points_length, sparse, mu_values, False
    )

    # calculate attributes for the nodes in MRG
    print("Calculating attributes for Tsets...")
    a_calc = AttributeCalculation()
    attributes = a_calc.do_process(points, points_length, sparse, mu_values, all_tsets, MRG, reebs, whole_area)

    # write MRG into a '.mrg' text file
    print("Saving MRG to a file...")
    save_mrg(filename, MRG, reebs, attributes, mrg.FINEST_RESOLUTION)


def main(argv):
    points_number = int(argv[0])
    mu_coeff = float(argv[1])
    mrg_number = int(argv[2])

    for filename in argv[3:]:
        main_one(filename, points_number, mu_coeff, mrg_number)


if __name__ == "__main__":
    main(sys.argv[1:])
