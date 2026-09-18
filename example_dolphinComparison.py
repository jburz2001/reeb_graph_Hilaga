import tempfile
from pathlib import Path

import numpy as np
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkTriangle, vtkUnstructuredGrid

from dolphin_comparison.point import Point
from dolphin_comparison.triangle import Triangle
from dolphin_comparison.sparse_matrix import SparseMatrix
from dolphin_comparison.mu_normalization import MuNormalization
from dolphin_comparison.mrg_constr_light import MRGConstrLight
from dolphin_comparison.attribute_calculation import AttributeCalculation
from dolphin_comparison.extract_reeb_graph import calculate_whole_area
from dolphin_comparison.mrg_io import save_mrg
from dolphin_comparison.compare_reeb_graph import CompareReebGraph


def _numpy_array_to_vtk_grid(array):
    rows, cols = array.shape

    def point_id(row, col):
        return row * cols + col

    grid = vtkUnstructuredGrid()
    points = vtkPoints()
    points.SetNumberOfPoints(rows * cols)
    for row in range(rows):
        for col in range(cols):
            points.SetPoint(point_id(row, col), float(col), float(row), 0.0)
    grid.SetPoints(points)

    for row in range(rows - 1):
        for col in range(cols - 1):
            p00, p10 = point_id(row, col), point_id(row, col + 1)
            p01, p11 = point_id(row + 1, col), point_id(row + 1, col + 1)
            for a, b, c in ((p00, p10, p11), (p00, p11, p01)):
                triangle = vtkTriangle()
                triangle.GetPointIds().SetId(0, a)
                triangle.GetPointIds().SetId(1, b)
                triangle.GetPointIds().SetId(2, c)
                grid.InsertNextCell(triangle.GetCellType(), triangle.GetPointIds())

    return grid


def _vtk_grid_to_points_and_triangles(grid):
    vtk_points = grid.GetPoints()
    points = [Point(*vtk_points.GetPoint(i)) for i in range(grid.GetNumberOfPoints())]

    triangles = []
    for cell_index in range(grid.GetNumberOfCells()):
        ids = grid.GetCell(cell_index).GetPointIds()
        triangles.append(Triangle(ids.GetId(0), ids.GetId(1), ids.GetId(2)))

    return points, triangles


def _build_mrg_from_array(array, mrg_size, label, output_dir):
    grid = _numpy_array_to_vtk_grid(array)
    points, triangles = _vtk_grid_to_points_and_triangles(grid)

    sparse_matrix = SparseMatrix()
    points, points_length, sparse = sparse_matrix.create_matrix(
        triangles, len(triangles), points, len(points)
    )

    shifted = array - array.min()
    mu_values = MuNormalization().normalize(list(shifted.ravel(order="C")))
    whole_area = calculate_whole_area(triangles, points)

    mrg = MRGConstrLight()
    points, points_length, sparse, mu_values, all_tsets, MRG, reebs = mrg.do_process(
        mrg_size, points, points_length, sparse, mu_values, False
    )
    for tset in all_tsets:
        tset.sort(reverse=True)

    attributes = AttributeCalculation().do_process(
        points, points_length, sparse, mu_values, all_tsets, MRG, reebs, whole_area
    )

    wrl_path = output_dir / f"{label}.wrl"
    save_mrg(str(wrl_path), MRG, reebs, attributes, mrg.FINEST_RESOLUTION)
    return str(wrl_path)


def compare_arrays(array_a, array_b, mrg_size=8, sim_weight=0.5):
    array_a = np.asarray(array_a, dtype=np.float64)
    array_b = np.asarray(array_b, dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        path_a = _build_mrg_from_array(array_a, mrg_size, "array_a", output_dir)
        path_b = _build_mrg_from_array(array_b, mrg_size, "array_b", output_dir)

        comparer = CompareReebGraph()
        comparer.w = sim_weight
        comparer.main_one(path_a, path_b)
        return comparer.SIM_R_S

