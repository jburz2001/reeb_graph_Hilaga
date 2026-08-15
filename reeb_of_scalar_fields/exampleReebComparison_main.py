#!/usr/bin/env python3

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

# TODO in production, remove .use("Agg") necessity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import orbithunter as orb

from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkImageData
from vtkmodules.vtkFiltersCore import vtkContourFilter
from topologytoolkit import ttkPersistenceDiagram, ttkTopologicalSimplification
from vtkmodules.vtkFiltersCore import vtkThreshold
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

from exampleReebComparison_miscellaneous import *

FPO_Path = Path(__file__).parent.parent / "inputs" / "fundamental_periodic_orbits.h5"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "orbit_0_copy"

SCALAR_NAME = "u"

DEFAULT_ARC_SAMPLING = 20
DEFAULT_ROLL_T = 128
DEFAULT_ROLL_X = 170
DEFAULT_DEGENERACY_TOL = 1e-10
DOMAIN_PERIODIC = "periodic"
DOMAIN_OPEN = "open"




def point_id(t_index: int, x_index: int, x_count: int) -> int:
    return t_index * x_count + x_index


def _periodic_triangles(field_shape: tuple[int, int]):
    """Yield the two triangles (as point-id triples) for every periodic grid cell."""
    t_count, x_count = field_shape
    for t_index in range(t_count):
        t_next = (t_index + 1) % t_count
        for x_index in range(x_count):
            x_next = (x_index + 1) % x_count
            p00 = point_id(t_index, x_index, x_count)
            p10 = point_id(t_index, x_next, x_count)
            p01 = point_id(t_next, x_index, x_count)
            p11 = point_id(t_next, x_next, x_count)
            yield p00, p10, p11
            yield p00, p11, p01


def periodic_grid_neighbors(field_shape: tuple[int, int]) -> dict[int, list[int]]:
    """Neighbors induced by the periodic triangulation used for TTK."""
    neighbors = defaultdict(set)
    for a, b, c in _periodic_triangles(field_shape):
        for u, v in ((a, b), (b, c), (c, a)):
            neighbors[u].add(v)
            neighbors[v].add(u)
    return {node: sorted(values) for node, values in neighbors.items()}


def periodic_grid_link_edges(field_shape: tuple[int, int]) -> dict[int, list[tuple[int, int]]]:
    """Link edges around each vertex for the periodic triangulation."""
    link_edges = defaultdict(set)
    for a, b, c in _periodic_triangles(field_shape):
        for vertex, e0, e1 in ((a, b, c), (b, a, c), (c, a, b)):
            link_edges[vertex].add(tuple(sorted((e0, e1))))
    return {node: sorted(values) for node, values in link_edges.items()}


def _connected_components(nodes, edges) -> list[list]:
    """BFS connected components of `nodes`, restricted to edges within `nodes`."""
    if not nodes:
        return []

    adjacency = {node: set() for node in nodes}
    for source, target in edges:
        if source in adjacency and target in adjacency:
            adjacency[source].add(target)
            adjacency[target].add(source)

    components = []
    seen = set()
    for node in nodes:
        if node in seen:
            continue
        component = []
        queue = deque([node])
        seen.add(node)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def count_components(nodes: set[int], edges: list[tuple[int, int]]) -> int:
    return len(_connected_components(nodes, edges))


def connected_components(nodes, edges) -> list[list]:
    return _connected_components(nodes, edges)


def pl_link_classification(field: np.ndarray, eps: float = 0.0) -> dict[int, dict]:
    """Classify vertices by lower/upper link components on the periodic PL mesh."""
    field = np.asarray(field, dtype=np.float64)
    flat = field.ravel(order="C")
    neighbors = periodic_grid_neighbors(field.shape)
    link_edges = periodic_grid_link_edges(field.shape)

    type_and_label_by_components = {
        (0, 1): (0, "minimum"),
        (1, 0): (3, "maximum"),
        (1, 1): (None, "regular"),
        (2, 2): (2, "simple saddle"),
    }

    classifications = {}
    for vertex, adjacent in neighbors.items():
        value = flat[vertex]
        lower = {node for node in adjacent if flat[node] < value - eps}
        upper = {node for node in adjacent if flat[node] > value + eps}
        equal = set(adjacent) - lower - upper
        lower_components = count_components(lower, link_edges[vertex])
        upper_components = count_components(upper, link_edges[vertex])

        critical_type, label = type_and_label_by_components.get(
            (lower_components, upper_components), (4, "degenerate saddle")
        )

        classifications[vertex] = {
            "pl_lower_components": lower_components,
            "pl_upper_components": upper_components,
            "pl_equal_neighbors": len(equal),
            "pl_critical_type": critical_type,
            "pl_label": label,
        }

    return classifications



def extract_zero_level_set(image):
    """Extract the u = 0 contour for saving/reference."""
    contour = vtkContourFilter()
    contour.SetInputData(image)
    contour.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, SCALAR_NAME)
    contour.SetValue(0, 0.0)
    contour.Update()
    return contour.GetOutput()


def contour_paths(field: np.ndarray, level: float = 0.0) -> list[np.ndarray]:
    """Extract contour paths as Nx2 arrays in x-t coordinates using matplotlib."""
    t_count, x_count = field.shape
    fig, ax = plt.subplots()
    contour = ax.contour(np.arange(x_count), np.arange(t_count), field, levels=[level])
    if hasattr(contour, "collections"):
        paths = [path.vertices.copy() for path in contour.collections[0].get_paths()]
    else:
        paths = [np.asarray(segment).copy() for segment in contour.allsegs[0]]
    plt.close(fig)
    return paths


def periodic_contour_paths(field: np.ndarray, level: float = 0.0) -> list[np.ndarray]:
    """Extract contours from a 3x3 tiled field and keep the central periodic window."""
    t_count, x_count = field.shape
    tiled = np.tile(field, (3, 3))
    paths = contour_paths(tiled, level=level)

    shifted_paths = []
    for path in paths:
        shifted = path.copy()
        shifted[:, 0] -= x_count
        shifted[:, 1] -= t_count
        shifted_paths.append(shifted)

    return clip_paths_to_window(shifted_paths, field.shape)


def clip_paths_to_window(paths: list[np.ndarray], field_shape: tuple[int, int]) -> list[np.ndarray]:
    """Keep only contiguous path segments inside the displayed domain."""
    t_count, x_count = field_shape
    bounds = (0.0, float(x_count - 1), 0.0, float(t_count - 1))
    clipped_paths = []
    for path in paths:
        clipped_paths.extend(clip_polyline_to_rect(path, bounds))
    return clipped_paths


def clip_segment_to_rect(p0: np.ndarray, p1: np.ndarray, bounds: tuple[float, float, float, float]):
    """Clip one line segment to xmin/xmax/ymin/ymax using Liang-Barsky."""
    xmin, xmax, ymin, ymax = bounds
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    t0, t1 = 0.0, 1.0

    for p, q in (
        (-dx, p0[0] - xmin),
        (dx, xmax - p0[0]),
        (-dy, p0[1] - ymin),
        (dy, ymax - p0[1]),
    ):
        if abs(p) < 1e-14:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)

    return p0 + t0 * (p1 - p0), p0 + t1 * (p1 - p0)


def clip_polyline_to_rect(path: np.ndarray, bounds: tuple[float, float, float, float]) -> list[np.ndarray]:
    """Clip a polyline to a rectangle while preserving boundary intersection points."""
    if len(path) < 2:
        return []

    segments = []
    current = []
    previous_end = None

    for start, end in zip(path[:-1], path[1:]):
        clipped = clip_segment_to_rect(start, end, bounds)
        if clipped is None:
            if len(current) >= 2:
                segments.append(np.asarray(current))
            current = []
            previous_end = None
            continue

        clipped_start, clipped_end = clipped
        if previous_end is None or np.linalg.norm(clipped_start - previous_end) > 1e-8:
            if len(current) >= 2:
                segments.append(np.asarray(current))
            current = [clipped_start]
        current.append(clipped_end)
        previous_end = clipped_end

    if len(current) >= 2:
        segments.append(np.asarray(current))

    return segments


def roll_paths(paths: list[np.ndarray], roll_t: int, roll_x: int, field_shape: tuple[int, int]) -> list[np.ndarray]:
    """Apply the same displacement as np.roll, shown in the central periodic window."""
    t_count, x_count = field_shape
    shifted = []
    for path in paths:
        for t_copy in (-t_count, 0, t_count):
            for x_copy in (-x_count, 0, x_count):
                rolled_path = path.copy()
                rolled_path[:, 0] += roll_x + x_copy
                rolled_path[:, 1] += roll_t + t_copy
                shifted.append(rolled_path)
    return clip_paths_to_window(shifted, field_shape)


def unroll_paths(paths: list[np.ndarray], roll_t: int, roll_x: int, field_shape: tuple[int, int]) -> list[np.ndarray]:
    """Map contours from rolled-field coordinates back into original coordinates."""
    return roll_paths(paths, -roll_t, -roll_x, field_shape)



def compute_reeb_graph(image, arc_sampling: int = 20, with_segmentation: bool = False):
    """Run TTK's Reeb graph filter on scalar field u."""
    ttkReebGraph = import_ttk_reeb_graph()
    reeb = ttkReebGraph()
    reeb.SetInputData(image)
    reeb.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, SCALAR_NAME)

    for setter_name in ("SetSampling", "SetArcSampling"):
        if hasattr(reeb, setter_name):
            getattr(reeb, setter_name)(arc_sampling)
            break

    if hasattr(reeb, "SetWithSegmentation"):
        reeb.SetWithSegmentation(with_segmentation)

    start = log_step("Running TTK Reeb graph...")
    reeb.Update()
    log_done(start)
    return [reeb.GetOutputDataObject(i) for i in range(reeb.GetNumberOfOutputPorts())]


def simplify_field_by_persistence(dataset, persistence_threshold: float):
    diagram_filter = ttkPersistenceDiagram()
    diagram_filter.SetInputData(dataset)
    diagram_filter.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, SCALAR_NAME)
    diagram_filter.Update()
    diagram = diagram_filter.GetOutput()

    threshold = vtkThreshold()
    threshold.SetInputData(diagram)
    threshold.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, "Persistence")
    threshold.SetUpperThreshold(persistence_threshold)
    threshold.SetThresholdFunction(vtkThreshold.THRESHOLD_UPPER)
    threshold.Update()

    kept = threshold.GetOutput().GetNumberOfCells()
    total = diagram.GetNumberOfCells()
    # print(f"Persistence simplification: kept {kept}/{total} pairs at threshold {persistence_threshold:g}", flush=True)

    simplification = ttkTopologicalSimplification()
    simplification.SetInputData(0, dataset)
    simplification.SetInputData(1, threshold.GetOutput())
    simplification.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, SCALAR_NAME)
    simplification.Update()
    return simplification.GetOutput()


def compute_field_reeb_outputs(field, arc_sampling=20, with_segmentation=False, domain_mode=DOMAIN_PERIODIC, persistence_threshold=None):
    image = field_to_vtk_dataset(field, domain_mode=domain_mode)
    if persistence_threshold is not None:
        image = simplify_field_by_persistence(image, persistence_threshold)
    zero_level_set = extract_zero_level_set(image)
    reeb_outputs = compute_reeb_graph(image, arc_sampling=arc_sampling, with_segmentation=with_segmentation)
    return image, zero_level_set, reeb_outputs



def reeb_arcs_output(reeb_outputs):
    """Return the TTK output that contains Reeb arcs."""
    if len(reeb_outputs) > 1 and reeb_outputs[1] is not None:
        return reeb_outputs[1]
    for output in reeb_outputs:
        if output is not None and hasattr(output, "GetNumberOfCells") and output.GetNumberOfCells() > 0:
            return output
    return None


def iter_datasets_with_output_index(reeb_outputs):
    for output_index, output in enumerate(reeb_outputs):
        for dataset in iter_leaf_datasets(output):
            yield output_index, dataset


def reeb_graph_to_networkx(
    field: np.ndarray,
    reeb_outputs,
    domain_mode: str = DOMAIN_PERIODIC,
    base_shape: tuple[int, int] | None = None,
    pl_classification: dict[int, dict] | None = None,
):
    """Convert TTK Reeb arc output to a coordinate-free NetworkX MultiGraph."""

    graph = nx.MultiGraph()
    if graph_from_any_ttk_arc_node_ids(graph, reeb_outputs):
        return graph

    if graph_from_arc_ids_and_critical_nodes(
        graph,
        field,
        reeb_outputs,
        domain_mode=domain_mode,
        base_shape=base_shape or field.shape,
        pl_classification=pl_classification,
    ):
        return graph

    reeb_arcs = reeb_arcs_output(reeb_outputs)
    if reeb_arcs is None:
        return graph

    # print("TTK arc node ID arrays not found; falling back to geometric endpoint inference.", flush=True)
    endpoint_to_node = {}

    def node_for_point(point):
        key = tuple(np.round(point[:2], 6))
        if key not in endpoint_to_node:
            node_id = len(endpoint_to_node)
            endpoint_to_node[key] = node_id
            graph.add_node(node_id, scalar_bin=round(nearest_field_value(field, point[:2]), 10))
        return endpoint_to_node[key]

    for leaf in iter_leaf_datasets(reeb_arcs):
        points = leaf.GetPoints()
        if points is None:
            continue

        for cell_index in range(leaf.GetNumberOfCells()):
            point_ids = leaf.GetCell(cell_index).GetPointIds()
            if point_ids.GetNumberOfIds() < 2:
                continue

            start_point = points.GetPoint(point_ids.GetId(0))
            end_point = points.GetPoint(point_ids.GetId(point_ids.GetNumberOfIds() - 1))
            source = node_for_point(start_point)
            target = node_for_point(end_point)
            if source != target:
                graph.add_edge(source, target)

    return graph


def vtk_array_tuple1(array, index: int) -> int:
    return int(array.GetTuple1(index))


def graph_from_ttk_arc_node_ids(graph, reeb_arcs) -> bool:
    """Populate graph using TTK's arc upNodeId/downNodeId arrays when present."""
    cell_data = reeb_arcs.GetCellData() if hasattr(reeb_arcs, "GetCellData") else None
    if cell_data is None:
        return False

    up, down, up_name, down_name = find_arc_endpoint_arrays(cell_data)
    if up is None or down is None:
        return False

    for cell_index in range(reeb_arcs.GetNumberOfCells()):
        source = vtk_array_tuple1(up, cell_index)
        target = vtk_array_tuple1(down, cell_index)
        graph.add_node(source)
        graph.add_node(target)
        if source != target:
            graph.add_edge(source, target)

    # print(f"Built NetworkX Reeb graph from TTK {up_name}/{down_name} arrays.", flush=True)
    return True


def graph_from_any_ttk_arc_node_ids(graph, reeb_outputs) -> bool:
    """Search all TTK outputs for arc endpoint arrays."""
    inspect_ttk_outputs(reeb_outputs)
    for output_index, dataset in iter_datasets_with_output_index(reeb_outputs):
        if graph_from_ttk_arc_node_ids(graph, dataset):
            # print(f"Using TTK output port {output_index} for Reeb graph connectivity.", flush=True)
            return True
    return False


def graph_from_arc_ids_and_critical_nodes(
    graph,
    field: np.ndarray,
    reeb_outputs,
    domain_mode: str = DOMAIN_PERIODIC,
    base_shape: tuple[int, int] | None = None,
    pl_classification: dict[int, dict] | None = None,
) -> bool:
    """Build graph from output-0 critical nodes and output-1 ArcId grouped arc geometry."""
    if len(reeb_outputs) < 2 or reeb_outputs[0] is None or reeb_outputs[1] is None:
        return False

    node_dataset, arc_dataset = reeb_outputs[0], reeb_outputs[1]
    if node_dataset.GetNumberOfPoints() == 0 or arc_dataset.GetNumberOfCells() == 0:
        return False

    arc_ids = arc_dataset.GetCellData().GetArray("ArcId")
    node_points, arc_points = node_dataset.GetPoints(), arc_dataset.GetPoints()
    if arc_ids is None or node_points is None or arc_points is None:
        return False

    base_shape = base_shape or field.shape
    node_labels = critical_node_labels(node_dataset)
    node_attributes = critical_node_attributes(node_dataset, base_shape)
    label_set = set(node_labels)
    label_to_point_index = {label: index for index, label in enumerate(node_labels)}
    for label in node_labels:
        graph.add_node(label, **node_attributes.get(label, {}))

    cells_by_arc = defaultdict(list)
    for cell_index in range(arc_dataset.GetNumberOfCells()):
        arc_id = vtk_array_tuple1(arc_ids, cell_index)
        point_ids = arc_dataset.GetCell(cell_index).GetPointIds()
        if point_ids.GetNumberOfIds() >= 2:
            cells_by_arc[arc_id].append([point_ids.GetId(i) for i in range(point_ids.GetNumberOfIds())])

    added_edges = 0
    for arc_id, cells in cells_by_arc.items():
        endpoints = arc_group_endpoint_point_ids(cells)
        if len(endpoints) != 2:
            # print(f"ArcId {arc_id} has {len(endpoints)} endpoints; skipping ambiguous arc.", flush=True)
            continue

        source = nearest_critical_node_label(
            arc_points.GetPoint(endpoints[0])[:2], node_points, node_labels, base_shape, label_to_point_index
        )
        target = nearest_critical_node_label(
            arc_points.GetPoint(endpoints[1])[:2], node_points, node_labels, base_shape, label_to_point_index
        )
        if source in label_set and target in label_set and source != target:
            graph.add_edge(source, target, arc_id=arc_id)
            added_edges += 1

    # print(
    #     f"Built NetworkX Reeb graph from ArcId groups snapped to critical-node output "
    #     f"({graph.number_of_nodes()} nodes, {added_edges} edges).",
    #     flush=True,
    # )
    apply_pl_classification_to_graph(graph, pl_classification)
    resolve_pl_degenerate_saddles(graph)
    return added_edges > 0




def critical_node_labels(node_dataset) -> list[int]:
    vertex_ids = node_dataset.GetPointData().GetArray("VertexId")
    if vertex_ids is not None:
        return [vtk_array_tuple1(vertex_ids, index) for index in range(node_dataset.GetNumberOfPoints())]
    return list(range(node_dataset.GetNumberOfPoints()))


def _periodic_wrap(value: float, count: int) -> float:
    return float(np.mod(value, count))


def critical_node_attributes(node_dataset, field_shape: tuple[int, int]) -> dict[int, dict]:
    vertex_ids = node_dataset.GetPointData().GetArray("VertexId")
    critical_types = node_dataset.GetPointData().GetArray("CriticalType")
    scalars = node_dataset.GetPointData().GetArray("Scalar")
    points = node_dataset.GetPoints()

    attributes = {}
    for index in range(node_dataset.GetNumberOfPoints()):
        label = vtk_array_tuple1(vertex_ids, index) if vertex_ids is not None else index
        point = points.GetPoint(index)[:2] if points is not None else (np.nan, np.nan)
        attributes[label] = {
            "critical_type": vtk_array_tuple1(critical_types, index) if critical_types is not None else None,
            "scalar": float(scalars.GetTuple1(index)) if scalars is not None else np.nan,
            "x": _periodic_wrap(point[0], field_shape[1]),
            "t": _periodic_wrap(point[1], field_shape[0]),
        }
    return attributes


def apply_pl_classification_to_graph(graph, pl_classification: dict[int, dict] | None) -> None:
    for node, data in graph.nodes(data=True):
        raw_type = data.get("critical_type")
        data["raw_critical_type"] = raw_type
        pl_data = pl_classification.get(int(node)) if pl_classification else None
        if pl_data is not None:
            data.update(pl_data)
            pl_type = pl_data.get("pl_critical_type")
            if pl_type is not None:
                data["critical_type_normalized"] = raw_type if pl_data.get("pl_label") == "simple saddle" else pl_type
                continue

        data["critical_type_normalized"] = raw_type if raw_type != 4 else saddle_type_from_neighbors(graph, node)


def saddle_type_from_neighbors(graph, node):
    data = graph.nodes[node]
    neighbor_scalars = [
        graph.nodes[neighbor].get("scalar")
        for neighbor in graph.neighbors(node)
        if graph.nodes[neighbor].get("scalar") is not None
    ]
    scalar = data.get("scalar")
    if scalar is None or not neighbor_scalars:
        return 1

    lower = sum(value < scalar for value in neighbor_scalars)
    upper = sum(value > scalar for value in neighbor_scalars)
    return 2 if lower > upper else 1


def resolve_pl_degenerate_saddles(graph) -> None:
    """Split high-valence PL-degenerate saddles into deterministic simple saddle chains."""
    for node in list(graph.nodes):
        data = graph.nodes[node]
        if data.get("pl_critical_type", data.get("critical_type_normalized")) != 4:
            continue

        degree = graph.degree(node)
        if degree <= 3:
            data["critical_type_normalized"] = saddle_type_from_neighbors(graph, node)
            data["pl_resolved"] = "relabel_degree_leq_3"
            continue

        scalar = data.get("scalar")
        neighbors = sorted(
            list(graph.neighbors(node)),
            key=lambda n: (
                graph.nodes[n].get("scalar", 0.0) >= scalar,
                graph.nodes[n].get("scalar", 0.0),
                n,
            ),
        )
        lower_neighbors = [n for n in neighbors if graph.nodes[n].get("scalar", 0.0) < scalar]
        upper_neighbors = [n for n in neighbors if graph.nodes[n].get("scalar", 0.0) >= scalar]
        primary = lower_neighbors if len(lower_neighbors) >= len(upper_neighbors) else upper_neighbors
        secondary = upper_neighbors if primary is lower_neighbors else lower_neighbors
        normalized_type = 2 if primary is lower_neighbors else 1

        if len(primary) <= 2 or not secondary:
            data["critical_type_normalized"] = saddle_type_from_neighbors(graph, node)
            data["pl_resolved"] = "relabel_unsplittable"
            continue

        graph.remove_node(node)

        saddle_nodes = []
        for index in range(len(primary) - 1):
            new_node = (node, "pl_split", index)
            attrs = data.copy()
            attrs["critical_type_normalized"] = normalized_type
            attrs["pl_resolved"] = "split"
            attrs["scalar"] = scalar + 1e-12 * index
            graph.add_node(new_node, **attrs)
            saddle_nodes.append(new_node)

        for left, right in zip(saddle_nodes[:-1], saddle_nodes[1:]):
            graph.add_edge(left, right, pl_resolution_edge=True)

        for group in (primary, secondary):
            for index, neighbor in enumerate(group):
                target = saddle_nodes[min(index, len(saddle_nodes) - 1)]
                graph.add_edge(target, neighbor, pl_resolved_from=node)


def arc_group_endpoint_point_ids(cells: list[list[int]]) -> list[int]:
    degree = defaultdict(int)
    for cell_point_ids in cells:
        for left, right in zip(cell_point_ids[:-1], cell_point_ids[1:]):
            degree[left] += 1
            degree[right] += 1
    return [point_id_ for point_id_, point_degree in degree.items() if point_degree == 1]


def nearest_critical_node_label(
    point,
    node_points,
    node_labels: list[int],
    field_shape: tuple[int, int],
    label_to_point_index: dict[int, int],
) -> int:
    t_count, x_count = field_shape
    best_label = node_labels[0]
    best_distance = np.inf
    for label in node_labels:
        index = label_to_point_index.get(label)
        if index is None:
            continue
        node_point = node_points.GetPoint(index)[:2]
        dx = abs(((point[0] - node_point[0] + x_count / 2.0) % x_count) - x_count / 2.0)
        dy = abs(((point[1] - node_point[1] + t_count / 2.0) % t_count) - t_count / 2.0)
        distance = dx * dx + dy * dy
        if distance < best_distance:
            best_distance = distance
            best_label = label
    return best_label


def find_arc_endpoint_arrays(attributes):
    """Locate an up/down (or equivalent) node-id array pair, case-insensitively."""
    names = vtk_data_array_names(attributes)
    name_lookup = {name.lower(): name for name in names if name is not None}

    # Matching against name_lookup is already case-insensitive, so each pair
    # below covers every casing TTK might emit without listing variants.
    candidate_pairs = (
        ("upNodeId", "downNodeId"),
        ("upNodeId_", "downNodeId_"),
        ("NodeId0", "NodeId1"),
        ("VertexId0", "VertexId1"),
        ("SourceId", "DestinationId"),
    )

    for up_name, down_name in candidate_pairs:
        actual_up = name_lookup.get(up_name.lower())
        actual_down = name_lookup.get(down_name.lower())
        if actual_up is not None and actual_down is not None:
            return attributes.GetArray(actual_up), attributes.GetArray(actual_down), actual_up, actual_down

    node_like = [name for name in names if name and "node" in name.lower()]
    if len(node_like) >= 2:
        return attributes.GetArray(node_like[0]), attributes.GetArray(node_like[1]), node_like[0], node_like[1]

    return None, None, None, None



def vtk_data_array_names(attributes) -> list[str]:
    return [
        (attributes.GetArray(index).GetName() if attributes.GetArray(index) is not None else f"<array {index}>")
        for index in range(attributes.GetNumberOfArrays())
    ]

def critical_type_style(critical_type):
    """Return a stable label/color/marker for TTK critical point classes."""
    styles = {
        0: ("minimum", "#2c7fb8", "v"),
        1: ("1-saddle", "#fdae61", "s"),
        2: ("2-saddle", "#d7191c", "D"),
        3: ("maximum", "#7b3294", "^"),
        4: ("degenerate saddle", "#1a9850", "P"),
        5: ("regular/degenerate", "#999999", "o"),
    }
    return styles.get(critical_type, (f"type {critical_type}", "#4d4d4d", "o"))



def contract_degree_two_vertices(graph):
    skeleton = nx.Graph(graph)
    changed = True
    while changed:
        changed = False
        for node in list(skeleton.nodes):
            if skeleton.degree(node) != 2:
                continue
            neighbors = list(skeleton.neighbors(node))
            if len(neighbors) != 2:
                continue

            left, right = neighbors
            skeleton.remove_node(node)
            if left != right:
                skeleton.add_edge(left, right)
            changed = True
            break

    return skeleton


def compare_networkx_graphs(original_graph, rolled_graph):
    original_skeleton = contract_degree_two_vertices(original_graph)
    rolled_skeleton = contract_degree_two_vertices(rolled_graph)

    node_match = nx.algorithms.isomorphism.categorical_node_match("critical_type_normalized", None)
    isomorphic = nx.is_isomorphic(original_skeleton, rolled_skeleton, node_match=node_match)

    original_hash = weisfeiler_lehman_graph_hash(
        original_skeleton, node_attr="critical_type_normalized", iterations=8
    )
    rolled_hash = weisfeiler_lehman_graph_hash(
        rolled_skeleton, node_attr="critical_type_normalized", iterations=8
    )

    return isomorphic, original_hash, rolled_hash, original_skeleton, rolled_skeleton


def get_nx_graph_and_reeb_outputs(field, domain_mode, eps, persistence_threshold, arc_sampling, with_segmentation):
    if domain_mode == DOMAIN_PERIODIC:
        pl_original = pl_link_classification(field, eps)
    else:
        pl_original = None

    _, _, outputs = compute_field_reeb_outputs(
        field, 
        arc_sampling=arc_sampling,
        with_segmentation=with_segmentation,
        domain_mode=domain_mode,
        persistence_threshold=persistence_threshold
    )
    graph = reeb_graph_to_networkx(
        field,
        outputs,
        domain_mode=domain_mode,
        base_shape=field.shape,
        pl_classification=pl_original
    )

    return graph, outputs




def run(
    arc_sampling: int = 20,
    with_segmentation: bool = False,
    roll_t: int = DEFAULT_ROLL_T,
    roll_x: int = DEFAULT_ROLL_X,
    periodic_domain: bool = True,
):

    orbit_index = 0
    OUTPUT_DIR = Path(__file__).parent.parent / "results" / f"orbit_{orbit_index}_copy"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    # 1
    print("Loading orbit from Orbithunter to Numpy array")
    # FPO_Path = Path(__file__).parent.parent / "inputs" / "fundamental_periodic_orbits.h5"
    FPO_Path = Path(__file__).parent.parent / "inputs" / "pitchfork_10_iterations.h5"
    orbits = orb.io.read_h5(str(FPO_Path))
    orbit = orbits[orbit_index]
    field_orbit = orbit.resize(512, 512).transform(to="field")
    field = np.asarray(field_orbit.state, dtype=np.float64)


    # # 2
    # # https://vtk.org/doc/nightly/html/classvtkImageData.html
    # print("Converting Numpy array to VTK Image")
    # nt, nx = field.shape
    # image = vtkImageData()
    # image.SetDimensions(nx, nt, 1)
    # image.SetOrigin(0.0, 0.0, 0.0)
    # image.SetSpacing(1.0, 1.0, 1.0)
    # scalars = numpy_to_vtk(np.ascontiguousarray(field).ravel(order="C"), deep=True)
    # scalars.SetName(SCALAR_NAME)
    # image.GetPointData().SetScalars(scalars)
    # image.GetPointData().AddArray(scalars)
    # write_vtk_dataset(image, OUTPUT_DIR / "field")


    # # 3
    # print("Computing u=0 level set contour")
    # zero_level_set = extract_zero_level_set(image)
    # write_vtk_dataset(zero_level_set, OUTPUT_DIR / "u0_level_set")


    # # 4
    # print("Computing Reeb graph")
    # reeb_outputs = compute_reeb_graph(image, arc_sampling=arc_sampling, with_segmentation=with_segmentation)
    # output_names = ["reeb_nodes", "reeb_arcs", "reeb_segmentation"]
    # for index, dataset in enumerate(reeb_outputs):
    #     if index < len(output_names):
    #         name = output_names[index]
    #     else:
    #         name = f"reeb_output_{index}"
    #     write_vtk_dataset(dataset, OUTPUT_DIR / name)

    if periodic_domain:
        domain_mode = DOMAIN_PERIODIC
    else:
        domain_mode = DOMAIN_OPEN

    eps = 0

    persistence_threshold = 0.01 * np.ptp(field)
    FPO_Path = Path(__file__).parent.parent / "inputs" / "fundamental_periodic_orbits.h5"
    orbits = orb.io.read_h5(str(FPO_Path))
    orbit = orbits[orbit_index]
    field_orbit = orbit.resize(512, 512).transform(to="field")
    field_sherwood = np.asarray(field_orbit.state, dtype=np.float64)
    graph_sherwood, outputs_sherwood = get_nx_graph_and_reeb_outputs(field_sherwood, domain_mode, eps, persistence_threshold, arc_sampling, with_segmentation)

    persistence_threshold = 0.01 * np.ptp(field)
    FPO_Path = Path(__file__).parent.parent / "inputs" / "fundamental_periodic_orbits.h5"
    orbits = orb.io.read_h5(str(FPO_Path))
    orbit = orbits[orbit_index]
    field_orbit = orbit.resize(512, 512).transform(to="field")
    field_sherwood = np.asarray(field_orbit.state, dtype=np.float64)
    field_sherwoodRolled = np.roll(field_sherwood, shift=(roll_t, roll_x), axis=(0, 1))
    graph_sherwoodRolled, outputs_sherwoodRolled = get_nx_graph_and_reeb_outputs(field_sherwoodRolled, domain_mode, eps, persistence_threshold, arc_sampling, with_segmentation)

    # persistence_threshold = 0.1 * np.ptp(field)
    # FPO_Path = Path(__file__).parent.parent / "inputs" / "pitchfork_10_iterations.h5"
    # orbits = orb.io.read_h5(str(FPO_Path))
    # orbit = orbits[orbit_index]
    # field_orbit = orbit.resize(512, 512).transform(to="field")
    # field_clipped10 = np.asarray(field_orbit.state, dtype=np.float64)
    # graph_clipped10, outputs_clipped10 = get_nx_graph_and_reeb_outputs(field_clipped10, domain_mode, eps, persistence_threshold, arc_sampling, with_segmentation)

    field_a, field_b = field_sherwood, field_sherwoodRolled
    graph_a, graph_b = graph_sherwood, graph_sherwoodRolled

    isomorphic, original_hash, rolled_hash, original_skeleton, rolled_skeleton = compare_networkx_graphs(graph_a, graph_b)
    print(f"WL hashes equal:  {isomorphic}")

    physical_plot_path = OUTPUT_DIR / "physical_reeb_comparison_orbit_0.png"
    plot_physical_reeb_comparison(
        field_a,
        field_b,
        original_skeleton,
        rolled_skeleton,
        isomorphic=isomorphic,
        save_path=physical_plot_path,
    )

    graph_plot_path = OUTPUT_DIR / "reeb_skeleton_isomorphism_orbit_0.png"
    plot_reeb_graph_isomorphism_comparison(
        original_skeleton,
        rolled_skeleton,
        isomorphic=isomorphic,
        save_path=graph_plot_path,
    )

    

if __name__ == "__main__":
    run(
        arc_sampling=DEFAULT_ARC_SAMPLING,
        with_segmentation=False,
        roll_t=DEFAULT_ROLL_T,
        roll_x=DEFAULT_ROLL_X,
        periodic_domain=True,
    )
    print('done')

