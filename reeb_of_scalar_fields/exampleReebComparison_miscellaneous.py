from __future__ import annotations
import importlib
import os
import site
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

# TODO in production, remove .use("Agg") necessity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import orbithunter as orb

from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkTriangle, vtkUnstructuredGrid
from vtkmodules.vtkIOXML import (
    vtkXMLImageDataWriter,
    vtkXMLMultiBlockDataWriter,
    vtkXMLPolyDataWriter,
    vtkXMLUnstructuredGridWriter,
)


SCALAR_NAME = "u"

DEFAULT_ARC_SAMPLING = 20
DEFAULT_ROLL_T = 128
DEFAULT_ROLL_X = 170
DEFAULT_DEGENERACY_TOL = 1e-10
DOMAIN_PERIODIC = "periodic"
DOMAIN_OPEN = "open"


def log_step(message: str) -> float:
    print(message, flush=True)
    return time.perf_counter()


def log_done(start: float) -> None:
    print(f"  done in {time.perf_counter() - start:.3f} s", flush=True)


def save_figure(fig, save_path: Path | None, *, dpi: int = 200, message: str = "wrote plot") -> None:
    if save_path is None:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    pdf_path = save_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    # print(f"{message}: {save_path}")
    # print(f"{message}: {pdf_path}")


def _attach_scalar_array(dataset, field: np.ndarray) -> None:
    """Attach `field` as the named point-scalar array on a VTK dataset."""
    scalars = numpy_to_vtk(np.ascontiguousarray(field).ravel(order="C"), deep=True)
    scalars.SetName(SCALAR_NAME)
    dataset.GetPointData().SetScalars(scalars)
    dataset.GetPointData().AddArray(scalars)


def numpy_field_to_vtk_image(field: np.ndarray):
    """Convert u[t, x] to vtkImageData with point scalars named 'u'."""
    nt, nx = field.shape
    image = vtkImageData()
    image.SetDimensions(nx, nt, 1)
    image.SetOrigin(0.0, 0.0, 0.0)
    image.SetSpacing(1.0, 1.0, 1.0)
    _attach_scalar_array(image, field)
    return image


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

def insert_triangle(grid, a: int, b: int, c: int) -> None:
    triangle = vtkTriangle()
    triangle.GetPointIds().SetId(0, a)
    triangle.GetPointIds().SetId(1, b)
    triangle.GetPointIds().SetId(2, c)
    grid.InsertNextCell(triangle.GetCellType(), triangle.GetPointIds())


def numpy_field_to_periodic_vtk_grid(field: np.ndarray):
    """Convert u[t, x] to a periodic triangulated grid.

    vtkImageData has open rectangular boundaries. np.roll is periodic. This
    unstructured grid connects the x and t boundaries with wraparound triangles,
    so TTK computes the Reeb graph on a torus-like periodic domain.
    """
    t_count, x_count = field.shape
    grid = vtkUnstructuredGrid()
    points = vtkPoints()
    points.SetNumberOfPoints(t_count * x_count)

    for t_index in range(t_count):
        for x_index in range(x_count):
            points.SetPoint(point_id(t_index, x_index, x_count), float(x_index), float(t_index), 0.0)
    grid.SetPoints(points)

    for a, b, c in _periodic_triangles(field.shape):
        insert_triangle(grid, a, b, c)

    _attach_scalar_array(grid, field)
    return grid


def field_to_vtk_dataset(field: np.ndarray, domain_mode: str):
    if domain_mode == DOMAIN_PERIODIC:
        return numpy_field_to_periodic_vtk_grid(field)
    if domain_mode == DOMAIN_OPEN:
        return numpy_field_to_vtk_image(field)
    raise ValueError(f"Unknown domain_mode {domain_mode!r}")


def import_ttk_reeb_graph():
    """Import ttkReebGraph from the common TTK Python layouts."""
    candidate_modules = (
        "topologytoolkit.ttkReebGraph",
        "ttk.ttkReebGraph",
        "ttkReebGraph",
        "topologytoolkit",
        "vtkmodules.vtkFiltersTTK",
    )
    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        cls = getattr(module, "ttkReebGraph", None)
        if cls is not None:
            # print(f"Loaded TTK Reeb graph from {module_name}", flush=True)
            return cls

    # print_ttk_diagnostics()
    raise ImportError(
        "Could not import ttkReebGraph. VTK is importable, but TTK's VTK filters are not "
        "registered in this Python environment. If you installed TTK through conda-forge, "
        "make sure Spyder is using that same environment."
    )


def print_ttk_diagnostics() -> None:
    """Print enough environment detail to diagnose TTK/Python wiring in Spyder."""
    # print("\nTTK import diagnostics", flush=True)
    # print(f"  Python executable: {sys.executable}", flush=True)
    # print(f"  CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', '<unset>')}", flush=True)
    # print(f"  vtk version: {vtkVersion.GetVTKVersion()}", flush=True)

    paths = []
    try:
        paths.extend(site.getsitepackages())
    except AttributeError:
        pass
    user_site = site.getusersitepackages()
    if user_site:
        paths.append(user_site)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        paths.append(str(Path(conda_prefix) / "lib"))

    seen = set()
    for base in paths:
        if not base or base in seen:
            continue
        seen.add(base)
        base_path = Path(base)
        matches = []
        if base_path.exists():
            matches.extend(base_path.glob("topologytoolkit*"))
            matches.extend(base_path.glob("ttk*"))
        if matches:
            # print(f"  possible TTK files under {base_path}:", flush=True)
            for match in matches[:12]:
                ...
                # print(f"    {match}", flush=True)
        else:
            ...
            # print(f"  no obvious TTK files under {base_path}", flush=True)



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


def _scatter_by_critical_type(ax, graph, position_fn, *, size: float, linewidth: float = 0.6):
    """Scatter graph nodes colored/marked by critical_type_normalized (fallback critical_type)."""
    if graph is None:
        return
    type_to_nodes = defaultdict(list)
    for node, data in graph.nodes(data=True):
        type_to_nodes[data.get("critical_type_normalized", data.get("critical_type"))].append(node)

    for critical_type, nodes in sorted(type_to_nodes.items(), key=lambda item: str(item[0])):
        label, color, marker = critical_type_style(critical_type)
        xy = np.array([position_fn(node) for node in nodes])
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=size,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=linewidth,
            label=label,
            zorder=3,
        )




def stable_node_sort_key(graph, node):
    data = graph.nodes[node]
    return (
        round(float(data.get("scalar", 0.0)), 12),
        int(data.get("critical_type_normalized", data.get("critical_type", -1)) or -1),
        round(float(data.get("x", 0.0)), 6),
        round(float(data.get("t", 0.0)), 6),
        repr(node),
    )

def scalar_graph_layout(graph) -> dict:
    """Deterministic Reeb layout with scalar value as vertical coordinate."""
    import networkx as nx

    if graph.number_of_nodes() == 0:
        return {}

    simple_graph = nx.Graph(graph)
    components = sorted(
        nx.connected_components(simple_graph),
        key=lambda component: min(stable_node_sort_key(graph, node) for node in component),
    )

    positions = {}
    x_offset = 0.0
    for component in components:
        ordered_nodes = sorted(component, key=lambda node: stable_node_sort_key(graph, node))
        subgraph = simple_graph.subgraph(ordered_nodes).copy()

        if len(ordered_nodes) == 1:
            x_coords = {ordered_nodes[0]: 0.0}
        else:
            # Relabel to stable integers so the layout is not affected by TTK's node IDs.
            to_int = {node: index for index, node in enumerate(ordered_nodes)}
            from_int = {index: node for node, index in to_int.items()}
            int_graph = nx.relabel_nodes(subgraph, to_int, copy=True)
            initial = {
                index: np.array(
                    [
                        np.cos(2.0 * np.pi * index / len(ordered_nodes)),
                        np.sin(2.0 * np.pi * index / len(ordered_nodes)),
                    ]
                )
                for index in range(len(ordered_nodes))
            }
            raw_layout = nx.kamada_kawai_layout(int_graph, pos=initial, weight=None)
            x_coords = {from_int[index]: float(raw_layout[index][0]) for index in raw_layout}

        x_values = np.array(list(x_coords.values()), dtype=float)
        width = max(float(x_values.max() - x_values.min()), 1e-12)
        x_min = float(x_values.min())

        for node in ordered_nodes:
            scalar = graph.nodes[node].get("scalar")
            if scalar is None or not np.isfinite(scalar):
                scalar = 0.0
            x = (x_coords[node] - x_min) / width
            positions[node] = (x_offset + x, float(scalar))

        x_offset += 1.35

    return positions

def plot_networkx_reeb_graph(ax, graph, title: str, scalar_limits: tuple[float, float] | None = None):
    """Visualize a Reeb graph as critical nodes connected by arcs."""
    if graph is None or graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "empty graph", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    positions = scalar_graph_layout(graph)
    simple_edges = sorted({tuple(sorted((u, v))) for u, v in graph.edges() if u != v})

    for source, target in simple_edges:
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        ax.plot([x0, x1], [y0, y1], color="#666666", linewidth=1.5, alpha=0.75, zorder=1)

    _scatter_by_critical_type(ax, graph, lambda node: positions[node], size=70, linewidth=0.6)

    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel("layout")
    ax.set_ylabel(SCALAR_NAME)
    ax.set_xticks([])
    if scalar_limits is not None:
        ax.set_ylim(*scalar_limits)
    ax.margins(x=0.12, y=0.12)


def plot_reeb_graph_isomorphism_comparison(
    original_graph,
    rolled_graph,
    isomorphic: bool,
    save_path: Path | None,
):
    """Plot original and rolled Reeb skeletons side by side."""
    scalar_values = [
        data.get("scalar")
        for graph in (original_graph, rolled_graph)
        for _, data in graph.nodes(data=True)
        if data.get("scalar") is not None and np.isfinite(data.get("scalar"))
    ]
    scalar_limits = None
    if scalar_values:
        margin = 0.08 * max(max(scalar_values) - min(scalar_values), 1.0)
        scalar_limits = (min(scalar_values) - margin, max(scalar_values) + margin)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True, sharey=True)
    plot_networkx_reeb_graph(axes[0], original_graph, "Original Reeb skeleton", scalar_limits=scalar_limits)
    plot_networkx_reeb_graph(axes[1], rolled_graph, "Rolled Reeb skeleton", scalar_limits=scalar_limits)
    axes[1].legend(loc="upper right", framealpha=0.9)
    fig.suptitle(f"Reeb graph comparison via VF2++: skeleton isomorphic = {isomorphic}")

    if save_path is not None:
        save_figure(fig, save_path, message="wrote Reeb graph plot")


def split_periodic_path_jumps(path: np.ndarray, field_shape: tuple[int, int]) -> list[np.ndarray]:
    if len(path) < 2:
        return []
    t_count, x_count = field_shape
    dx = np.abs(np.diff(path[:, 0]))
    dy = np.abs(np.diff(path[:, 1]))
    jumps = np.where((dx > x_count / 2.0) | (dy > t_count / 2.0))[0] + 1
    return [segment for segment in np.split(path, jumps) if len(segment) >= 2]


def periodic_straight_edge_segments(
    source_data: dict,
    target_data: dict,
    field_shape: tuple[int, int],
    samples: int = 80,
) -> list[np.ndarray]:
    """Draw a graph edge along the shortest periodic straight segment."""
    t_count, x_count = field_shape
    x0, y0 = float(source_data.get("x", 0.0)), float(source_data.get("t", 0.0))
    x1, y1 = float(target_data.get("x", 0.0)), float(target_data.get("t", 0.0))

    dx = ((x1 - x0 + x_count / 2.0) % x_count) - x_count / 2.0
    dy = ((y1 - y0 + t_count / 2.0) % t_count) - t_count / 2.0

    s = np.linspace(0.0, 1.0, samples)
    path = np.column_stack(((x0 + s * dx) % x_count, (y0 + s * dy) % t_count))
    return split_periodic_path_jumps(path, field_shape)


def plot_physical_reeb_comparison(
    original_field: np.ndarray,
    rolled_field: np.ndarray,
    original_graph,
    rolled_graph,
    isomorphic: bool,
    save_path: Path | None,
):
    """Plot the Reeb skeleton in physical coordinates.

    This intentionally uses the PL-normalized NetworkX skeleton rather than
    raw TTK sampled arc polylines. The sampled polylines can loop through the
    periodic seam and make an unreadable plot; the skeleton is the object used
    for the actual isomorphism check.
    """
    field_shape = original_field.shape
    t_count, x_count = field_shape

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True, sharex=True, sharey=True)
    vmin = min(float(original_field.min()), float(rolled_field.min()))
    vmax = max(float(original_field.max()), float(rolled_field.max()))

    for ax, field, graph, title in (
        (axes[0], original_field, original_graph, "Original: physical Reeb skeleton"),
        (axes[1], rolled_field, rolled_graph, "Rolled: physical Reeb skeleton"),
    ):
        image = ax.imshow(
            field,
            origin="lower",
            cmap="RdBu_r",
            extent=(-0.5, x_count - 0.5, -0.5, t_count - 0.5),
            aspect="equal",
            alpha=0.42,
            vmin=vmin,
            vmax=vmax,
        )
        if graph is not None:
            simple_edges = sorted({tuple(sorted((u, v), key=repr)) for u, v in graph.edges() if u != v}, key=repr)
            for source, target in simple_edges:
                for segment in periodic_straight_edge_segments(graph.nodes[source], graph.nodes[target], field_shape):
                    ax.plot(segment[:, 0], segment[:, 1], color="#333333", linewidth=1.7, alpha=0.74, zorder=2)

        _scatter_by_critical_type(
            ax, graph, lambda node: (graph.nodes[node]["x"], graph.nodes[node]["t"]), size=58, linewidth=0.55
        )

        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("t index")
        ax.set_xlim(-0.5, x_count - 0.5)
        ax.set_ylim(-0.5, t_count - 0.5)

    axes[1].legend(loc="upper right", framealpha=0.9)
    fig.colorbar(image, ax=axes, label=SCALAR_NAME, shrink=0.82)
    fig.suptitle(f"Physical Reeb skeleton comparison: skeleton isomorphic = {isomorphic}")

    if save_path is not None:
        save_figure(fig, save_path, message="wrote physical Reeb plot")


def iter_leaf_datasets(dataset):
    """Yield non-multiblock datasets so plotting also works for TTK multiblock output."""
    if dataset is None:
        return
    if dataset.IsA("vtkMultiBlockDataSet"):
        for index in range(dataset.GetNumberOfBlocks()):
            block = dataset.GetBlock(index)
            if block is not None:
                yield from iter_leaf_datasets(block)
    else:
        yield dataset


def plot_vtk_cells(ax, dataset, *, color: str, linewidth: float, label: str | None = None):
    """Plot VTK line/polyline cells in x-y coordinates."""
    y_min, y_max = ax.get_ylim()
    flip_y = y_max < y_min
    first_line = True
    for leaf in iter_leaf_datasets(dataset):
        points = leaf.GetPoints()
        if points is None:
            continue

        for cell_index in range(leaf.GetNumberOfCells()):
            point_ids = leaf.GetCell(cell_index).GetPointIds()
            if point_ids.GetNumberOfIds() < 2:
                continue

            xy = np.array([points.GetPoint(point_ids.GetId(i))[:2] for i in range(point_ids.GetNumberOfIds())])
            if flip_y:
                xy[:, 1] = y_min + y_max - xy[:, 1]
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=linewidth, label=label if first_line else None)
            first_line = False



def nearest_field_value(field: np.ndarray, point) -> float:
    """Sample u at the nearest grid point to a VTK x-y point."""
    t_count, x_count = field.shape
    x = int(np.clip(round(point[0]), 0, x_count - 1))
    t = int(np.clip(round(point[1]), 0, t_count - 1))
    return float(field[t, x])


def reeb_graph_edges_from_arcs(field: np.ndarray, reeb_arcs):
    """Extract abstract graph edges from the endpoints of TTK arc polylines."""
    node_values = {}
    edges = []

    for leaf in iter_leaf_datasets(reeb_arcs):
        points = leaf.GetPoints()
        if points is None:
            continue

        for cell_index in range(leaf.GetNumberOfCells()):
            point_ids = leaf.GetCell(cell_index).GetPointIds()
            if point_ids.GetNumberOfIds() < 2:
                continue

            start_point = points.GetPoint(point_ids.GetId(0))[:2]
            end_point = points.GetPoint(point_ids.GetId(point_ids.GetNumberOfIds() - 1))[:2]
            start_key = tuple(np.round(start_point, 6))
            end_key = tuple(np.round(end_point, 6))

            node_values[start_key] = nearest_field_value(field, start_point)
            node_values[end_key] = nearest_field_value(field, end_point)
            if start_key != end_key:
                edges.append((start_key, end_key))

    return node_values, edges


def abstract_reeb_layout(node_values, edges):
    """Lay out a Reeb graph with scalar value on the vertical axis."""
    positions = {}
    for component_index, component in enumerate(connected_components(node_values.keys(), edges)):
        center_x = 1.6 * component_index
        by_level = defaultdict(list)
        for node in component:
            by_level[round(node_values[node], 8)].append(node)

        for level, level_nodes in sorted(by_level.items()):
            level_nodes = sorted(level_nodes)
            offsets = [0.0] if len(level_nodes) == 1 else np.linspace(-0.35, 0.35, len(level_nodes))
            for node, offset in zip(level_nodes, offsets):
                positions[node] = (center_x + float(offset), node_values[node])

    return positions



def plot_abstract_reeb_graph(ax, field: np.ndarray, reeb_arcs):
    """Plot the Reeb graph as topology, not as physical-domain spaghetti."""
    node_values, edges = reeb_graph_edges_from_arcs(field, reeb_arcs)
    if not node_values:
        ax.text(0.5, 0.5, "No Reeb arcs found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    positions = abstract_reeb_layout(node_values, edges)
    for source, target in edges:
        if source not in positions or target not in positions:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        ax.plot([x0, x1], [y0, y1], color="gold", linewidth=2.2, zorder=2)

    xy = np.array([positions[node] for node in positions])
    ax.scatter(xy[:, 0], xy[:, 1], s=60, c="crimson", edgecolors="black", linewidths=0.5, zorder=3)

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.55)
    ax.set_title("Abstract Reeb graph")
    ax.set_xlabel("component / branch layout")
    ax.set_ylabel("u")
    ax.set_xticks([])
    ax.margins(x=0.18, y=0.12)




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


def count_components(nodes: set[int], edges: list[tuple[int, int]]) -> int:
    return len(_connected_components(nodes, edges))

def connected_components(nodes, edges) -> list[list]:
    return _connected_components(nodes, edges)

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


def pl_degeneracy_report(field: np.ndarray, eps: float = 0.0) -> dict:
    classifications = pl_link_classification(field, eps=eps)
    counts = defaultdict(int)
    degenerate_vertices = []
    for vertex, data in classifications.items():
        counts[data["pl_label"]] += 1
        if data["pl_label"] == "degenerate saddle":
            degenerate_vertices.append(vertex)

    return {
        "counts": dict(sorted(counts.items())),
        "degenerate_vertices": degenerate_vertices[:20],
        "degenerate_count": len(degenerate_vertices),
    }



def _central_diff(field: np.ndarray, axis: int) -> np.ndarray:
    return (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / 2.0


def _second_diff(field: np.ndarray, axis: int) -> np.ndarray:
    return np.roll(field, -1, axis=axis) - 2.0 * field + np.roll(field, 1, axis=axis)


def finite_difference_degeneracy_report(
    field: np.ndarray,
    tol: float = DEFAULT_DEGENERACY_TOL,
) -> dict:
    """Check sampled periodic field for near degeneracies."""
    field = np.asarray(field, dtype=np.float64)
    amplitude = max(float(np.ptp(field)), 1.0)
    value_tol = tol * amplitude

    unique_values = np.unique(np.round(field / value_tol).astype(np.int64)) if value_tol > 0 else np.unique(field)
    duplicate_value_count = field.size - len(unique_values)

    equal_neighbor_edges = sum(
        int(np.count_nonzero(np.abs(field - np.roll(field, -1, axis=axis)) <= value_tol))
        for axis in (0, 1)
    )

    # Flat 2x2 periodic cells, vectorized: compare each cell's 4 corners at once
    # instead of a Python double loop over every (t, x) pair.
    v00 = field
    v10 = np.roll(field, -1, axis=0)
    v01 = np.roll(field, -1, axis=1)
    v11 = np.roll(np.roll(field, -1, axis=0), -1, axis=1)
    cell_max = np.maximum(np.maximum(v00, v10), np.maximum(v01, v11))
    cell_min = np.minimum(np.minimum(v00, v10), np.minimum(v01, v11))
    flat_cells = int(np.count_nonzero((cell_max - cell_min) <= value_tol))

    ux, ut = _central_diff(field, 1), _central_diff(field, 0)
    uxx, utt = _second_diff(field, 1), _second_diff(field, 0)
    uxt = (
        np.roll(np.roll(field, -1, axis=0), -1, axis=1)
        - np.roll(np.roll(field, -1, axis=0), 1, axis=1)
        - np.roll(np.roll(field, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(field, 1, axis=0), 1, axis=1)
    ) / 4.0

    grad_norm = np.sqrt(ux * ux + ut * ut)
    hessian_det = uxx * utt - uxt * uxt
    grad_tol = 1e-2 * amplitude
    hessian_tol = 1e-6 * amplitude * amplitude
    critical_candidates = grad_norm <= grad_tol
    degenerate_candidates = critical_candidates & (np.abs(hessian_det) <= hessian_tol)

    return {
        "duplicate_scalar_values": int(duplicate_value_count),
        "equal_periodic_neighbor_edges": int(equal_neighbor_edges),
        "flat_2x2_periodic_cells": flat_cells,
        "finite_difference_critical_candidates": int(np.count_nonzero(critical_candidates)),
        "near_degenerate_fd_candidates": int(np.count_nonzero(degenerate_candidates)),
        "min_abs_hessian_det_at_candidates": float(np.min(np.abs(hessian_det[critical_candidates])))
        if np.any(critical_candidates)
        else None,
    }

# def print_degeneracy_report(label: str, field: np.ndarray, tol: float = DEFAULT_DEGENERACY_TOL) -> None:
#     report = finite_difference_degeneracy_report(field, tol=tol)
#     print(f"{label} degeneracy report:", flush=True)
#     for key, value in report.items():
#         print(f"  {key}: {value}", flush=True)
#     pl_report = pl_degeneracy_report(field)
#     print(f"{label} PL link report:", flush=True)
#     print(f"  counts: {pl_report['counts']}", flush=True)
#     print(f"  degenerate_count: {pl_report['degenerate_count']}", flush=True)
#     if pl_report["degenerate_vertices"]:
#         print(f"  first_degenerate_vertices: {pl_report['degenerate_vertices']}", flush=True)


def plot_paths(ax, paths: list[np.ndarray], *, color: str, linewidth: float, label: str | None = None):
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    jump_threshold = 0.45 * max(x_span, y_span)
    first = True
    for path in paths:
        if len(path) < 2:
            continue

        jumps = np.where(np.linalg.norm(np.diff(path, axis=0), axis=1) > jump_threshold)[0] + 1
        for segment in np.split(path, jumps):
            if len(segment) < 2:
                continue
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                color=color,
                linewidth=linewidth,
                label=label if first else None,
            )
            first = False


def _plot_field_panel(ax, field, paths, title, extent, vmin, vmax, label):
    """Draw one imshow + u=0 contour panel; returns the imshow handle."""
    image = ax.imshow(field, origin="lower", cmap="RdBu_r", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
    plot_paths(ax, paths, color="black", linewidth=1.8, label=label)
    ax.set_title(title)
    return image


def iter_datasets_with_output_index(reeb_outputs):
    for output_index, output in enumerate(reeb_outputs):
        for dataset in iter_leaf_datasets(output):
            yield output_index, dataset


def vtk_data_array_names(attributes) -> list[str]:
    return [
        (attributes.GetArray(index).GetName() if attributes.GetArray(index) is not None else f"<array {index}>")
        for index in range(attributes.GetNumberOfArrays())
    ]


def inspect_ttk_outputs(reeb_outputs) -> None:
    """Print VTK data arrays available on each TTK output for debugging."""
    for output_index, dataset in iter_datasets_with_output_index(reeb_outputs):
        point_arrays = vtk_data_array_names(dataset.GetPointData()) if hasattr(dataset, "GetPointData") else []
        cell_arrays = vtk_data_array_names(dataset.GetCellData()) if hasattr(dataset, "GetCellData") else []
        # print(
        #     f"TTK output {output_index}: {dataset.GetClassName()}, "
        #     f"{dataset.GetNumberOfPoints() if hasattr(dataset, 'GetNumberOfPoints') else 0} points, "
        #     f"{dataset.GetNumberOfCells() if hasattr(dataset, 'GetNumberOfCells') else 0} cells, "
        #     f"point arrays={point_arrays}, cell arrays={cell_arrays}",
        #     flush=True,
        # )


def write_vtk_dataset(dataset, path: Path) -> Path:
    """Write a VTK dataset and return the final path including suffix."""
    writer_by_type = (
        ("vtkPolyData", vtkXMLPolyDataWriter, ".vtp"),
        ("vtkUnstructuredGrid", vtkXMLUnstructuredGridWriter, ".vtu"),
        ("vtkImageData", vtkXMLImageDataWriter, ".vti"),
        ("vtkMultiBlockDataSet", vtkXMLMultiBlockDataWriter, ".vtm"),
    )

    for vtk_type_name, writer_type, suffix in writer_by_type:
        if dataset is not None and dataset.IsA(vtk_type_name):
            out_path = path.with_suffix(suffix)
            writer = writer_type()
            writer.SetFileName(str(out_path))
            writer.SetInputData(dataset)
            if writer.Write() != 1:
                raise RuntimeError(f"VTK failed to write {out_path}")
            return out_path

    raise TypeError(f"No XML writer configured for {dataset.GetClassName()}")


def iter_leaf_datasets(dataset):
    """Yield non-multiblock datasets so plotting also works for TTK multiblock output."""
    if dataset is None:
        return
    if dataset.IsA("vtkMultiBlockDataSet"):
        for index in range(dataset.GetNumberOfBlocks()):
            block = dataset.GetBlock(index)
            if block is not None:
                yield from iter_leaf_datasets(block)
    else:
        yield dataset

