"""
.. module:: cosmolayer.cosmosac.visualize
   :synopsis: Visualize COSMO-SAC surface segments.

.. functionauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import argparse
import pathlib

import cmap
import networkx as nx
import numpy as np
import open3d as o3d
import periodictable as pt

from cosmolayer.cosmosac import Component

RADII_MULTIPLIERS: tuple[float, float, float] = (1.5, 2.5, 4.0)

ELEMENT_COLORS = {  # https://pymolwiki.org/Color_Values
    "Br": (0.650980392, 0.160784314, 0.160784314),
    "C": (0.2, 1.0, 0.2),
    "Cl": (0.121568627, 0.941176471, 0.121568627),
    "F": (0.701960784, 1.0, 1.0),
    "H": (0.9, 0.9, 0.9),
    "I": (0.580392157, 0.0, 0.580392157),
    "N": (0.2, 0.2, 1.0),
    "O": (1.0, 0.3, 0.3),
    "P": (1.0, 0.501960784, 0.0),
    "Si": (0.941176471, 0.784313725, 0.627450980),
    "S": (0.9, 0.775, 0.25),
}


TOLERANCE: float = 1e-10
DOT_PRODUCT_TOLERANCE: float = 0.9
X_AXIS: np.ndarray = np.array([1.0, 0.0, 0.0])
Y_AXIS: np.ndarray = np.array([0.0, 1.0, 0.0])
Z_AXIS: np.ndarray = np.array([0.0, 0.0, 1.0])


def estimate_vdw_radius(element: str) -> float:
    return float(pt.elements.symbol(element).covalent_radius) + 0.8  # Å


def create_atom_spheres(
    component: Component,
    radius_scale: float,
    resolution: int = 40,
    default_color: tuple[float, float, float] = (0.7, 0.7, 0.7),
) -> list[o3d.geometry.TriangleMesh]:
    atom_df = component.get_atom_data()
    spheres: list[o3d.geometry.TriangleMesh] = []
    for item in atom_df.itertuples(index=False):
        element = str(item.element).strip()
        radius = estimate_vdw_radius(element) * radius_scale
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=radius,
            resolution=resolution,
        )
        sphere.compute_vertex_normals()
        sphere.translate((item.x, item.y, item.z))
        rgb = np.array(ELEMENT_COLORS.get(element, default_color))
        sphere.paint_uniform_color(rgb)
        spheres.append(sphere)
    return spheres


def compute_rotation_matrix(
    original_axis: np.ndarray, target_axis: np.ndarray, normalize: bool = False
) -> np.ndarray:
    """Rodrigues' rotation formula between two unit-direction vectors."""
    if normalize:
        original_axis = original_axis / np.linalg.norm(original_axis)
        target_axis = target_axis / np.linalg.norm(target_axis)
    v = np.cross(original_axis, target_axis)
    c = original_axis.dot(target_axis)
    s2 = v.dot(v)
    if s2 < TOLERANCE:
        if c > 0:
            return np.eye(3)  # Parallel (c ≈ 1) → identity
        arbitrary = X_AXIS if abs(original_axis[0]) < DOT_PRODUCT_TOLERANCE else Y_AXIS
        arbitrary -= original_axis * original_axis.dot(arbitrary)
        orthogonal = arbitrary / np.linalg.norm(arbitrary)
        return 2.0 * np.outer(orthogonal, orthogonal) - np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation: np.ndarray = np.eye(3) + kmat + ((1 - c) / s2) * kmat @ kmat
    return rotation


def create_bond_sticks(
    component: Component,
    atom_radius_scale: float,
    bond_radius: float,
    resolution: int = 100,
    default_color: tuple[float, float, float] = (0.7, 0.7, 0.7),
) -> list[o3d.geometry.TriangleMesh]:
    atom_df = component.get_atom_data()
    coords = atom_df[["x", "y", "z"]].values
    elements = atom_df["element"].values
    radii = atom_df["element"].apply(estimate_vdw_radius).values * atom_radius_scale
    bonds = component.get_bonds()
    cylinders: list[o3d.geometry.TriangleMesh] = []
    for i, j in bonds:
        vector = coords[j] - coords[i]
        length = np.linalg.norm(vector)
        if length < radii[i] + radii[j]:
            continue
        axis = vector / length
        rotation = compute_rotation_matrix(Z_AXIS, axis)
        midpoint = (coords[i] + coords[j] + (radii[i] - radii[j]) * axis) / 2
        for k in (i, j):
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=bond_radius,
                height=np.linalg.norm(coords[k] - midpoint),
                resolution=resolution,
            )
            cylinder.rotate(rotation, center=np.zeros(3))
            cylinder.translate((coords[k] + midpoint) / 2)
            cylinder.compute_vertex_normals()
            rgb = np.array(ELEMENT_COLORS.get(elements[k], default_color))
            cylinder.paint_uniform_color(rgb)
            cylinders.append(cylinder)
    return cylinders


def ball_pivoting_algorithm(
    points: np.ndarray,
    normals: np.ndarray,
    vertex_rgb: np.ndarray,
    radii_multipliers: tuple[float, float, float],
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.normals = o3d.utility.Vector3dVector(normals)

    spacing = np.asarray(pcd.compute_nearest_neighbor_distance()).mean().item()
    radii = o3d.utility.DoubleVector([m * spacing for m in radii_multipliers])

    mesh_bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, radii
    )

    mesh_bpa.remove_degenerate_triangles()
    mesh_bpa.remove_duplicated_triangles()
    mesh_bpa.remove_non_manifold_edges()
    mesh_bpa.remove_unreferenced_vertices()

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    indices = np.empty(len(mesh_bpa.vertices), dtype=int)
    for vi, v in enumerate(mesh_bpa.vertices):
        _, idx, _ = kdtree.search_knn_vector_3d(v, 1)
        indices[vi] = int(idx[0])

    vertex_rgb = vertex_rgb[indices]
    mesh_bpa.vertex_colors = o3d.utility.Vector3dVector(vertex_rgb)

    return mesh_bpa, indices


def find_loops(
    mesh: o3d.geometry.TriangleMesh, edge_color: str
) -> list[o3d.geometry.LineSet]:
    graph = nx.Graph()
    for triangle in mesh.triangles:
        _, j, k = map(int, triangle)
        graph.add_edge(j, k)
    loops = nx.cycle_basis(graph)

    vertices = np.asarray(mesh.vertices, dtype=float)
    linesets: list[o3d.geometry.LineSet] = []

    for loop in loops:
        idx = np.asarray(loop + [loop[0]], dtype=int)
        pts = vertices[idx]
        lines = np.column_stack(
            [np.arange(len(idx) - 1), np.arange(1, len(idx))]
        ).astype(np.int32)
        lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        if edge_color is not None:
            rgb = np.asarray(cmap.Color(edge_color))[:3]
            lineset.paint_uniform_color(rgb)
        linesets.append(lineset)

    return linesets


def geodesic_centroid(center: np.ndarray, *vertices: np.ndarray) -> np.ndarray:
    num_vertices = len(vertices)
    vectors = [v - center for v in vertices]
    norms = [np.linalg.norm(v) for v in vectors]
    radius = sum(norms) / num_vertices
    mean_vector = sum(vectors) / num_vertices
    centroid: np.ndarray = center + radius * mean_vector / np.linalg.norm(mean_vector)
    return centroid


def surface_tessellation(
    component: Component,
    original_charge_densities: bool = False,
    interpolated_colors: bool = False,
    colormap: str = "jet",
) -> o3d.geometry.TriangleMesh:
    segment_data = component.get_segment_data()
    atom_data = component.get_atom_data()
    sigma_grid = component.get_sigma_grid()
    vmin, vmax = sigma_grid[0], sigma_grid[-1]
    sigmas = segment_data[
        "sigma" if original_charge_densities else "sigma_avg"
    ].values.clip(vmin, vmax)

    atom_coords = np.stack(
        [segment_data["atom"].map(atom_data[axis]).values for axis in "xyz"], axis=1
    )
    pts = segment_data[["x", "y", "z"]].values
    displacements = pts - atom_coords
    normals = displacements / np.linalg.norm(displacements, axis=1, keepdims=True)

    normalized_sigmas = (sigmas.clip(vmin, vmax) - vmin) / (vmax - vmin)
    mapper = cmap.Colormap(colormap)
    vertex_rgb = mapper(normalized_sigmas)[:, :3]

    mesh_bpa, indices = ball_pivoting_algorithm(
        pts, normals, vertex_rgb, RADII_MULTIPLIERS
    )

    if interpolated_colors:
        return mesh_bpa

    vertices = np.asarray(mesh_bpa.vertices, dtype=float)
    triangles = np.asarray(mesh_bpa.triangles, dtype=int)
    colors = np.asarray(mesh_bpa.vertex_colors, dtype=float)
    atoms = segment_data["atom"].values[indices]

    new_vertices = vertices.tolist()
    new_colors = colors.tolist()

    def add_vertex(v: np.ndarray, c: np.ndarray) -> int:
        idx = len(new_vertices)
        new_vertices.append(v)
        new_colors.append(c)
        return idx

    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint_vertices(i: int, j: int) -> tuple[int, int]:
        if (i, j) in midpoint_cache:
            return midpoint_cache[(i, j)], midpoint_cache[(j, i)]

        if atoms[i] == atoms[j]:
            midpoint = geodesic_centroid(
                atom_coords[atoms[i]], vertices[i], vertices[j]
            )
        else:
            midpoint = (vertices[i] + vertices[j]) / 2

        mij = midpoint_cache[(i, j)] = add_vertex(midpoint, colors[i])
        mji = midpoint_cache[(j, i)] = add_vertex(midpoint, colors[j])
        return mij, mji

    new_triangles = []

    for triangle in triangles:
        i, j, k = map(int, triangle)
        mij, mji = midpoint_vertices(i, j)
        mjk, mkj = midpoint_vertices(j, k)
        mik, mki = midpoint_vertices(i, k)

        if atoms[i] == atoms[j] == atoms[k]:
            centroid = geodesic_centroid(
                atom_coords[atoms[i]], vertices[i], vertices[j], vertices[k]
            )
        else:
            centroid = (vertices[i] + vertices[j] + vertices[k]) / 3

        mijk = add_vertex(centroid, colors[i])
        mjki = add_vertex(centroid, colors[j])
        mkij = add_vertex(centroid, colors[k])

        new_triangles += [
            [i, mij, mijk],
            [i, mik, mijk],
            [j, mji, mjki],
            [j, mjk, mjki],
            [k, mkj, mkij],
            [k, mki, mkij],
        ]

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(new_vertices),
        triangles=o3d.utility.Vector3iVector(new_triangles),
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    mesh.compute_vertex_normals()

    return mesh


def generate_geometries(
    component: Component,
    original_charge_densities: bool = False,
    use_continuous_colors: bool = False,
    colormap: str = "jet",
    segment_edge_color: str | None = None,
) -> tuple[o3d.geometry.Geometry3D, ...]:
    """Build Open3D geometries for visualizing a component's COSMO surface.

    Returns a tuple of Open3D geometries:

    (1) a tessellated surface mesh colored by screening charge density;
    (2) optionally, segment-boundary loop line sets when ``segment_edge_color`` is set;
    (3) atom spheres; and
    (4) bond sticks.

    Parameters
    ----------
    component : Component
        The molecular component whose COSMO surface is to be visualized.
    original_charge_densities : bool, optional
        If ``True``, color the surface using the original (unsmoothed) segment
        charge densities instead of the distance-weighted averages. Default is
        ``False``.
    use_continuous_colors : bool, optional
        If ``True``, use interpolated colors across the surface; otherwise,
        segments are uniformly colored. Default is ``False``.
    colormap : str, optional
        Name of the colormap used to map charge density to color (e.g.
        ``"jet"``, ``"viridis"``). Default is ``"jet"``.
    segment_edge_color : str or None, optional
        Color name for the edges between segments (e.g. ``"black"``).
        If ``None`` or if ``use_continuous_colors`` is ``True``, no edge
        loops are drawn. Default is ``None``.

    Returns
    -------
    tuple of Geometry3D
        A sequence of Open3D geometries: mesh, loops (if any), atom spheres,
        and bond sticks.

    Examples
    --------
    >>> from importlib.resources import files
    >>> from cosmolayer.cosmosac import Component
    >>> from cosmolayer.cosmosac.visualize import generate_geometries
    >>> path = files("cosmolayer.data") / "C=C(N)O.cosmo"
    >>> component = Component(path.read_text())
    >>> geometries = generate_geometries(component)
    >>> len(geometries) >= 1
    True
    >>> type(geometries[0]).__name__
    'TriangleMesh'
    >>> geometries_loops = generate_geometries(component, segment_edge_color="black")
    >>> len(geometries_loops) > len(geometries)
    True
    """
    mesh = surface_tessellation(
        component,
        original_charge_densities,
        use_continuous_colors,
        colormap,
    )
    if segment_edge_color is None or use_continuous_colors:
        loops = []
    else:
        loops = find_loops(mesh, segment_edge_color)
    atom_spheres = create_atom_spheres(component, 0.25)
    bond_sticks = create_bond_sticks(component, 0.25, 0.1)
    return (mesh, *loops, *atom_spheres, *bond_sticks)


def get_parser() -> argparse.ArgumentParser:
    """Return the argument parser for cosmoview (used by sphinx-argparse)."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Visualize COSMO files",
    )
    parser.add_argument(
        "cosmo_file",
        type=pathlib.Path,
        help="Path to a COSMO quantum mechanical output file",
    )
    parser.add_argument(
        "--show-original-charge-densities",
        action="store_true",
        help="Show original charge densities instead of smoothed ones",
    )
    parser.add_argument(
        "--use-continuous-colors",
        action="store_true",
        help="Use continuous colors instead of uniformly colored segments",
    )
    parser.add_argument(
        "--segment-edge-color",
        type=str,
        default=None,
        help="Color of the edges between segments (default: None)",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        help="Matplotlib colormap name (default: jet)",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()
    component = Component(args.cosmo_file.read_text())
    geometries = generate_geometries(
        component,
        args.show_original_charge_densities,
        args.use_continuous_colors,
        args.colormap,
        args.segment_edge_color,
    )
    o3d.visualization.draw_geometries(
        geometries,
        mesh_show_back_face=True,
        window_name=f"Surface Segments from {args.cosmo_file.name}",
    )


if __name__ == "__main__":
    main()
