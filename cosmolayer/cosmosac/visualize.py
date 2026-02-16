"""
.. module:: cosmolayer.cosmosac.visualize
   :synopsis: Visualize COSMO-SAC surface segments.

.. functionauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import argparse
import pathlib

import cmap
import numpy as np
import open3d as o3d

from cosmolayer.cosmosac import Component

RADII_MULTIPLIERS: tuple[float, float, float] = (1.5, 2.5, 4.0)


def ball_pivoting_algorithm(
    points: np.ndarray,
    normals: np.ndarray,
    vertex_rgb: np.ndarray,
    radii_multipliers: tuple[float, float, float],
) -> o3d.geometry.TriangleMesh:
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

    return mesh_bpa


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

    mesh_bpa = ball_pivoting_algorithm(pts, normals, vertex_rgb, RADII_MULTIPLIERS)

    if interpolated_colors:
        return mesh_bpa

    vertices = np.asarray(mesh_bpa.vertices, dtype=float)
    triangles = np.asarray(mesh_bpa.triangles, dtype=np.int64)
    colors = np.asarray(mesh_bpa.vertex_colors, dtype=float)

    new_vertices = vertices.tolist()
    new_colors = colors.tolist()

    def add_vertex(v: np.ndarray, c: np.ndarray) -> int:
        idx = len(new_vertices)
        new_vertices.append(v)
        new_colors.append(c)
        return idx

    midpoint_cache: dict[tuple[np.int64, np.int64], int] = {}

    def midpoint_vertices(i: np.int64, j: np.int64) -> tuple[int, int]:
        if (i, j) in midpoint_cache:
            return midpoint_cache[(i, j)], midpoint_cache[(j, i)]
        midpoint = (vertices[i] + vertices[j]) / 2
        mij = midpoint_cache[(i, j)] = add_vertex(midpoint, colors[i])
        mji = midpoint_cache[(j, i)] = add_vertex(midpoint, colors[j])
        return mij, mji

    new_triangles = []

    for i, j, k in triangles:
        mij, mji = midpoint_vertices(i, j)
        mjk, mkj = midpoint_vertices(j, k)
        mik, mki = midpoint_vertices(i, k)

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


def main() -> None:
    parser = argparse.ArgumentParser()
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
        "--interpolate-colors",
        action="store_true",
        help="Use interpolated colors instead of uniform segment colors",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        help="Matplotlib colormap name (default: jet)",
    )
    args = parser.parse_args()
    component = Component(args.cosmo_file.read_text())
    mesh = surface_tessellation(
        component,
        args.show_original_charge_densities,
        args.interpolate_colors,
        args.colormap,
    )
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


if __name__ == "__main__":
    main()
