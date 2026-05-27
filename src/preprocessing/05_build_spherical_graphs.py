# ============================================================
# CELL 5 — Physics-aware multi-resolution spherical graph
# Dynamic-ready transport-conditioned graph framework
#
# Builds:
#   HR graph  : target resolution
#   MR graph  : input for MR -> HR evaluation / target for LR -> MR training
#   LR graph  : input for LR -> MR training
#
# Provides:
#   - deterministic structured spherical stencil edges
#   - edge directions for future flow-aware attention
#   - edge distances for pressure-dependent transport kernels
#   - area-weighted pooling/unpooling
# ============================================================

import os
import json
import numpy as np
import scipy.sparse as sp

from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components


# ============================================================
# Coordinate transforms
# ============================================================

def longlat_to_cartesian(arr, radius=1.0):
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.shape[1] == 2

    lon = arr[:, 0]
    lat = arr[:, 1]

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)

    coords = np.stack([x, y, z], axis=1).astype(np.float32)

    assert np.isfinite(coords).all()

    return coords


# ============================================================
# Structured spherical stencil graph
# ============================================================

def build_structured_edges(n_lat, n_lon):
    """
    Deterministic spherical grid stencil.

    This avoids pure kNN tie-breaking instability near the equator.

    Edge convention:
        edge_index[0] = source node
        edge_index[1] = destination node

    Edge types:
        0 = north/south
        1 = east/west
        2 = diagonal
    """

    edge_src = []
    edge_dst = []
    edge_type = []

    def idx(y, x):
        return y * n_lon + x

    for y in range(n_lat):
        for x in range(n_lon):

            center = idx(y, x)

            neighbors = [
                # north / south
                (y - 1, x, 0),
                (y + 1, x, 0),

                # west / east, periodic longitude
                (y, (x - 1) % n_lon, 1),
                (y, (x + 1) % n_lon, 1),

                # diagonals, periodic longitude
                (y - 1, (x - 1) % n_lon, 2),
                (y - 1, (x + 1) % n_lon, 2),
                (y + 1, (x - 1) % n_lon, 2),
                (y + 1, (x + 1) % n_lon, 2),
            ]

            for ny, nx, etype in neighbors:

                if ny < 0 or ny >= n_lat:
                    continue

                neighbor = idx(ny, nx)

                edge_src.append(center)
                edge_dst.append(neighbor)
                edge_type.append(etype)

    edge_index = np.stack(
        [np.array(edge_src), np.array(edge_dst)],
        axis=0
    ).astype(np.int64)

    edge_type = np.array(edge_type).astype(np.int64)

    return edge_index, edge_type


# ============================================================
# Edge geometry
# ============================================================

def compute_edge_geometry(coords, edge_index):
    """
    Computes directional edge vectors and chordal distances.

    edge_vec points from source -> destination.
    """

    src = edge_index[0]
    dst = edge_index[1]

    vecs = coords[dst] - coords[src]

    dist = np.linalg.norm(vecs, axis=1)

    edge_vecs = vecs / np.clip(dist[:, None], 1e-12, None)

    assert np.isfinite(edge_vecs).all()
    assert np.isfinite(dist).all()
    assert dist.min() > 0

    return edge_vecs.astype(np.float32), dist.astype(np.float32)


# ============================================================
# Optional sparse transport candidate edges
# ============================================================

def build_transport_edges(coords, k_transport=4):
    """
    Adds candidate non-stencil edges.

    These are not used directly by the static Laplacian.
    They are saved for future flow-aware / attention operators.
    """

    tree = cKDTree(coords)

    _, ind = tree.query(coords, k=k_transport + 1)

    src_extra = []
    dst_extra = []

    n_nodes = coords.shape[0]

    for i in range(n_nodes):
        for j_idx in range(1, k_transport + 1):

            j = ind[i, j_idx]

            src_extra.append(i)
            dst_extra.append(j)

    extra_edges = np.stack(
        [np.array(src_extra), np.array(dst_extra)],
        axis=0
    ).astype(np.int64)

    return extra_edges


# ============================================================
# Physics-aware static adjacency / Laplacian
# ============================================================

def build_physics_graph(
    coords,
    edge_index,
    edge_distance,
    edge_type,
    t_kernel=None,
    laplacian_type="scaled_normalized"
):
    """
    Static geometric graph used as the spherical scaffold.

    Dynamic/flow-aware effects should later modulate messages,
    not replace this fixed anchor.
    """

    n_nodes = coords.shape[0]

    if t_kernel is None:
        avg_dist_sq = np.mean(edge_distance ** 2)
        t_kernel = 0.375 * avg_dist_sq

    assert t_kernel > 0

    weights = np.exp(
        -(edge_distance ** 2) / (4.0 * t_kernel)
    )

    # Slightly weaker diagonal diffusion.
    edge_scaling = np.ones_like(weights)
    edge_scaling[edge_type == 2] *= 0.85
    weights *= edge_scaling

    rows = edge_index[0]
    cols = edge_index[1]

    A = sp.coo_matrix(
        (weights, (rows, cols)),
        shape=(n_nodes, n_nodes)
    )

    A.sum_duplicates()

    # Enforce exact symmetry for spectral stability.
    A = 0.5 * (A + A.T)
    A = A.tocsr()

    degree = np.array(A.sum(axis=1)).ravel()

    assert degree.min() > 0, "Graph contains isolated nodes"

    n_components, _ = connected_components(A, directed=False)

    assert n_components == 1, (
        f"Graph disconnected: {n_components} components"
    )

    if laplacian_type == "combinatorial":

        D = sp.diags(degree)
        L = D - A

    elif laplacian_type in ["normalized", "scaled_normalized"]:

        inv_sqrt_degree = 1.0 / np.sqrt(
            np.maximum(degree, 1e-12)
        )

        D_inv_sqrt = sp.diags(inv_sqrt_degree)
        I = sp.eye(n_nodes, format="csr")

        L_norm = I - D_inv_sqrt @ A @ D_inv_sqrt

        if laplacian_type == "normalized":
            L = L_norm

        else:
            # Spectrum approximately [-1, 1] for Chebyshev filters.
            L = L_norm - I

    else:
        raise ValueError(f"Unknown Laplacian type: {laplacian_type}")

    L = L.tocoo()
    A = A.tocoo()

    assert np.isfinite(L.data).all()
    assert np.isfinite(A.data).all()

    return L, A, float(t_kernel)


# ============================================================
# Area-weighted pooling
# ============================================================

def create_pooling_matrix(
    n_lat_low,
    n_lon_low,
    lat_high_2d,
    scale_factor=2
):
    """
    Area-weighted parent-child pooling.

    Pool:
        high -> low

    Unpool:
        low -> high parent-child broadcast
    """

    assert lat_high_2d.ndim == 2

    n_low = n_lat_low * n_lon_low

    n_lat_high = n_lat_low * scale_factor
    n_lon_high = n_lon_low * scale_factor

    n_high = n_lat_high * n_lon_high

    assert lat_high_2d.shape == (n_lat_high, n_lon_high)

    dlat_deg = float(
        abs(lat_high_2d[1, 0] - lat_high_2d[0, 0])
    )

    lat_rad = np.deg2rad(lat_high_2d)
    dlat = np.deg2rad(dlat_deg)

    area = np.abs(
        np.sin(lat_rad + dlat / 2.0)
        -
        np.sin(lat_rad - dlat / 2.0)
    )

    assert np.isfinite(area).all()
    assert area.min() > 0

    row_pool = []
    col_pool = []
    data_pool = []

    row_unpool = []
    col_unpool = []
    data_unpool = []

    for y in range(n_lat_low):
        for x in range(n_lon_low):

            low_idx = y * n_lon_low + x

            child_idx = []
            child_area = []

            for dy in range(scale_factor):
                for dx in range(scale_factor):

                    hy = y * scale_factor + dy
                    hx = x * scale_factor + dx

                    high_idx = hy * n_lon_high + hx

                    child_idx.append(high_idx)
                    child_area.append(area[hy, hx])

            child_area = np.array(child_area)

            weights = child_area / child_area.sum()

            for c_idx, w in zip(child_idx, weights):

                row_pool.append(low_idx)
                col_pool.append(c_idx)
                data_pool.append(w)

                row_unpool.append(c_idx)
                col_unpool.append(low_idx)
                data_unpool.append(1.0)

    P = sp.coo_matrix(
        (data_pool, (row_pool, col_pool)),
        shape=(n_low, n_high)
    ).tocoo()

    U = sp.coo_matrix(
        (data_unpool, (row_unpool, col_unpool)),
        shape=(n_high, n_low)
    ).tocoo()

    row_sums = np.array(P.sum(axis=1)).ravel()

    assert np.allclose(row_sums, 1.0, atol=1e-6)

    return P, U


# ============================================================
# Save utilities
# ============================================================

def save_array(data, filename, output_dir, prefix=""):

    if not filename.endswith(".npy"):
        filename += ".npy"

    path = os.path.join(output_dir, prefix + filename)

    np.save(path, data)

    print("Saved:", path)


def save_sparse(matrix, filename, output_dir, prefix=""):

    if not filename.endswith(".npz"):
        filename += ".npz"

    path = os.path.join(output_dir, prefix + filename)

    coo = matrix.tocoo()

    np.savez(
        path,
        row=coo.row,
        col=coo.col,
        data=coo.data,
        shape=coo.shape
    )

    print("Saved:", path)


# ============================================================
# Single-level graph builder
# ============================================================

def build_level_graph(
    lat_2d,
    lon_2d,
    level_name,
    sphere_radius=1.0,
    laplacian_type="scaled_normalized",
    k_transport=4
):
    """
    Builds one graph resolution level.
    """

    assert lat_2d.shape == lon_2d.shape
    assert lat_2d.ndim == 2

    n_lat, n_lon = lat_2d.shape

    lat = lat_2d.reshape(-1).astype(np.float32)
    lon = lon_2d.reshape(-1).astype(np.float32)

    coords = longlat_to_cartesian(
        np.stack([lon, lat], axis=1),
        radius=sphere_radius
    )

    edge_index, edge_type = build_structured_edges(
        n_lat,
        n_lon
    )

    edge_vectors, edge_distance = compute_edge_geometry(
        coords,
        edge_index
    )

    transport_edges = build_transport_edges(
        coords,
        k_transport=k_transport
    )

    L, A, t_kernel = build_physics_graph(
        coords=coords,
        edge_index=edge_index,
        edge_distance=edge_distance,
        edge_type=edge_type,
        laplacian_type=laplacian_type
    )

    print(f"\n{level_name.upper()} GRAPH")
    print(f"  grid            : ({n_lat}, {n_lon})")
    print(f"  nodes           : {lat.size}")
    print(f"  stencil edges   : {edge_index.shape[1]}")
    print(f"  transport edges : {transport_edges.shape[1]}")
    print(f"  t_kernel        : {t_kernel:.6e}")

    return {
        f"lat_{level_name}": lat,
        f"lon_{level_name}": lon,
        f"coords_{level_name}": coords,

        f"L_{level_name}": L,
        f"A_{level_name}": A,

        f"edge_index_{level_name}": edge_index,
        f"edge_type_{level_name}": edge_type,
        f"edge_vectors_{level_name}": edge_vectors,
        f"edge_distance_{level_name}": edge_distance,
        f"transport_edges_{level_name}": transport_edges,

        f"t_kernel_{level_name}": float(t_kernel),

        f"n_{level_name}": int(lat.size),
        f"n_lat_{level_name}": int(n_lat),
        f"n_lon_{level_name}": int(n_lon),
    }


# ============================================================
# Master multi-resolution graph builder
# ============================================================

def compute_spatial_structures(
    lat_grid,
    lon_grid,
    scale_factor=2,
    use_unit_sphere=True,
    sphere_radius=None,
    save_dir="/content/deepsphere_resnet_clean/artifacts/static_structures",
    file_prefix="q_",
    laplacian_type="scaled_normalized",
    k_transport=4
):
    """
    Builds HR, MR, LR graph hierarchy.

    Required by:
        LR -> MR training
        MR -> HR evaluation
        future flow-aware dynamic message passing
    """

    assert lat_grid.shape == lon_grid.shape
    assert lat_grid.ndim == 2

    os.makedirs(save_dir, exist_ok=True)

    n_lat, n_lon = lat_grid.shape

    n_lat_hr = n_lat - (n_lat % (scale_factor ** 2))
    n_lon_hr = n_lon - (n_lon % (scale_factor ** 2))

    assert n_lat_hr > 0
    assert n_lon_hr > 0

    lat_hr_2d = lat_grid[:n_lat_hr, :n_lon_hr]
    lon_hr_2d = lon_grid[:n_lat_hr, :n_lon_hr]

    lat_mr_2d = lat_hr_2d[::scale_factor, ::scale_factor]
    lon_mr_2d = lon_hr_2d[::scale_factor, ::scale_factor]

    lat_lr_2d = lat_hr_2d[::scale_factor ** 2, ::scale_factor ** 2]
    lon_lr_2d = lon_hr_2d[::scale_factor ** 2, ::scale_factor ** 2]

    graph_radius = 1.0 if use_unit_sphere else sphere_radius

    if graph_radius is None:
        raise ValueError(
            "sphere_radius must be provided when use_unit_sphere=False"
        )

    print("=" * 70)
    print("PHYSICS-AWARE MULTI-RESOLUTION SPHERICAL GRAPH")
    print("=" * 70)

    print(f"Original grid : {lat_grid.shape}")
    print(f"HR grid       : {lat_hr_2d.shape}")
    print(f"MR grid       : {lat_mr_2d.shape}")
    print(f"LR grid       : {lat_lr_2d.shape}")

    hr = build_level_graph(
        lat_hr_2d,
        lon_hr_2d,
        "hr",
        sphere_radius=graph_radius,
        laplacian_type=laplacian_type,
        k_transport=k_transport
    )

    mr = build_level_graph(
        lat_mr_2d,
        lon_mr_2d,
        "mr",
        sphere_radius=graph_radius,
        laplacian_type=laplacian_type,
        k_transport=k_transport
    )

    lr = build_level_graph(
        lat_lr_2d,
        lon_lr_2d,
        "lr",
        sphere_radius=graph_radius,
        laplacian_type=laplacian_type,
        k_transport=k_transport
    )

    pool_hr_mr, unpool_mr_hr = create_pooling_matrix(
        n_lat_low=mr["n_lat_mr"],
        n_lon_low=mr["n_lon_mr"],
        lat_high_2d=lat_hr_2d,
        scale_factor=scale_factor
    )

    pool_mr_lr, unpool_lr_mr = create_pooling_matrix(
        n_lat_low=lr["n_lat_lr"],
        n_lon_low=lr["n_lon_lr"],
        lat_high_2d=lat_mr_2d,
        scale_factor=scale_factor
    )

    spatial_dict = {
        **hr,
        **mr,
        **lr,

        "scale_factor": int(scale_factor),
        "laplacian_type": laplacian_type,
        "graph_type": "structured_spherical_transport_ready",

        "use_unit_sphere": bool(use_unit_sphere),
        "sphere_radius": float(graph_radius),

        "k_transport": int(k_transport),

        "pool_hr_mr": pool_hr_mr,
        "unpool_mr_hr": unpool_mr_hr,
        "pool_mr_lr": pool_mr_lr,
        "unpool_lr_mr": unpool_lr_mr,
    }

    # ========================================================
    # Save outputs
    # ========================================================

    for level in ["hr", "mr", "lr"]:

        save_array(
            spatial_dict[f"lat_{level}"],
            f"{level}_lat",
            save_dir,
            file_prefix
        )

        save_array(
            spatial_dict[f"lon_{level}"],
            f"{level}_lon",
            save_dir,
            file_prefix
        )

        save_array(
            spatial_dict[f"coords_{level}"],
            f"{level}_coords",
            save_dir,
            file_prefix
        )

        save_array(
            spatial_dict[f"edge_index_{level}"],
            f"{level}_edge_index",
            save_dir,
            file_prefix
        )

        save_array(
            spatial_dict[f"edge_type_{level}"],
            f"{level}_edge_type",
            save_dir,
            file_prefix
        )

        save_array(
            spatial_dict[f"edge_vectors_{level}"],
            f"{level}_edge_vectors",
            save_dir,
            file_prefix
        )

        save_array(
            spatial_dict[f"edge_distance_{level}"],
            f"{level}_edge_distance",
            save_dir,
            file_prefix
        )

        save_array(
            spatial_dict[f"transport_edges_{level}"],
            f"{level}_transport_edges",
            save_dir,
            file_prefix
        )

        save_array(
            np.array(spatial_dict[f"t_kernel_{level}"]),
            f"{level}_t_kernel",
            save_dir,
            file_prefix
        )

        save_sparse(
            spatial_dict[f"L_{level}"],
            f"{level}_L",
            save_dir,
            file_prefix
        )

        save_sparse(
            spatial_dict[f"A_{level}"],
            f"{level}_A",
            save_dir,
            file_prefix
        )

    save_sparse(pool_hr_mr, "pool_hr_mr", save_dir, file_prefix)
    save_sparse(unpool_mr_hr, "unpool_mr_hr", save_dir, file_prefix)
    save_sparse(pool_mr_lr, "pool_mr_lr", save_dir, file_prefix)
    save_sparse(unpool_lr_mr, "unpool_lr_mr", save_dir, file_prefix)

    metadata = {
        "n_hr": spatial_dict["n_hr"],
        "n_mr": spatial_dict["n_mr"],
        "n_lr": spatial_dict["n_lr"],

        "n_lat_hr": spatial_dict["n_lat_hr"],
        "n_lon_hr": spatial_dict["n_lon_hr"],
        "n_lat_mr": spatial_dict["n_lat_mr"],
        "n_lon_mr": spatial_dict["n_lon_mr"],
        "n_lat_lr": spatial_dict["n_lat_lr"],
        "n_lon_lr": spatial_dict["n_lon_lr"],

        "scale_factor": int(scale_factor),
        "laplacian_type": laplacian_type,
        "graph_type": "structured_spherical_transport_ready",

        "use_unit_sphere": bool(use_unit_sphere),
        "sphere_radius": float(graph_radius),
        "k_transport": int(k_transport),

        "num_edges_hr": int(spatial_dict["edge_index_hr"].shape[1]),
        "num_edges_mr": int(spatial_dict["edge_index_mr"].shape[1]),
        "num_edges_lr": int(spatial_dict["edge_index_lr"].shape[1]),

        "num_transport_edges_hr": int(spatial_dict["transport_edges_hr"].shape[1]),
        "num_transport_edges_mr": int(spatial_dict["transport_edges_mr"].shape[1]),
        "num_transport_edges_lr": int(spatial_dict["transport_edges_lr"].shape[1]),
    }

    with open(
        os.path.join(save_dir, f"{file_prefix}spatial_dict.json"),
        "w"
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\nCELL 5 PASSED")
    print("HR nodes:", spatial_dict["n_hr"])
    print("MR nodes:", spatial_dict["n_mr"])
    print("LR nodes:", spatial_dict["n_lr"])

    return spatial_dict
