# ============================================================
# Build physics-aware spherical graph structures
# ============================================================

spatial_dict_q = compute_spatial_structures(
    lat_grid=lat_grid_2d,
    lon_grid=lon_grid_2d,
    scale_factor=2,
    use_unit_sphere=True,
    save_dir=str(STATIC_DIR),
    file_prefix="q_regular3h_",
    laplacian_type="scaled_normalized"
)

print("\n" + "=" * 70)
print("SPATIAL GRAPH SUMMARY")
print("=" * 70)

print(f"HR nodes              : {spatial_dict_q['n_hr']}")
print(f"MR nodes              : {spatial_dict_q['n_mr']}")
print(f"LR nodes              : {spatial_dict_q['n_lr']}")

print(f"HR edges              : {spatial_dict_q['edge_index_hr'].shape[1]}")
print(f"MR edges              : {spatial_dict_q['edge_index_mr'].shape[1]}")
print(f"LR edges              : {spatial_dict_q['edge_index_lr'].shape[1]}")

print(f"HR transport edges    : {spatial_dict_q['transport_edges_hr'].shape[1]}")
print(f"MR transport edges    : {spatial_dict_q['transport_edges_mr'].shape[1]}")
print(f"LR transport edges    : {spatial_dict_q['transport_edges_lr'].shape[1]}")

print(f"Laplacian type        : {spatial_dict_q['laplacian_type']}")
print(f"Graph type            : {spatial_dict_q['graph_type']}")
print(f"Sphere representation : {'unit sphere' if spatial_dict_q['use_unit_sphere'] else 'physical radius'}")

print("\nGrid hierarchy:")
print(f"HR grid               : {spatial_dict_q['n_lat_hr']} x {spatial_dict_q['n_lon_hr']}")
print(f"MR grid               : {spatial_dict_q['n_lat_mr']} x {spatial_dict_q['n_lon_mr']}")
print(f"LR grid               : {spatial_dict_q['n_lat_lr']} x {spatial_dict_q['n_lon_lr']}")

print("\nAvailable structures:")
for k in spatial_dict_q.keys():
    print(" -", k)

print("\nCELL 6 PASSED")
