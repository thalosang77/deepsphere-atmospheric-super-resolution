# ------------------------------------------------------------
# Safe trainable-parameter transfer
# Skips graph-specific sparse buffers and mismatched tensors
# ------------------------------------------------------------

def transfer_weights_only(model, state_dict):

    model_dict = model.state_dict()

    transferred = {}
    skipped_graph = []
    skipped_shape = []
    skipped_missing = []

    GRAPH_KEYWORDS = [
        "L_",
        "indices",
        "values",
        "shape",
        "inv_indices",
        "edge_index",
        "edge_vectors",
        "edge_distance",
        "t_kernel",
        "unpool",
    ]

    for k, v in state_dict.items():

        # ----------------------------------------------------
        # Skip graph-specific buffers
        # ----------------------------------------------------
        if any(word in k for word in GRAPH_KEYWORDS):
            skipped_graph.append(k)
            continue

        # ----------------------------------------------------
        # Skip missing keys
        # ----------------------------------------------------
        if k not in model_dict:
            skipped_missing.append(k)
            continue

        # ----------------------------------------------------
        # Skip incompatible tensor shapes
        # ----------------------------------------------------
        if model_dict[k].shape != v.shape:
            skipped_shape.append(
                (k, tuple(v.shape), tuple(model_dict[k].shape))
            )
            continue

        transferred[k] = v

    # --------------------------------------------------------
    # Load transferable weights
    # --------------------------------------------------------
    missing, unexpected = model.load_state_dict(
        transferred,
        strict=False
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    print("=" * 70)
    print("WEIGHT TRANSFER SUMMARY")
    print("=" * 70)

    print("Transferred tensors :", len(transferred))
    print("Skipped graph tensors:", len(skipped_graph))
    print("Skipped shape mismatch:", len(skipped_shape))
    print("Skipped missing keys :", len(skipped_missing))
    print("Missing after load   :", len(missing))
    print("Unexpected after load:", len(unexpected))

    if len(skipped_shape) > 0:
        print("\nExample shape mismatches:")
        for item in skipped_shape[:5]:
            print(item)

    return model
