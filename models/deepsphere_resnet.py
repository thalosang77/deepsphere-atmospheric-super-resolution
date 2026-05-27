# ============================================================
# Physics-aware DeepSphereResNet model block
#
# Key version:
#   prediction = bilinear(input) + learned graph residual
#
# Important edit:
#   residual_scale bottleneck removed from forward pass
#   so the residual branch can fully correct the bilinear baseline.
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Geographical embedding
# ============================================================

class GeographicalEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim=8, use_embedding=True):
        super().__init__()

        self.use_embedding = use_embedding

        if use_embedding:
            self.proj = nn.Linear(4, embed_dim)
            self.out_channels = in_channels + embed_dim
        else:
            self.out_channels = in_channels

    def forward(self, x, lon, lat):
        if not self.use_embedding:
            return x

        lon = torch.as_tensor(lon, dtype=torch.float32, device=x.device)
        lat = torch.as_tensor(lat, dtype=torch.float32, device=x.device)

        lon_rad = torch.deg2rad(lon)
        lat_rad = torch.deg2rad(lat)

        geo_features = torch.stack(
            [
                torch.sin(lat_rad),
                torch.cos(lat_rad),
                torch.sin(lon_rad),
                torch.cos(lon_rad),
            ],
            dim=-1
        ).to(x.dtype)

        geo_emb = self.proj(geo_features)
        geo_emb = geo_emb.unsqueeze(0).expand(x.shape[0], -1, -1)

        return torch.cat([x, geo_emb], dim=-1)


# ============================================================
# Sparse Chebyshev Spherical Graph Convolution
# ============================================================

class SphericalGCN(nn.Module):
    def __init__(self, in_channels, out_channels, L_sparse, poly_order):
        super().__init__()

        assert poly_order >= 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.poly_order = poly_order

        self.cheb_weights = nn.Parameter(
            torch.empty(poly_order, in_channels, out_channels)
        )

        nn.init.xavier_uniform_(self.cheb_weights)

        L_sparse = L_sparse.tocoo()

        self.register_buffer(
            "L_indices",
            torch.LongTensor(np.vstack([L_sparse.row, L_sparse.col]))
        )

        self.register_buffer(
            "L_values",
            torch.FloatTensor(L_sparse.data)
        )

        self.register_buffer(
            "L_shape",
            torch.LongTensor(list(L_sparse.shape))
        )

    def _build_sparse_tensor(self, device):
        return torch.sparse_coo_tensor(
            self.L_indices,
            self.L_values,
            tuple(self.L_shape.tolist()),
            device=device
        ).coalesce()

    def _sparse_multiply(self, sparse_mat, x):
        original_dtype = x.dtype
        B, N, C = x.shape

        with torch.cuda.amp.autocast(enabled=False):
            x_t = x.float().permute(1, 0, 2).reshape(N, B * C)
            sparse_mat = sparse_mat.float()
            y_t = torch.sparse.mm(sparse_mat, x_t)
            y = y_t.reshape(N, B, C).permute(1, 0, 2)

        return y.to(original_dtype)

    def forward(self, x):
        B, N, C_in = x.shape

        assert C_in == self.in_channels

        L = self._build_sparse_tensor(x.device)

        Tx_0 = x

        out = torch.matmul(
            Tx_0,
            self.cheb_weights[0].to(x.dtype)
        )

        if self.poly_order > 1:
            Tx_1 = self._sparse_multiply(L, x)

            out = out + torch.matmul(
                Tx_1,
                self.cheb_weights[1].to(x.dtype)
            )

            for k in range(2, self.poly_order):
                Tx_k = 2.0 * self._sparse_multiply(L, Tx_1) - Tx_0

                out = out + torch.matmul(
                    Tx_k,
                    self.cheb_weights[k].to(x.dtype)
                )

                Tx_0, Tx_1 = Tx_1, Tx_k

        return out


# ============================================================
# CNN-like local spherical stencil convolution
# ============================================================

class LocalSphericalStencilConv(nn.Module):
    """
    CNN-like local convolution on the spherical graph stencil.

    Edge types:
        0 = north/south
        1 = east/west
        2 = diagonal
    """

    def __init__(
        self,
        channels,
        edge_index,
        edge_type,
        dropout_rate=0.1
    ):
        super().__init__()

        self.channels = channels

        edge_index = torch.LongTensor(edge_index)
        edge_type = torch.LongTensor(edge_type)

        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)

        self.src = edge_index[0]
        self.dst = edge_index[1]

        self.center_kernel = nn.Linear(channels, channels, bias=True)
        self.ns_kernel = nn.Linear(channels, channels, bias=False)
        self.ew_kernel = nn.Linear(channels, channels, bias=False)
        self.diag_kernel = nn.Linear(channels, channels, bias=False)

        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        # Strengthened local CNN-like branch.
        self.branch_scale = nn.Parameter(
            torch.tensor(0.3, dtype=torch.float32)
        )

    def forward(self, x):
        B, N, C = x.shape

        assert C == self.channels

        src = self.src.to(x.device)
        dst = self.dst.to(x.device)
        etype = self.edge_type.to(x.device)

        out = self.center_kernel(x)

        for edge_class, kernel in [
            (0, self.ns_kernel),
            (1, self.ew_kernel),
            (2, self.diag_kernel),
        ]:
            mask = etype == edge_class

            if not torch.any(mask):
                continue

            src_k = src[mask]
            dst_k = dst[mask]

            msg = kernel(x[:, src_k, :])

            accum = torch.zeros(
                B,
                N,
                C,
                device=x.device,
                dtype=x.dtype
            )

            for b in range(B):
                accum[b].scatter_add_(
                    0,
                    dst_k.unsqueeze(-1).expand_as(msg[b]),
                    msg[b]
                )

            degree = torch.zeros(
                N,
                device=x.device,
                dtype=x.dtype
            )

            degree.scatter_add_(
                0,
                dst_k,
                torch.ones_like(dst_k, dtype=x.dtype)
            )

            accum = accum / degree.clamp_min(1.0).view(1, N, 1)

            out = out + accum

        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)

        return self.branch_scale.to(x.dtype) * out


# ============================================================
# Static DeepSphere residual block
# ============================================================

class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        L_sparse,
        dropout_rate=0.1,
        poly_order=3
    ):
        super().__init__()

        self.conv1 = SphericalGCN(
            channels,
            channels,
            L_sparse,
            poly_order
        )

        self.norm1 = nn.LayerNorm(channels)

        self.conv2 = SphericalGCN(
            channels,
            channels,
            L_sparse,
            poly_order
        )

        self.norm2 = nn.LayerNorm(channels)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.act(out)

        return out


# ============================================================
# Segment softmax
# ============================================================

def segment_softmax(scores, dst, num_nodes):
    B, E = scores.shape
    alpha = torch.zeros_like(scores)

    for i in range(num_nodes):
        mask = dst == i

        if mask.any():
            alpha[:, mask] = torch.softmax(scores[:, mask], dim=1)

    return alpha


# ============================================================
# Flow-aware dynamic graph correction
# ============================================================

class FlowAwareGraphCorrection(nn.Module):
    def __init__(
        self,
        channels,
        cond_channels,
        edge_index,
        edge_vectors,
        edge_distance,
        hidden_dim=64,
        t_kernel_init=1.0
    ):
        super().__init__()

        self.channels = channels
        self.cond_channels = cond_channels

        edge_index = torch.LongTensor(edge_index)

        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_vectors", torch.FloatTensor(edge_vectors))
        self.register_buffer("edge_distance", torch.FloatTensor(edge_distance))

        self.src = edge_index[0]
        self.dst = edge_index[1]

        self.dynamic_proj = nn.Linear(channels, channels)

        self.edge_mlp = nn.Sequential(
            nn.Linear(
                2 * channels + 2 * cond_channels + 3 + 1,
                hidden_dim
            ),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.raw_t_kernel = nn.Parameter(
            torch.tensor(float(t_kernel_init))
        )

        self.wind_strength = nn.Parameter(
            torch.tensor(0.1, dtype=torch.float32)
        )

    def forward(self, x, cond=None):
        if cond is None:
            return torch.zeros_like(x)

        B, N, C = x.shape

        assert C == self.channels
        assert cond.shape[0] == B
        assert cond.shape[1] == N
        assert cond.shape[-1] == self.cond_channels

        src = self.src.to(x.device)
        dst = self.dst.to(x.device)

        edge_vectors = self.edge_vectors.to(x.device)
        edge_distance = self.edge_distance.to(x.device)

        E = src.numel()

        x_src = x[:, src, :]
        x_dst = x[:, dst, :]

        c_src = cond[:, src, :]
        c_dst = cond[:, dst, :]

        edge_vec = edge_vectors.unsqueeze(0).expand(B, E, 3)
        edge_dist = edge_distance.view(1, E, 1).expand(B, E, 1)

        t_kernel = F.softplus(self.raw_t_kernel) + 1e-6

        with torch.cuda.amp.autocast(enabled=False):
            distance_logit = (
                -edge_distance.float().pow(2)
                / (4.0 * t_kernel.float())
            ).view(1, E)

        distance_logit = distance_logit.to(x.dtype)

        directional_logit = torch.zeros(
            B,
            E,
            device=x.device,
            dtype=x.dtype
        )

        if self.cond_channels >= 2:
            wind_src = c_src[..., :2]
            wind_dst = c_dst[..., :2]
            wind_edge = 0.5 * (wind_src + wind_dst)

            edge_h = edge_vec[..., :2].to(x.dtype)
            wind_proj = torch.sum(wind_edge * edge_h, dim=-1)

            directional_logit = self.wind_strength.to(x.dtype) * torch.clamp(
                wind_proj,
                -5.0,
                5.0
            )

        edge_input = torch.cat(
            [
                x_src,
                x_dst,
                c_src,
                c_dst,
                edge_vec.to(x.dtype),
                edge_dist.to(x.dtype),
            ],
            dim=-1
        )

        mlp_logit = self.edge_mlp(edge_input).squeeze(-1)

        logits = distance_logit + directional_logit + mlp_logit
        logits = torch.clamp(logits, -20.0, 20.0)

        alpha = segment_softmax(
            logits.float(),
            dst,
            N
        ).to(x.dtype)

        msg = self.dynamic_proj(x_src)
        msg = msg * alpha.unsqueeze(-1)

        out = torch.zeros(
            B,
            N,
            C,
            device=x.device,
            dtype=x.dtype
        )

        for b in range(B):
            out[b].scatter_add_(
                0,
                dst.unsqueeze(-1).expand_as(msg[b]),
                msg[b]
            )

        return out


# ============================================================
# Physics-aware residual block
# ============================================================

class PhysicsAwareResBlock(nn.Module):
    def __init__(
        self,
        channels,
        L_sparse,
        dropout_rate=0.1,
        poly_order=3,

        use_flow_correction=False,
        cond_channels=None,

        edge_index=None,
        edge_type=None,
        edge_vectors=None,
        edge_distance=None,
        t_kernel_init=1.0,

        use_local_stencil=True
    ):
        super().__init__()

        self.use_flow_correction = use_flow_correction
        self.use_local_stencil = use_local_stencil

        self.static_block = ResBlock(
            channels=channels,
            L_sparse=L_sparse,
            dropout_rate=dropout_rate,
            poly_order=poly_order
        )

        if use_local_stencil:
            assert edge_index is not None
            assert edge_type is not None

            self.local_stencil = LocalSphericalStencilConv(
                channels=channels,
                edge_index=edge_index,
                edge_type=edge_type,
                dropout_rate=dropout_rate
            )

        if use_flow_correction:
            assert cond_channels is not None
            assert edge_index is not None
            assert edge_vectors is not None
            assert edge_distance is not None

            self.flow_correction = FlowAwareGraphCorrection(
                channels=channels,
                cond_channels=cond_channels,
                edge_index=edge_index,
                edge_vectors=edge_vectors,
                edge_distance=edge_distance,
                hidden_dim=64,
                t_kernel_init=t_kernel_init
            )

            self.flow_gate = nn.Sequential(
                nn.Linear(channels + cond_channels, channels),
                nn.Sigmoid()
            )

        self.mix = nn.Sequential(
            nn.LayerNorm(channels),
            nn.GELU()
        )

    def forward(self, x, cond=None):
        static_out = self.static_block(x)

        out = static_out

        if self.use_local_stencil:
            local_out = self.local_stencil(x)
            out = out + local_out

        if self.use_flow_correction and cond is not None:
            dynamic_out = self.flow_correction(x, cond=cond)

            gamma = self.flow_gate(
                torch.cat([x, cond], dim=-1)
            )

            out = gamma * out + (1.0 - gamma) * dynamic_out

        out = self.mix(out)

        return out


# ============================================================
# Explicit graph subpixel unpooling
# ============================================================

class SphericalSubpixelUnpool(nn.Module):
    def __init__(self, unpool_matrix, scale_factor=2):
        super().__init__()

        self.scale_factor = scale_factor
        self.n_children = scale_factor ** 2

        U = unpool_matrix.tocoo()
        N_high, N_low = U.shape

        child_order = np.full(N_high, -1, dtype=np.int64)
        parent_of_child = np.full(N_high, -1, dtype=np.int64)

        parent_counter = {}

        for row, col in zip(U.row, U.col):
            parent = int(col)
            child = int(row)

            if parent not in parent_counter:
                parent_counter[parent] = 0

            local_child = parent_counter[parent]
            parent_counter[parent] += 1

            child_order[child] = local_child
            parent_of_child[child] = parent

        assert np.all(child_order >= 0)
        assert np.all(parent_of_child >= 0)

        self.register_buffer(
            "parent_of_child",
            torch.LongTensor(parent_of_child)
        )

        self.register_buffer(
            "child_order",
            torch.LongTensor(child_order)
        )

        self.N_high = int(N_high)
        self.N_low = int(N_low)

    def forward(self, x):
        B, N_low, C_total = x.shape

        assert N_low == self.N_low
        assert C_total % self.n_children == 0

        C_out = C_total // self.n_children

        x = x.view(B, N_low, self.n_children, C_out)

        parent = self.parent_of_child.to(x.device)
        child = self.child_order.to(x.device)

        out = x[:, parent, child, :]

        return out


# ============================================================
# Sparse nearest-neighbour skip
# ============================================================

class SparseUnpoolSkip(nn.Module):
    def __init__(self, unpool_matrix):
        super().__init__()

        U = unpool_matrix.tocoo()

        self.register_buffer(
            "indices",
            torch.LongTensor(np.vstack([U.row, U.col]))
        )

        self.register_buffer(
            "values",
            torch.FloatTensor(U.data)
        )

        self.register_buffer(
            "shape",
            torch.LongTensor(list(U.shape))
        )

    def forward(self, x):
        original_dtype = x.dtype

        B, N_low, C = x.shape
        N_high = int(self.shape[0].item())

        with torch.cuda.amp.autocast(enabled=False):
            P = torch.sparse_coo_tensor(
                self.indices,
                self.values.float(),
                tuple(self.shape.tolist()),
                device=x.device
            ).coalesce()

            x_t = x.float().permute(1, 0, 2).reshape(N_low, B * C)
            y_t = torch.sparse.mm(P, x_t)
            y = y_t.reshape(N_high, B, C).permute(1, 0, 2)

        return y.to(original_dtype)


# ============================================================
# Bilinear grid skip
# ============================================================

class BilinearGridSkip(nn.Module):
    def __init__(
        self,
        n_lat_low,
        n_lon_low,
        n_lat_high,
        n_lon_high
    ):
        super().__init__()

        self.n_lat_low = int(n_lat_low)
        self.n_lon_low = int(n_lon_low)
        self.n_lat_high = int(n_lat_high)
        self.n_lon_high = int(n_lon_high)

    def forward(self, x):
        original_dtype = x.dtype

        B, N_low, C = x.shape
        expected_nodes = self.n_lat_low * self.n_lon_low

        assert N_low == expected_nodes

        with torch.cuda.amp.autocast(enabled=False):
            x_grid = x.float().reshape(
                B,
                self.n_lat_low,
                self.n_lon_low,
                C
            )

            x_grid = x_grid.permute(0, 3, 1, 2)

            y_grid = F.interpolate(
                x_grid,
                size=(self.n_lat_high, self.n_lon_high),
                mode="bilinear",
                align_corners=False
            )

            y_grid = y_grid.permute(0, 2, 3, 1)

            y = y_grid.reshape(
                B,
                self.n_lat_high * self.n_lon_high,
                C
            )

        return y.to(original_dtype)


# ============================================================
# DeepSphere-ResNet main model
# ============================================================

class DeepSphereResNet(nn.Module):
    def __init__(
        self,
        l_low,
        l_high,
        unpool_matrix,

        in_channels=37,
        out_channels=37,

        embed_dim=8,
        hidden_dim=64,
        n_resblock=2,
        poly_order=3,
        dropout_rate=0.1,

        scale_factor=2,
        use_absolute_embedding=True,
        use_hr_embedding=True,

        use_residual_skip=True,
        residual_scale_init=0.01,
        skip_type="bilinear",

        n_lat_low=None,
        n_lon_low=None,
        n_lat_high=None,
        n_lon_high=None,

        use_flow_blocks=False,
        cond_channels=None,

        use_local_stencil=True,

        edge_index_low=None,
        edge_type_low=None,
        edge_vectors_low=None,
        edge_distance_low=None,
        t_kernel_low=1.0
    ):
        super().__init__()

        assert in_channels == out_channels
        assert skip_type in ["nearest", "bilinear"]

        self.scale_factor = scale_factor
        self.n_children = scale_factor ** 2

        self.use_residual_skip = use_residual_skip
        self.skip_type = skip_type
        self.use_flow_blocks = use_flow_blocks
        self.use_hr_embedding = use_hr_embedding
        self.use_local_stencil = use_local_stencil

        self.coord_embed_low = GeographicalEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            use_embedding=use_absolute_embedding
        )

        input_ch = self.coord_embed_low.out_channels

        self.conv_in = SphericalGCN(
            input_ch,
            hidden_dim,
            l_low,
            poly_order
        )

        self.resblocks = nn.ModuleList()

        for _ in range(n_resblock):
            self.resblocks.append(
                PhysicsAwareResBlock(
                    channels=hidden_dim,
                    L_sparse=l_low,
                    dropout_rate=dropout_rate,
                    poly_order=poly_order,
                    use_flow_correction=use_flow_blocks,
                    cond_channels=cond_channels,
                    edge_index=edge_index_low,
                    edge_type=edge_type_low,
                    edge_vectors=edge_vectors_low,
                    edge_distance=edge_distance_low,
                    t_kernel_init=t_kernel_low,
                    use_local_stencil=use_local_stencil
                )
            )

        self.pre_upsample = SphericalGCN(
            hidden_dim,
            hidden_dim * self.n_children,
            l_low,
            poly_order=1
        )

        self.upsample = SphericalSubpixelUnpool(
            unpool_matrix=unpool_matrix,
            scale_factor=scale_factor
        )

        if use_hr_embedding:
            self.coord_embed_high = GeographicalEmbedding(
                in_channels=hidden_dim,
                embed_dim=embed_dim,
                use_embedding=True
            )

            high_input_ch = hidden_dim + embed_dim
        else:
            self.coord_embed_high = None
            high_input_ch = hidden_dim

        self.conv_out = SphericalGCN(
            high_input_ch,
            hidden_dim,
            l_high,
            poly_order
        )

        self.final_conv = SphericalGCN(
            hidden_dim,
            out_channels,
            l_high,
            poly_order=1
        )

        if self.use_residual_skip:
            if skip_type == "nearest":
                self.skip_unpool = SparseUnpoolSkip(unpool_matrix)
            else:
                assert n_lat_low is not None
                assert n_lon_low is not None
                assert n_lat_high is not None
                assert n_lon_high is not None

                self.skip_unpool = BilinearGridSkip(
                    n_lat_low=n_lat_low,
                    n_lon_low=n_lon_low,
                    n_lat_high=n_lat_high,
                    n_lon_high=n_lon_high
                )

            # Kept only for diagnostics/backward compatibility.
            # It is no longer used in forward().
            self.residual_scale = nn.Parameter(
                torch.ones(1, 1, out_channels)
                * float(residual_scale_init)
            )

    def forward(
        self,
        x_low,
        lon_low,
        lat_low,
        lon_high=None,
        lat_high=None,
        cond_low=None
    ):
        skip = None

        if self.use_residual_skip:
            skip = self.skip_unpool(x_low)

        out = self.coord_embed_low(
            x_low,
            lon_low,
            lat_low
        )

        out = self.conv_in(out)

        for block in self.resblocks:
            out = block(
                out,
                cond=cond_low
            )

        out = self.pre_upsample(out)
        out = self.upsample(out)

        if self.use_hr_embedding:
            assert lon_high is not None
            assert lat_high is not None

            out = self.coord_embed_high(
                out,
                lon_high,
                lat_high
            )

        out = self.conv_out(out)
        out = F.gelu(out)

        residual = self.final_conv(out)

        if self.use_residual_skip:
            return skip + residual

        return residual


# ============================================================
# Loss and metrics
# ============================================================

class SRLoss(nn.Module):
    def __init__(self, loss_type="mse"):
        super().__init__()

        self.loss_type = loss_type
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, sr_data, hr_data):
        if self.loss_type == "mse":
            return self.mse(sr_data, hr_data)

        elif self.loss_type == "mae":
            return self.mae(sr_data, hr_data)

        elif self.loss_type == "combined":
            return (
                0.5 * self.mse(sr_data, hr_data)
                +
                0.5 * self.mae(sr_data, hr_data)
            )

        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def compute_metrics(self, sr_data, hr_data):
        eps = 1e-9

        sr_data = sr_data.float()
        hr_data = hr_data.float()

        rmse = torch.sqrt(self.mse(sr_data, hr_data))
        mae = self.mae(sr_data, hr_data)

        mape = torch.mean(
            torch.abs(
                (hr_data - sr_data)
                /
                (torch.abs(hr_data) + eps)
            )
        ) * 100.0

        sr_flat = sr_data.reshape(-1)
        hr_flat = hr_data.reshape(-1)

        corr = torch.corrcoef(
            torch.stack([sr_flat, hr_flat])
        )[0, 1]

        data_range = torch.max(hr_data) - torch.min(hr_data)

        psnr = 20.0 * torch.log10(
            data_range / (rmse + eps)
        )

        ss_res = torch.sum((hr_data - sr_data) ** 2)
        ss_tot = torch.sum((hr_data - torch.mean(hr_data)) ** 2)

        r2 = 1.0 - (ss_res / (ss_tot + eps))

        return {
            "RMSE": rmse.item(),
            "MAE": mae.item(),
            "MAPE": mape.item(),
            "Corr": corr.item(),
            "PSNR": psnr.item(),
            "R2": r2.item(),
        }


print("CELL 9 PASSED")
