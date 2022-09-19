# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable

import torch
from torch import nn
from torch_scatter import scatter
from torch_sparse import SparseTensor
import torch.nn.functional as F
from math import sqrt, pi as PI
from torch.nn import MultiheadAttention, TransformerEncoderLayer
from torch_geometric.nn import GATConv, radius_graph, DimeNet
from torch_geometric.nn.acts import swish
from torch_geometric.nn.inits import glorot_orthogonal


class RotomerLdq(nn.Module):
    def __init__(self, scale_factor: int, radius_distance: float, cal_dim: int, GAT_dim: int, GAT_head: int,
                 GAT_layers: int):
        super().__init__()
        self.radius_distance = radius_distance
        self.cal_dim = cal_dim
        self.GAT_layers = GAT_layers
        self.res_embed = nn.Embedding(20, 4 * scale_factor)
        self.atom_embed = nn.Embedding(5, 1 * scale_factor)
        self.count_embed = nn.Embedding(21, 4 * scale_factor)
        self.xyz_embed = nn.Linear(3, 23 * scale_factor)

        self.act = nn.Hardswish()

        self.CoefficientGAT = nn.ModuleList([
            GATConv(32 * scale_factor, GAT_dim, GAT_head, concat=False)
        ])
        self.BiasGAT = nn.ModuleList([
            GATConv(32 * scale_factor, GAT_dim, GAT_head, concat=False)
        ])
        self.ABiasGAT = nn.ModuleList([
            GATConv(32 * scale_factor, GAT_dim, GAT_head, concat=False)
        ])
        for layer_i in range(GAT_layers - 1):
            self.CoefficientGAT.append(
                    GATConv(GAT_dim, GAT_dim, GAT_head, concat=False)
            )
            self.BiasGAT.append(
                    GATConv(GAT_dim, GAT_dim, GAT_head, concat=False)
            )
            self.ABiasGAT.append(
                    GATConv(GAT_dim, GAT_dim, GAT_head, concat=False)
            )
        self.CoefficientOutput = nn.Linear(GAT_dim, cal_dim)
        self.BiasOutput = nn.Linear(GAT_dim, cal_dim)
        self.ABiasOutput = nn.Linear(GAT_dim, cal_dim)

    def forward(self, batch_data):
        x, pos, edge_index, batch, edge_attr = batch_data.x, batch_data.pos, batch_data.edge_index, batch_data.batch, batch_data.edge_attr
        res_idx, atom_idx, count_idx, xyz = x[:, 0].long(), x[:, 1].long(), x[:, 2].long(), pos
        x = torch.cat([self.res_embed(res_idx),
                       self.atom_embed(atom_idx),
                       self.count_embed(count_idx),
                       self.xyz_embed(xyz)], dim=-1)
        radius_index = radius_graph(xyz, r=self.radius_distance, batch=batch, max_num_neighbors=x.shape[0])
        radius_index_batch = batch[radius_index[0]]

        edge_index_with_radius = torch.cat([edge_index, radius_index], dim=-1)

        C, B, A = x, x, x
        for i in range(self.GAT_layers):
            CoeLayer = self.CoefficientGAT[i]
            BiaLayer = self.BiasGAT[i]
            ABiaLayer = self.ABiasGAT[i]
            C = CoeLayer(C, edge_index_with_radius)
            B = BiaLayer(B, edge_index_with_radius)
            A = ABiaLayer(A, edge_index_with_radius)
            C = self.act(C)
            B = self.act(B)
            edge_index_with_radius = radius_index

        C = self.CoefficientOutput(C)
        C = self.act(C)
        B = self.BiasOutput(B)
        A = self.ABiasOutput(A)

        C = (C[radius_index[0]] + C[radius_index[1]]) / 2
        B = (B[radius_index[0]] + B[radius_index[1]]) / 2
        A = (A[radius_index[0]] + A[radius_index[1]]) / 2

        dist = torch.sum(torch.pow(torch.abs(xyz[radius_index[0]] - xyz[radius_index[1]]), 2), dim=-1)
        dist_matrix = torch.stack([torch.pow(dist, i) for i in range(self.cal_dim)], dim=-1)
        energy_matrix = torch.sum(C * (1 / dist_matrix + B) + A, dim=-1)

        energy = torch.zeros(batch_data.num_graphs).to(x.device)

        for i in range(batch_data.num_graphs):
            energy[i] += torch.sum(energy_matrix[radius_index_batch == i])

        return energy.view(-1, 1) / self.cal_dim


class RotomerTransformer(nn.Module):
    def __init__(self, h_dim: int, h_head: int, num_layers: int):
        super().__init__()
        scale_factor = h_dim // 256
        self.h_dim = h_dim
        self.res_embed = nn.Embedding(20, 28 * scale_factor)
        self.atom_embed = nn.Embedding(5, 28 * scale_factor)
        self.count_embed = nn.Embedding(21, 28 * scale_factor)
        self.xyz_embed = nn.Linear(3, 172 * scale_factor)

        self.norms = nn.ModuleList(
                [nn.LayerNorm(h_dim) for _ in range(num_layers)]
        )
        self.layers = nn.ModuleList(
                [TransformerEncoderLayer(h_dim, h_head) for _ in range(num_layers)]
        )

        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)

    def forward(self, batch_data):
        x, pos, edge_index, batch, edge_attr = batch_data.x, batch_data.pos, batch_data.edge_index, batch_data.batch, batch_data.edge_attr
        res_idx, atom_idx, count_idx, xyz = x[:, 0].long(), x[:, 1].long(), x[:, 2].long(), pos
        x = torch.cat([self.res_embed(res_idx),
                       self.atom_embed(atom_idx),
                       self.count_embed(count_idx),
                       self.xyz_embed(xyz / 10)], dim=-1)

        x = x.reshape((batch_data.num_graphs, batch.shape[0] // batch_data.num_graphs, self.h_dim))
        x = x.transpose(0, 1)

        for i, layer in enumerate(self.layers):
            x = self.norms[i](x)
            x = layer(x)

        x_mean, _ = x.max(dim=0)
        hidden = F.relu(self.fc1(x_mean))
        energy = self.fc2(hidden)

        return energy


class RotomerDimeNet(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int,
                 num_blocks: int, num_bilinear: int, num_spherical: int,
                 num_radial, cutoff: float = 5.0, max_num_neighbors: int = 32,
                 envelope_exponent: int = 5, num_before_skip: int = 1,
                 num_after_skip: int = 2, num_output_layers: int = 3,
                 act: Callable = swish):
        super(RotomerDimeNet, self).__init__()
        self.DimeNet = DimeNet(hidden_channels, out_channels, num_blocks, num_bilinear, num_spherical, num_radial,
                               cutoff, max_num_neighbors, envelope_exponent, num_before_skip, num_after_skip,
                               num_output_layers, act)

    def forward(self, batch_data):
        x, pos, edge_index, batch, edge_attr = batch_data.x, batch_data.pos, batch_data.edge_index, \
                                               batch_data.batch, batch_data.edge_attr
        return self.DimeNet(x[:, 1].long(), pos, batch)


class RotomerTransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, embed_dim, encoder_attention_heads, encoder_ffn_embed_dim: int = 1024,
                 attention_dropout: float = 0.0, dropout: float = 0.0, relu_dropout: float = 0.0,
                 encoder_normalize_before=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(
                self.embed_dim, encoder_attention_heads, dropout=attention_dropout,
        )
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.normalize_before = encoder_normalize_before
        self.fc1 = nn.Linear(self.embed_dim, encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(encoder_ffn_embed_dim, self.embed_dim)

        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class DimeNetPlus(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int,
                 num_blocks: int, num_bilinear: int, num_spherical: int,
                 num_radial, cutoff: float = 5.0, max_num_neighbors: int = 32,
                 envelope_exponent: int = 5, num_before_skip: int = 1,
                 num_after_skip: int = 2, num_output_layers: int = 3,
                 act: Callable = swish, smooth_factor=0.75):
        super(DimeNetPlus, self).__init__()

        self.smooth_factor = smooth_factor
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff,
                                       envelope_exponent)

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.output_blocks = torch.nn.ModuleList([
            OutputBlock(num_radial, hidden_channels, out_channels,
                        num_output_layers, act) for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionBlock(hidden_channels, num_bilinear, num_spherical,
                             num_radial, num_before_skip, num_after_skip, act)
            for _ in range(num_blocks)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def triplets(self, edge_index, num_nodes):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, batch_data):
        z, pos, batch = batch_data.x.long(), batch_data.pos, batch_data.batch
        bonds = batch_data.edge_index
        edge_select = z[:, 3]
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        select_edge_index = edge_index[:, edge_select[edge_index[0]] == 1]
        del edge_index

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
                select_edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt() + self.smooth_factor

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i, num_nodes=pos.size(0))

        return P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)


class DimeNetPlusPlus(DimeNetPlus):
    def __init__(self, hidden_channels: int, out_channels: int,
                 num_blocks: int, int_emb_size: int, basis_emb_size: int,
                 out_emb_channels: int, num_spherical: int, num_radial: int,
                 cutoff: float = 5.0, max_num_neighbors: int = 32,
                 envelope_exponent: int = 5, num_before_skip: int = 1,
                 num_after_skip: int = 2, num_output_layers: int = 3,
                 act: Callable = swish, smooth_factor=0.75):
        super().__init__(
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                num_bilinear=1,
                num_spherical=num_spherical,
                num_radial=num_radial,
                cutoff=cutoff,
                max_num_neighbors=max_num_neighbors,
                envelope_exponent=envelope_exponent,
                num_before_skip=num_before_skip,
                num_after_skip=num_after_skip,
                num_output_layers=num_output_layers,
                act=act,
                smooth_factor=smooth_factor,
        )
        # We are re-using the RBF, SBF and embedding layers of `DimeNet` and
        # redefine output_block and interaction_block in DimeNet++.
        # Hence, it is to be noted that in the above initalization, the
        # variable `num_bilinear` does not have any purpose as it is used
        # solely in the `OutputBlock` of DimeNet:
        self.output_blocks = torch.nn.ModuleList([
            OutputPPBlock(num_radial, hidden_channels, out_emb_channels,
                          out_channels, num_output_layers, act)
            for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionPPBlock(hidden_channels, int_emb_size, basis_emb_size,
                               num_spherical, num_radial, num_before_skip,
                               num_after_skip, act) for _ in range(num_blocks)
        ])

        self.reset_parameters()


class DimeNetPlusPlusGraph(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int,
                 num_blocks: int, int_emb_size: int, basis_emb_size: int,
                 out_emb_channels: int, num_spherical: int, num_radial: int,
                 heads: int, bond_channels: int, smooth_factor: float,
                 cutoff: float = 5.0, max_num_neighbors: int = 32,
                 envelope_exponent: int = 5, num_before_skip: int = 1,
                 num_after_skip: int = 2, num_output_layers: int = 3,
                 act: Callable = swish):
        super().__init__()

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks
        self.smooth_factor = smooth_factor

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff,
                                       envelope_exponent)
        self.emb = EmbeddingPPGBlock(num_radial, hidden_channels, act)

        self.bond_blocks = torch.nn.ModuleList([
            GraphRbfBlock(hidden_channels, num_radial, bond_channels, heads, act)
            for _ in range(num_blocks)
        ])

        self.output_blocks = torch.nn.ModuleList([
            OutputPPBlock(num_radial, hidden_channels, out_emb_channels,
                          out_channels, num_output_layers, act)
            for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionPPGBlock(hidden_channels, int_emb_size, basis_emb_size,
                                num_spherical, num_radial, num_before_skip,
                                num_after_skip, act, bond_channels) for _ in range(num_blocks)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def triplets(self, edge_index, num_nodes):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, batch_data):
        z, pos, batch = batch_data.x.long(), batch_data.pos, batch_data.batch
        bonds = batch_data.edge_index
        edge_select = z[:, 3]
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        select_edge_index = edge_index[:, edge_select[edge_index[0]] == 1]

        del edge_index

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
                select_edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist+self.smooth_factor)
        sbf = self.sbf(dist+self.smooth_factor, angle, idx_kj)
        # sbf = torch.clamp(self.sbf(dist, angle, idx_kj), min=-1e3, max=1e3)

        # Embedding block.
        x, emb = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        # noinspection PyTypeChecker
        for interaction_block, bond_block, output_block in zip(self.interaction_blocks,
                                                               self.bond_blocks,
                                                               self.output_blocks[1:],
                                                               ):
            bond_x, emb = bond_block(emb, bonds, rbf, i, j)
            x = interaction_block(x, torch.cat([rbf, bond_x], dim=-1), sbf, idx_kj, idx_ji)
            # DimeNetPlusPlusGraphNoStrict2
            # bond_x, _ = bond_block(emb, bonds, rbf, i, j)
            # x = interaction_block(x, torch.cat([rbf, bond_x], dim=-1), sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i, num_nodes=pos.size(0))

        return P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)


class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super(BesselBasisLayer, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class SphericalBasisLayer(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super(SphericalBasisLayer, self).__init__()
        import sympy as sym
        from torch_geometric.nn.models.dimenet_utils import (bessel_basis,
                                                             real_sph_harm)

        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish):
        super(EmbeddingBlock, self).__init__()
        self.act = act
        factor = int(hidden_channels / 4)
        self.resi_emb = nn.Embedding(20, factor * 2)
        self.atom_emb = nn.Embedding(95, factor * 2)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels)
        self.lin = nn.Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.resi_emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.atom_emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, rbf, i, j):
        x = torch.cat([
            self.resi_emb(x[:, 0]),
            self.atom_emb(x[:, 1]),
        ], dim=-1)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class EmbeddingPPGBlock(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish):
        super().__init__()
        self.act = act
        factor = int(hidden_channels / 4)
        self.resi_emb = nn.Embedding(20, factor * 2)
        self.atom_emb = nn.Embedding(95, factor * 2)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels)
        self.lin = nn.Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.resi_emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.atom_emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, rbf, i, j):
        x = torch.cat([
            self.resi_emb(x[:, 0]),
            self.atom_emb(x[:, 1]),
        ], dim=-1)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1))), x


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_bilinear, num_spherical,
                 num_radial, num_before_skip, num_after_skip, act=swish):
        super(InteractionBlock, self).__init__()
        self.act = act

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)
        self.lin_sbf = nn.Linear(num_spherical * num_radial, num_bilinear,
                                 bias=False)

        # Dense transformations of input messages.
        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.W = torch.nn.Parameter(
                torch.Tensor(hidden_channels, num_bilinear, hidden_channels))

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)
        self.W.data.normal_(mean=0, std=2 / self.W.size(0))
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        rbf = self.lin_rbf(rbf)
        sbf = self.lin_sbf(sbf)

        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        x_kj = x_kj * rbf
        x_kj = torch.einsum('wj,wl,ijl->wi', sbf, x_kj[idx_kj], self.W)
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputBlock(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, out_channels, num_layers,
                 act=swish):
        super(OutputBlock, self).__init__()
        self.act = act

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(0)

    def forward(self, x, rbf, i, num_nodes=None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class InteractionPPBlock(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size,
                 num_spherical, num_radial, num_before_skip, num_after_skip,
                 act):
        super().__init__()
        self.act = act

        # Transformation of Bessel and spherical basis representations:
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size,
                                  bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)

        # Hidden transformation of input message:
        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets:
        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before and after skip connection:
        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        # Initial transformation:
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transformation via Bessel basis:
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down project embedding and generating triple-interactions:
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis:
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings:
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputPPBlock(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, out_emb_channels,
                 out_channels, num_layers, act):
        super().__init__()
        self.act = act

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        # The up-projection layer:
        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(0)

    def forward(self, x, rbf, i, num_nodes=None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class GraphRbfBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_radial, bond_channels, head, act):
        super().__init__()
        self.head = head
        self.hidden_channels = hidden_channels
        self.act = act
        self.GAT = GATConv(hidden_channels, hidden_channels, head)
        self.norm = nn.InstanceNorm1d(hidden_channels)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels)
        self.lin_GAT = nn.Linear(head * hidden_channels, hidden_channels)
        self.lin_output = nn.Linear(3 * hidden_channels, bond_channels)

    def reset_parameters(self):
        self.lin_rbf.reset_parameters()
        self.lin_GAT.reset_parameters()
        self.lin_output.reset_parameters()

    def forward(self, emb, edge_index, rbf, i, j):
        x = self.GAT(emb, edge_index)
        rbf = self.act(self.lin_rbf(rbf))
        x = x.reshape(-1, self.head, self.hidden_channels)
        x = self.act(
                self.lin_GAT(
                        self.norm(
                                x + emb.reshape(-1, 1, self.hidden_channels).repeat(1, self.head, 1)
                        ).reshape(-1, self.head * self.hidden_channels)))
        return self.act(self.lin_output(torch.cat([x[i], x[j], rbf], dim=-1))), x


class InteractionPPGBlock(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size,
                 num_spherical, num_radial, num_before_skip, num_after_skip,
                 act, bond_channels):
        super().__init__()
        self.act = act

        # Transformation of Bessel and spherical basis representations:
        self.lin_rbf1 = nn.Linear(num_radial + bond_channels, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size,
                                  bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)

        # Hidden transformation of input message:
        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets:
        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before and after skip connection:
        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        # Initial transformation:
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transformation via Bessel basis:
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down project embedding and generating triple-interactions:
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis:
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings:
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h
