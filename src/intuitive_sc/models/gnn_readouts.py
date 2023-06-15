"""
Code adapted from LineEvo: https://github.com/fate1997/LineEvo/tree/main
"""
from collections import defaultdict
from itertools import chain, combinations
from typing import Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.nn import global_add_pool, global_max_pool


class LineEvo(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        dropout: float = 0,
        num_layers: int = 1,
        if_pos: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LineEvoLayer(dim, dropout, if_pos))

    def forward(
        self, x: torch.Tensor, edge_index, edge_attr, pos, batch: torch.Tensor
    ) -> torch.Tensor:
        edges = torch.as_tensor(
            np.array(nx.from_edgelist(edge_index.T.tolist()).edges), device=x.device
        )

        mol_repr_all = 0
        for layer in self.layers:
            x, edges, edge_attr, pos, batch, mol_repr = layer(
                x, edges, edge_attr, pos, batch
            )
            mol_repr_all = mol_repr_all + mol_repr

        return mol_repr_all


class LineEvoLayer(nn.Module):
    def __init__(
        self,
        in_dim: int = 128,
        dim: int = 128,
        dropout: float = 0.1,
        if_pos: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.if_pos = if_pos

        # feature evolution
        self.linear = nn.Linear(in_dim, dim)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Parameter(torch.randn(1, dim))

        if self.if_pos:
            self.rbf_expand = RBFExpansion(0, 5, 6)
            self.linear_rbf = nn.Linear(6, dim, bias=False)

        self.init_params()
        # readout phase
        self.readout = ReadoutPhase(dim)

    def init_params(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn)
        nn.init.zeros_(self.linear.bias)

        if self.if_pos:
            nn.init.xavier_uniform_(self.linear_rbf.weight)

    def forward(
        self,
        x: torch.Tensor,
        edges: torch.Tensor,
        pos: Union[None, torch.Tensor],
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # create edges for isolated nodes
        num_nodes = x.shape[0]
        isolated_nodes = set(range(num_nodes)).difference(set(edges.flatten().tolist()))
        edges = torch.cat(
            [
                edges,
                torch.LongTensor([[i, i] for i in isolated_nodes]).to(edges.device),
            ],
            dim=0,
        ).to(torch.long)

        # feature evolution
        x = self.dropout(x)
        x_src = self.linear(x).index_select(0, edges[:, 0])
        x_dst = self.linear(x).index_select(0, edges[:, 1])
        x = self.act(x_src + x_dst)

        if self.if_pos:
            pos_src = pos.index_select(0, edges[:, 0])
            pos_dst = pos.index_select(0, edges[:, 1])
            vector = pos_dst - pos_src
            distance = torch.norm(vector, p=2, dim=1).unsqueeze(-1)
            torch.clamp_(distance, min=0.1)
            distance_matrix = self.rbf_expand(distance)
            dist_emd = self.linear_rbf(distance_matrix)
            x = x * dist_emd
            pos = (pos_src + pos_dst) / 2
        atom_repr = x * self.attn

        # test
        atom_repr = nn.ELU()(atom_repr)

        # update batch and edges
        batch = batch.index_select(0, edges[:, 0])
        edges = self.edge_evolve(edges.to(x.device))

        # final readout
        mol_repr = self.readout(atom_repr, batch)

        return (atom_repr, edges, pos, batch, mol_repr)

    def edge_evolve(self, edges: torch.Tensor) -> torch.Tensor:
        lin_edge = edges[:, 0].tolist() + edges[:, 1].tolist()
        tally = defaultdict(list)
        for i, item in enumerate(lin_edge):
            tally[item].append(i if i < len(lin_edge) // 2 else i - len(lin_edge) // 2)

        output = []
        for _, locs in tally.items():
            if len(locs) > 1:
                output.append(list(combinations(locs, 2)))

        return torch.tensor(list(chain(*output)), device=edges.device)


class GATv2Layer(nn.Module):
    """
    GATv2 layer (Graph Attention Network)
    """

    def __init__(
        self,
        num_node_features: int,
        output_dim: int,
        num_heads: int,
        activation: nn.Module = nn.PReLU(),
        concat: bool = True,
        residual: bool = True,
        bias: bool = True,
        dropout: float = 0.1,
        share_weights: bool = False,
    ) -> None:
        super(GATv2Layer, self).__init__()

        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.residual = residual
        self.activation = activation
        self.concat = concat
        self.dropout = dropout
        self.share_weights = share_weights

        # Embedding by linear projection
        self.linear_src = nn.Linear(
            num_node_features, output_dim * num_heads, bias=False
        )
        if self.share_weights:
            self.linear_dst = self.linear_src
        else:
            self.linear_dst = nn.Linear(
                num_node_features, output_dim * num_heads, bias=False
            )

        # The learnable parameters to compute attention coefficients
        self.double_attn = nn.Parameter(torch.Tensor(1, num_heads, output_dim))

        # Bias and concat
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(output_dim * num_heads))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter("bias", None)

        if residual:
            if num_node_features == num_heads * output_dim:
                self.residual_linear = nn.Identity()
            else:
                self.residual_linear = nn.Linear(
                    num_node_features, num_heads * output_dim, bias=False
                )
        else:
            self.register_parameter("residual_linear", None)

        # Some fixed function
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # Readout
        self.readout = ReadoutPhase(output_dim * num_heads)

        self.init_params()

    def init_params(self) -> None:
        nn.init.xavier_uniform_(self.linear_src.weight)
        nn.init.xavier_uniform_(self.linear_dst.weight)

        nn.init.xavier_uniform_(self.double_attn)
        if self.residual:
            if self.num_node_features != self.num_heads * self.output_dim:
                nn.init.xavier_uniform_(self.residual_linear.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x, edge_index, batch):
        # Input preprocessing
        edge_src_index, edge_dst_index = edge_index

        # Projection on the new space
        src_projected = self.linear_src(self.dropout(x)).view(
            -1, self.num_heads, self.output_dim
        )
        dst_projected = self.linear_dst(self.dropout(x)).view(
            -1, self.num_heads, self.output_dim
        )

        #######################################
        # Edge Attn
        #######################################

        # Edge attention coefficients
        edge_attn = self.leakyReLU(
            (
                src_projected.index_select(0, edge_src_index)
                + dst_projected.index_select(0, edge_dst_index)
            )
        )
        edge_attn = (self.double_attn * edge_attn).sum(-1)
        exp_edge_attn = (edge_attn - edge_attn.max()).exp()

        # sum the edge scores to destination node
        num_nodes = x.shape[0]
        edge_node_score_sum = torch.zeros(
            [num_nodes, self.num_heads],
            dtype=exp_edge_attn.dtype,
            device=exp_edge_attn.device,
        )
        edge_dst_index_broadcast = edge_dst_index.unsqueeze(-1).expand_as(exp_edge_attn)
        edge_node_score_sum.scatter_add_(0, edge_dst_index_broadcast, exp_edge_attn)

        # normalized edge attention
        # edge_attn shape = [num_edges, num_heads, 1]
        exp_edge_attn = exp_edge_attn / (
            edge_node_score_sum.index_select(0, edge_dst_index) + 1e-16
        )
        exp_edge_attn = self.dropout(exp_edge_attn).unsqueeze(-1)

        # summation from one-hop atom
        edge_x_projected = src_projected.index_select(0, edge_src_index) * exp_edge_attn
        edge_output = torch.zeros(
            [num_nodes, self.num_heads, self.output_dim],
            dtype=exp_edge_attn.dtype,
            device=exp_edge_attn.device,
        )
        edge_dst_index_broadcast = (
            (edge_dst_index.unsqueeze(-1)).unsqueeze(-1).expand_as(edge_x_projected)
        )
        edge_output.scatter_add_(0, edge_dst_index_broadcast, edge_x_projected)

        output = edge_output
        # residual, concat, bias, activation
        if self.residual:
            output += self.residual_linear(x).view(num_nodes, -1, self.output_dim)
        if self.concat:
            output = output.view(-1, self.num_heads * self.output_dim)
        else:
            output = output.mean(dim=1)

        if self.bias is not None:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output, self.readout(output, batch)


class RBFExpansion(nn.Module):
    def __init__(
        self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50
    ) -> None:
        super(RBFExpansion, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class PositionEncoder(nn.Module):
    def __init__(self, d_model, seq_len: int = 4, device: str = "cuda:0") -> None:
        super().__init__()
        # position_enc.shape = [seq_len, d_model]
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)]
                for pos in range(seq_len)
            ]
        )
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        self.position_enc = (
            torch.tensor(position_enc, device=device).unsqueeze(0).float()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [batch_size, seq_length, d_model]
        x = x * Variable(self.position_enc, requires_grad=False)
        return x


class ReadoutPhase(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        # readout phase
        self.weighting = nn.Linear(dim, 1)
        self.score = nn.Sigmoid()

        nn.init.xavier_uniform_(self.weighting.weight)
        nn.init.constant_(self.weighting.bias, 0)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        weighted = self.weighting(x)
        score = self.score(weighted)
        output1 = global_add_pool(score * x, batch)
        output2 = global_max_pool(x, batch)

        output = torch.cat([output1, output2], dim=1)
        return output
