"""
Code for GNNs adapted from LineEvo: https://github.com/fate1997/LineEvo/tree/main
"""
from typing import Dict, Type

import torch.nn as nn
from torch_geometric.utils import to_undirected

from intuitive_sc.data.graph_dataset import GraphData
from intuitive_sc.models.gnn_readouts import GATv2Layer, LineEvoLayer

AVAILABLE_GRAPH_ENCODERS: Dict[str, Type[nn.Module]] = {}


def register_encoder(name: str):
    def register_function(cls: Type[nn.Module]):
        AVAILABLE_GRAPH_ENCODERS[name] = cls
        return cls

    return register_function


# TODO see if can also include GIN since layers chosen separately anyway
# TODO make naming more intuitive
@register_encoder("GNN")
class GNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        use_geom: bool = False,
        arrange: str = "GGLGGL",
    ):
        super(GNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.line_dropout = 0.1
        self.if_pos = use_geom
        self.arrange = arrange

        self.layers = nn.ModuleList()
        for i, ind in enumerate(list(self.arrange)):
            if ind == "G":
                layer = GATv2Layer(
                    num_node_features=self.input_dim if i == 0 else self.hidden_dim,
                    output_dim=self.hidden_dim // self.num_heads,
                    num_heads=self.num_heads,
                    concat=True,
                    activation=nn.PReLU(),
                    residual=True,
                    bias=True,
                    dropout=self.dropout,
                )
                self.layers.append(layer)

            elif ind == "L":
                layer = LineEvoLayer(
                    in_dim=self.input_dim if i == 0 else self.hidden_dim,
                    dim=self.hidden_dim,
                    dropout=self.line_dropout,
                    if_pos=self.if_pos,
                )
                self.layers.append(layer)

            else:
                raise ValueError("Indicator should be G or L.")

    def forward(self, graph: GraphData):
        x, edge_index, edge_attr, pos, batch = (  # noqa: F841
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.pos,
            graph.batch,
        )

        mol_repr_all = 0
        line_index = 0
        for i, ind in enumerate(list(self.arrange)):
            if ind == "G":
                x, mol_repr = self.layers[i](x, edge_index, batch)
            elif ind == "L":
                edges = getattr(graph, f"edges_{line_index}")
                x, pos, batch, mol_repr = self.layers[i](x, edges, pos, batch)
                line_index += 1
                edge_index = to_undirected(edges.T)
            mol_repr_all += mol_repr

        return mol_repr_all
