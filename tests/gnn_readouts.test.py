import torch

from intuitive_sc.models.gnn_readouts import GATv2Layer, LineEvoLayer


# Test case 1
def test_gatv2_layer():
    layer = GATv2Layer(num_node_features=16, output_dim=32, num_heads=4)
    x = torch.randn(10, 16)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
    )
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    out = layer(x, edge_index, batch)
    assert out.shape == (10, 32)


# Test case 2
def test_gatv2_layer_with_dropout():
    layer = GATv2Layer(num_node_features=16, output_dim=32, num_heads=4, dropout=0.5)
    x = torch.randn(10, 16)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
    )
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    out = layer(x, edge_index, batch)
    assert out.shape == (10, 32)


# Test case 3
def test_gatv2_layer_with_residual():
    layer = GATv2Layer(num_node_features=16, output_dim=32, num_heads=4, residual=True)
    x = torch.randn(10, 16)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
    )
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    out = layer(x, edge_index, batch)
    assert out.shape == (10, 32)


def test_LineEvoLayer():
    layer = LineEvoLayer(in_dim=128, dim=128, dropout=0.0, if_pos=False)
    x = torch.randn(10, 128)
    edges = torch.randint(0, 10, (20, 2))
    pos = None
    batch = torch.randint(0, 5, (10,))
    atom_repr, pos, batch, mol_repr = layer(x, edges, pos, batch)
    assert atom_repr.shape == (20, 128)
    assert pos is None
    assert batch.shape == (20,)
    assert mol_repr.shape == (5, 128)


def test_GATv2Layer():
    layer = GATv2Layer(num_node_features=128, output_dim=128, num_heads=4)
    x = torch.randn(10, 128)
    edges = torch.randint(0, 10, (20, 2))
    pos = None
    batch = torch.randint(0, 5, (10,))
    atom_repr, pos, batch, mol_repr = layer(x, edges, pos, batch)
    assert atom_repr.shape == (20, 128)
    assert pos is None
    assert batch.shape == (20,)
    assert mol_repr.shape == (5, 128)
