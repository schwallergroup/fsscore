import pytorch_lightning as pl
import torch
import torch.nn as nn

from fsscore.models.ranknet import LitRankNet, RankNet


def test_fp_ranknet():
    # Test input tensors
    x_i = torch.randn(32, 2048)
    x_j = torch.randn(32, 2048)

    # Test initialization
    ranknet = RankNet(
        input_size=2048,
        dropout_p=0.0,
        fp=True,
        use_geom=False,
    )
    assert isinstance(ranknet, nn.Module)

    # Test forward pass
    out_i, out_j, out_diff = ranknet(x_i, x_j)
    assert out_i.shape == out_j.shape == out_diff.shape == (32, 1)

    # Test score function
    score = ranknet.score(x_i)
    assert score.shape == (32, 1)


def test_litranknet():
    # Test input tensors
    # Test input tensors
    x_i = torch.randn(32, 2048)
    x_j = torch.randn(32, 2048)
    y = torch.randint(0, 2, (32, 1)).float()

    # Test initialization
    litranknet = LitRankNet(
        input_size=2048,
        dropout_p=0.0,
        fp=True,
        use_geom=False,
    )
    assert isinstance(litranknet, pl.LightningModule)

    # Test forward pass
    out_i, out_j, out_diff = litranknet(x_i, x_j)
    assert isinstance(out_i, torch.Tensor)
    assert isinstance(out_j, torch.Tensor)
    assert isinstance(out_diff, torch.Tensor)

    # Test training step
    optimizer = torch.optim.Adam(litranknet.parameters(), lr=1e-3)
    train_step_output = litranknet.training_step(((x_i, x_j), y), 0)
    assert isinstance(train_step_output, torch.Tensor)

    # Test validation step
    val_step_output = litranknet.validation_step(((x_i, x_j), y), 0)
    assert isinstance(val_step_output, torch.Tensor)

    # Test test step
    test_step_output = litranknet.test_step(((x_i, x_j), y), 0)
    assert isinstance(test_step_output, dict)
    assert "acc" in test_step_output.keys()

    # Test configure_optimizers
    optimizer = litranknet.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
