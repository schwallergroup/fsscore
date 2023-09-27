import pytorch_lightning as pl
import torch
import torch.nn as nn

from intuitive_sc.models.ranknet import LitRankNet, RankNet


def test_ranknet():
    # Test input tensors
    x_i = torch.randn(32, 2048)
    x_j = torch.randn(32, 2048)

    # Test initialization
    ranknet = RankNet()
    assert isinstance(ranknet, nn.Module)

    # Test forward pass
    out_i, out_j, out_diff = ranknet(x_i, x_j)
    assert out_i.shape == out_j.shape == out_diff.shape == (32, 1)

    # Test score function
    score = ranknet.score(x_i)
    assert score.shape == (32, 1)


def test_litranknet():
    # Test input tensors
    x_i = torch.randn(32, 2048)
    x_j = torch.randn(32, 2048)
    y = torch.randint(0, 2, (32,))

    # Test initialization
    litranknet = LitRankNet()
    assert isinstance(litranknet, pl.LightningModule)

    # Test forward pass
    out = litranknet(x_i, x_j)
    assert isinstance(out, torch.Tensor)

    # Test training step
    optimizer = torch.optim.Adam(litranknet.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train_step_output = litranknet.training_step(
        (x_i, x_j), y, optimizer=optimizer, loss_fn=loss_fn
    )
    assert isinstance(train_step_output, dict)
    assert "loss" in train_step_output

    # Test validation step
    val_step_output = litranknet.validation_step((x_i, x_j), y)
    assert isinstance(val_step_output, dict)
    assert "val_loss" in val_step_output

    # Test test step
    test_step_output = litranknet.test_step((x_i, x_j), y)
    assert isinstance(test_step_output, dict)
    assert "test_loss" in test_step_output

    # Test configure_optimizers
    optimizer = litranknet.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
