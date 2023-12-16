"""
Code adapted from molskill
github repo: https://github.com/microsoft/molskill
"""
import os
from typing import Callable, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger

from fsscore.models.ranknet import LitRankNet, RankNet
from fsscore.utils.earlyreloading import DataLoaderReloader


def get_new_model_and_trainer(
    save_dir: Optional[str] = None,
    input_size: int = 2048,  # TODO hard coded; fix for graphs
    lr: float = 3e-4,
    reg_factor: float = 1e-4,
    n_epochs: int = 100,
    log_every: int = 10,
    loss_fn: Callable = F.binary_cross_entropy_with_logits,
    mc_dropout_samples: int = 1,
    dropout_p: float = 0.0,
    encoder: Optional[str] = None,
    use_fp: bool = False,
    use_geom: bool = False,
    arrange: str = "GGLGGL",
    reload_interval: int = 0,
    early_reloading: bool = False,
    use_val: float = 1.0,
) -> Tuple[LitRankNet, pl.Trainer]:
    """
    Creates a new RankNet model and trainer.

    Args:
        save_dir: Directory to save the model and results
        lr: Learning rate
        reg_factor: Regularization factor - small value is sufficient
        n_epochs: Number of epochs
        log_every: Log every n epochs
        loss_fn: Loss function
        mc_dropout_samples: Number of MC dropout samples
        dropout_p: Dropout probability
        sigmoid: Whether to use sigmoid
        encoder: Encoder to use
        use_fp: Whether to use fingerprints
        use_geom: Whether to use geometry
        arrange: Arrangement of layers

    Returns:
        Tuple of model and trainer
    """
    os.makedirs(save_dir, exist_ok=True)

    # wandb logger
    logger = WandbLogger(
        name="train",
        project="fsscore",
        save_dir=save_dir,
    )

    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(save_dir, "checkpoints"),
        filename="ranknet-{epoch:02d}-best_val_loss",
        monitor="val/loss",
        save_last=True,
        mode="min",
    )

    data_loader_reloader = DataLoaderReloader(
        monitor="val/loss",
        mode="min",
        patience=3,
        reload_every_n_epochs=reload_interval,
        check_on_train_epoch_end=False,
    )

    net = RankNet(
        input_size=input_size,
        hidden_size=256,  # TODO hard coded
        n_layers=3,  # TODO hard coded
        dropout_p=dropout_p,
        encoder=encoder,
        fp=use_fp,
        use_geom=use_geom,
        arrange=arrange,
    )
    model = LitRankNet(
        net=net,
        lr=lr,
        regularization_factor=reg_factor,
        loss_fn=loss_fn,
        input_size=input_size,
        mc_dropout_samples=mc_dropout_samples,
        dropout_p=dropout_p,
        encoder=encoder,
        fp=use_fp,
        use_geom=use_geom,
        arrange=arrange,
    )
    trainer = pl.Trainer(
        precision="16-mixed",
        logger=logger,
        accelerator="auto",
        devices=torch.cuda.device_count(),
        strategy="ddp",
        max_epochs=n_epochs,
        log_every_n_steps=log_every,
        callbacks=[ckpt] + ([data_loader_reloader] if early_reloading else []),
        deterministic=False,  # some components cannot be deterministic
        profiler="simple",
        reload_dataloaders_every_n_epochs=reload_interval if not early_reloading else 0,
        num_sanity_val_steps=0,
        limit_val_batches=use_val,
    )

    return model, trainer
