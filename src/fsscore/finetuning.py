"""
Fine-tuning the model
"""
import argparse
import os
import shutil
from datetime import date
from typing import List, Optional, Tuple, Union

import finetuning_scheduler as fts
import pandas as pd

# https://github.com/speediedan/finetuning-scheduler/blob/main/README.md#installation-using-the-standalone-pytorch-lightning-package
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from fsscore.data.datamodule import CustomDataModule
from fsscore.data.featurizer import AVAILABLE_FEATURIZERS
from fsscore.models.ranknet import LitRankNet
from fsscore.utils.earlystop_ft import EarlyStoppingFT
from fsscore.utils.logging_utils import get_logger
from fsscore.utils.paths import (
    DATA_PATH,
    INPUT_TEST_PATH,
    MODEL_PATH,
    PRETRAIN_MODEL_PATH,
    PROCESSED_PATH,
)

LOGGER = get_logger(__name__)


class TrackImprovementToPretain(pl.Callback):
    def __init__(
        self,
        pretrained_model: pl.LightningModule,
        depth_edges: int,
        graph_datapath: str,
        logger: WandbLogger,
        smiles: List[Tuple[str, str]],
        target: List[float],
    ) -> None:
        self.pretrained_model = pretrained_model
        self.depth_edges = depth_edges
        self.graph_datapath = graph_datapath
        self.logger = logger
        self.smiles = smiles
        self.target = target

        self._petrained_metrics()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        new_acc = trainer.callback_metrics["val/acc"]
        if "val/auroc" in trainer.callback_metrics:
            # if small sample and only one class represented, auroc is not calculated
            new_auroc = trainer.callback_metrics["val/auroc"]
        else:
            new_auroc = None

        self.logger.log_metrics({"val/acc_improvement": new_acc - self.acc})
        if new_auroc is not None and self.auroc is not None:
            self.logger.log_metrics({"val/auc_improvement": new_auroc - self.auroc})

    def _petrained_metrics(self) -> None:
        graph_datapath = os.path.join(DATA_PATH, "ft_improv_val.pt")
        dm = CustomDataModule(
            smiles=self.smiles,
            target=self.target,
            use_fp=self.pretrained_model._hparams.fp,
            featurizer=args.featurizer,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            graph_datapath=None
            if self.pretrained_model._hparams.fp
            else graph_datapath,
            use_geom=self.pretrained_model._hparams.use_geom,
            depth_edges=self.depth_edges,
        )
        old_trainer = pl.Trainer(
            precision="16-mixed",
            accelerator="auto",
            strategy="ddp",
            devices=torch.cuda.device_count(),
            max_epochs=-1,
            logger=self.logger,
        )

        old_trainer.test(self.pretrained_model, datamodule=dm)

        self.acc = old_trainer.callback_metrics["test/acc"]
        if "test/auroc" in old_trainer.callback_metrics:
            # if small sample and only one class represented, auroc is not calculated
            self.auroc = old_trainer.callback_metrics["test/auroc"]
        else:
            self.auroc = None

        # delete the graphdatapath if it was created
        # FIXME: want to remove the saving part anyways because clogs up the disk
        if not self.pretrained_model._hparams.fp:
            os.remove(graph_datapath)


def finetune(
    smiles: List[Tuple[str, str]],
    target: List[float],
    graph_datapath: Optional[str] = None,
    filename: Optional[str] = None,
    save_dir: Optional[str] = None,
    featurizer: str = "graph_2D",
    model_path: str = None,
    batch_size: int = 32,
    n_epochs: int = 10,
    num_workers: Optional[int] = None,
    val_indices: Optional[List[int]] = None,
    val_size: Union[float, int] = 0.2,
    log_every: int = 10,
    lr: float = 3e-4,
    track_improvement: bool = False,
    smiles_val_add: List[Tuple[str, str]] = None,
    target_val_add: List[float] = None,
    ft_schedule_yaml: str = None,
    earlystopping: bool = True,
    patience: int = 3,
    datapoints: int = None,
) -> None:
    """
    Fine-tunes the model
    """
    os.makedirs(save_dir, exist_ok=True)

    # wandb logger
    logger = WandbLogger(
        name="finetuning",
        project="fsscore",
        save_dir=save_dir,
        tags=[str(datapoints)],
    )

    if graph_datapath is None:
        graph_datapath = os.path.join(
            PROCESSED_PATH, f"temp_{logger.version}", f"{filename}_ft.pt"
        )

    # load model
    model = LitRankNet.load_from_checkpoint(
        model_path,
        lr=lr,
    )

    depth_edges = model._hparams.arrange.count("L") - 1
    if model._hparams.arrange[-1] != "L":
        depth_edges += 1

    if val_indices is None and val_size > 0:
        if val_size > 1:
            val_size = int(val_size)
        (
            smiles_train,
            smiles_val,
            target_train,
            target_val,
            indices_train,
            val_indices,
        ) = train_test_split(
            smiles,
            target,
            list(range(len(smiles))),
            test_size=val_size,
            shuffle=True,
        )
        monitor = "val/loss"
        val_batches = 1
    elif val_indices is not None:
        val_size = len(val_indices) / len(smiles)
        smiles_val = [smiles[i] for i in val_indices]
        target_val = [target[i] for i in val_indices]
        monitor = "val/loss"
        val_batches = 1
    else:
        monitor = "train/loss"
        if earlystopping:
            val_batches = 1
        else:
            val_batches = 0

    # load data
    dm = CustomDataModule(
        smiles=smiles,
        target=target,
        batch_size=batch_size,
        featurizer=featurizer,
        graph_datapath=graph_datapath,
        num_workers=num_workers,
        use_fp=model._hparams.fp,
        use_geom=model._hparams.use_geom,
        random_split=True if val_indices is None else False,
        depth_edges=depth_edges,
        val_indices=val_indices if val_indices is None else val_indices,
        val_size=val_size,
        smiles_val_add=smiles_val_add,
        target_val_add=target_val_add,
    )

    # callback to track improvement to pre-trained model
    if track_improvement:
        track_improvement = TrackImprovementToPretain(
            model,
            depth_edges,
            graph_datapath,
            logger,
            smiles_val,
            target_val,
        )

    # checkpoint callback
    ckpt_kwargs = {
        "monitor": monitor,
        "mode": "min",
        "save_last": True,
        "dirpath": os.path.join(save_dir, "checkpoints", f"run_{logger.version}"),
        "filename": "ft_{epoch:02d}-best" + f"_{monitor.split('/')[0]}_loss",
    }
    ckpt = ModelCheckpoint(**ckpt_kwargs)

    earlystop = EarlyStoppingFT(
        monitor_ft=monitor,
        # both metrics below are based on pretraining test data
        monitor_pt="val2/acc" if val_size > 0 else "val/acc",
        mode_ft="min",
        mode_pt="max",
        patience=patience,  # only for finetuning loss
        check_on_train_epoch_end=False,
        min_delta=0.02,  # only applies to pre-training metric
    )

    # TODO max_depth (how many fts phases to do) as args
    # FIXME no error msg but FTS not working yet
    earlystopping_kwargs = {"monitor": monitor, "patience": patience}
    fts_sched = fts.FinetuningScheduler(ft_schedule=ft_schedule_yaml, max_depth=-1)
    fts_early_stopping = fts.FTSEarlyStopping(**earlystopping_kwargs)
    fts_ckpt = fts.FTSCheckpoint(**ckpt_kwargs)

    callbacks = (
        ([ckpt] if not ft_schedule_yaml else [])
        + ([track_improvement] if track_improvement else [])
        + ([fts_sched, fts_early_stopping, fts_ckpt] if ft_schedule_yaml else [])
        + ([earlystop] if earlystopping else [])
    )

    # fine-tuning scheduler
    trainer = pl.Trainer(
        precision="16-mixed",
        accelerator="auto",
        strategy="ddp",
        devices=torch.cuda.device_count(),
        log_every_n_steps=log_every,
        max_epochs=n_epochs,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        limit_val_batches=val_batches,
    )

    trainer.fit(
        model,
        dm,
    )

    # delete all folder with temp_{run_id} in PROCESSED_PATH
    for folder in os.listdir(PROCESSED_PATH):
        if folder.startswith(f"temp_{logger.version}"):
            shutil.rmtree(os.path.join(PROCESSED_PATH, folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the csv file.",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--graph_datapath",
        type=str,
        help="Path to the pt file with featurized graphs and SMILES as ID.",
        default=None,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the pre-trained model checkpoint",
        default=PRETRAIN_MODEL_PATH,
    )
    parser.add_argument(
        "--compound_cols",
        type=str,
        default=["smiles_i", "smiles_j"],
        nargs="*",
        help="Column names with SMILES for each comparison.",
    )
    parser.add_argument(
        "--rating_col",
        type=str,
        default="target",
        help="Column name with the target rating label",
    )
    parser.add_argument(
        "--featurizer",
        choices=list(AVAILABLE_FEATURIZERS.keys()),
        help="Name of the featurizer to use",
        default="graph_2D",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save the model and results",
        default=os.path.join(MODEL_PATH, f"finetuned_{date.today()}"),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=32,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers for data loading",
        default=None,
    )
    parser.add_argument(
        "--mc_dropout_samples",
        type=int,
        help="Number of MC dropout samples",
        default=1,
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        help="Dropout probability",
        default=0.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
        default=42,
    )
    parser.add_argument(
        "--log_every",
        type=int,
        help="Log every n steps",
        default=1,
    )
    parser.add_argument(
        "--val_size",
        type=float,
        help="Validation fraction (float) or size (int) for random split",
        default=0.2,
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        help="Number of epochs.",
        default=20,
    )
    parser.add_argument(
        "--datapoints",
        type=int,
        help="Number of datapoints from the top of the df to use for fine-tuning.",
        default=None,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
        default=1e-4,  # TODO choose smaller lr for ft?
    )
    parser.add_argument(
        "--track_improvement",
        action="store_true",
        help="Whether to track improvement to pre-trained model",
        default=False,
    )
    parser.add_argument(
        "--track_pretest",
        action="store_true",
        help="Track performance on pre-training test set.",
        default=False,
    )
    parser.add_argument(
        "--ft_schedule_yaml",
        type=str,
        help="Path to the yaml file with the fine-tuning schedule.",
        # TODO add default in paths.py
        default=None,
    )
    parser.add_argument(
        "--earlystopping",
        action="store_true",
        help="Whether to use early stopping.",
        default=False,
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Patience for early stopping.",
        default=3,
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    # load data
    df = pd.read_csv(args.data_path)
    if args.datapoints is not None:
        df = df.iloc[: args.datapoints]
    else:
        args.datapoints = len(df)
    smiles_pairs = df[args.compound_cols].values.tolist()
    target = df[args.rating_col].values.tolist()

    if args.earlystopping:
        LOGGER.info("Using early stopping.")
        args.track_pretest = True
        args.ft_schedule_yaml = None
    if args.track_improvement and args.val_size > 0:
        LOGGER.info("Tracking improvement to pre-trained model.")
    elif args.track_improvement and args.val_size == 0:
        LOGGER.warning(
            "Cannot track improvement to pre-trained model with 0 validation size."
        )
        args.track_improvement = False
    if args.track_pretest:
        LOGGER.info("Tracking performance on pre-training test set.")
        df_test = pd.read_csv(INPUT_TEST_PATH)
        df_test = df_test.sample(n=5000, random_state=args.seed)
        smiles_test = df_test[["smiles_i", "smiles_j"]].values.tolist()
        target_test = df_test["target"].values.tolist()

    finetune(
        smiles_pairs,
        target,
        graph_datapath=args.graph_datapath,
        filename=os.path.basename(args.data_path).split(".")[0],
        save_dir=args.save_dir,
        featurizer=args.featurizer,
        model_path=args.model_path,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        num_workers=args.num_workers,
        log_every=args.log_every,
        lr=args.lr,
        track_improvement=args.track_improvement,
        smiles_val_add=smiles_test if args.track_pretest else None,
        target_val_add=target_test if args.track_pretest else None,
        ft_schedule_yaml=args.ft_schedule_yaml,
        earlystopping=args.earlystopping,
        patience=args.patience,
        val_size=args.val_size,
        datapoints=args.datapoints,
    )
