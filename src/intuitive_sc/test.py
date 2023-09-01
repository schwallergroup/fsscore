import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from intuitive_sc.data.datamodule import CustomDataModule
from intuitive_sc.models.ranknet import LitRankNet
from intuitive_sc.utils.logging import get_logger
from intuitive_sc.utils.paths import DATA_PATH, INPUT_TEST_PATH, MODEL_PATH

LOGGER = get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the csv file. Must contain 2 SMILES columns\
            and 'target'",
        default=INPUT_TEST_PATH,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(MODEL_PATH, "last.ckpt"),
        help="Path to the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers to use for the dataloader",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to print training logs",
    )
    parser.add_argument(
        "--graph_datapath",
        type=str,
        default=None,
        help="Path to the graph dataset",
    )
    parser.add_argument(
        "--featurizer",
        type=str,
        default="graph_2D",
        help="Featurizer to use",
    )
    parser.add_argument(
        "--compound_cols",
        type=List[str],
        default=["smiles_i", "smiles_j"],
        help="Column names with SMILES for each comparison.",
    )
    parser.add_argument(
        "--rating_col",
        type=str,
        default="target",
        help="Column name with the target rating label",
    )
    args = parser.parse_args()

    wandb_logger = WandbLogger(
        name="ranknet",
        project="intuitive-sc",
        save_dir=os.path.dirname(os.path.dirname(args.model_path)),
    )

    df = pd.read_csv(args.data_path)
    smiles_pairs = df[args.compound_cols].values.tolist()
    target = df[args.rating_col].values.tolist()

    LOGGER.info("Loading model")
    model = LitRankNet.load_from_checkpoint(args.model_path)

    depth_edges = model._hparams.arrange.count("L") - 1
    if model._hparams.arrange[-1] != "L":
        depth_edges += 1

    if args.graph_datapath is None and not model._hparams.fp:
        filename = os.path.basename(args.data_path).split(".")[0]
        graph_datapath = os.path.join(
            DATA_PATH, "test", f"{filename}_graphs_depth{depth_edges}.pt"
        )

    dm = CustomDataModule(
        smiles=smiles_pairs,
        target=target,
        use_fp=model._hparams.fp,
        featurizer=args.featurizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        graph_datapath=None if model._hparams.fp else graph_datapath,
        use_geom=model._hparams.use_geom,
        depth_edges=depth_edges,
    )

    LOGGER.info("Testing model")
    trainer = pl.Trainer(
        precision="16-mixed",
        accelerator="auto",
        devices=1,
        max_epochs=-1,
        logger=wandb_logger,
        enable_progress_bar=args.verbose,
    )

    trainer.test(model, datamodule=dm)

    preds_i = np.array(model.test_scores_i)
    preds_j = np.array(model.test_scores_j)

    LOGGER.info("Creating dataframe")
    df_scored = pd.DataFrame(
        {
            args.compound_cols[0]: [x[0] for x in smiles_pairs],
            args.compound_cols[1]: [x[1] for x in smiles_pairs],
            "target": target,
            f"{args.compound_cols[0]}_pred": preds_i,
            f"{args.compound_cols[1]}_pred": preds_j,
            "pred_diff": preds_i - preds_j,
        }
    )

    df_scored.to_csv(
        os.path.join(
            os.path.dirname(os.path.dirname(args.model_path)),
            f"filename_{os.path.basename(args.model_path).split('.')[0]}_scored.csv",
        ),
        index=False,
    )

    LOGGER.info("Done")
