import argparse
import multiprocessing
import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from rdkit import Chem

from intuitive_sc.data.datamodule import CustomDataModule
from intuitive_sc.data.featurizer import AVAILABLE_FEATURIZERS, Featurizer
from intuitive_sc.models.ranknet import LitRankNet
from intuitive_sc.utils.logging import get_logger
from intuitive_sc.utils.paths import DATA_PATH, RESULTS_PATH  # TODO default modelpath

LOGGER = get_logger(__name__)


class Scorer:
    def __init__(
        self,
        model: LitRankNet = None,
        featurizer: Featurizer = None,
        num_workers: Optional[int] = None,
        mc_dropout_samples: int = 1,
        verbose: bool = False,
        batch_size: int = 32,
        graph_datapath: str = None,
    ) -> None:
        self.model = model
        self.featurizer = featurizer
        if num_workers is None:
            num_workers = multiprocessing.cpu_count() // 2
        self.num_workers = num_workers
        self.model.mc_dropout_samples = mc_dropout_samples
        self.batch_size = batch_size
        self.graph_datapath = None if self.model._hparams.fp else graph_datapath
        self.depth_edges = self.model._hparams.arrange.count("L") - 1
        if self.model._hparams.arrange[-1] != "L":
            self.depth_edges += 1

        self.trainer = Trainer(
            precision="16-mixed",
            accelerator="auto",
            devices=1,
            max_epochs=-1,
            logger=verbose,
            enable_progress_bar=verbose,
        )

    def score(
        self,
        smiles: List[str],
        read_fn: Callable = Chem.MolFromSmiles,
    ) -> np.ndarray[float]:
        """
        Scores a a list of SMILES string.

        Args:
            smiles: list of SMILES string
            read_fn: function to read SMILES string into a Mol object

        Returns:
            np.ndarray of scores
        """
        dm = CustomDataModule(
            smiles=smiles,
            featurizer=self.featurizer,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            read_fn=read_fn,
            use_fp=self.model._hparams.fp,
            use_geom=self.model._hparams.use_geom,
            graph_datapath=self.graph_datapath,
            depth_edges=self.depth_edges,
        )

        preds = self.trainer.predict(self.model, dm)

        if self.model.mc_dropout_samples > 1:
            scores_mean, scores_var = [pred[0] for pred in preds], [
                pred[1] for pred in preds
            ]
            return torch.cat(scores_mean).numpy(), torch.cat(scores_var).numpy()
        else:
            return torch.cat(preds).numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score molecules with a trained model",
        prog="intuitive_sc score",
    )
    parser.add_argument(
        # TODO add a default path (import from paths)
        "--model_path",
        type=str,
        help="Path to the model checkpoint",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the csv file with SMILES to score",
        required=True,
    )
    parser.add_argument(
        "--compound_col",
        type=str,
        default="smiles",
        help="Column name with SMILES to score",
    )
    parser.add_argument(
        "--save_filepath",
        type=str,
        help="Filename to save the results to (.csv)",
        default=None,
    )
    parser.add_argument(
        "--featurizer",
        choices=list(AVAILABLE_FEATURIZERS.keys()),
        help="Name of the featurizer to use",
        default="graph_2D",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to print verbose output",
        default=False,
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

    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    # load data
    df = pd.read_csv(args.data_path)
    LOGGER.info(f"Loaded {len(df)} SMILES from {args.data_path}")
    smiles = df[args.compound_col].tolist()

    # load model
    model = LitRankNet.load_from_checkpoint(
        args.model_path,
    )

    input_base = os.path.basename(args.data_path).split(".")[0]
    graph_datapath = os.path.join(DATA_PATH, f"{input_base}_graphs_score.pt")

    # score
    scorer = Scorer(
        model=model,
        featurizer=args.featurizer,
        verbose=args.verbose,
        batch_size=args.batch_size,
        graph_datapath=graph_datapath,
        mc_dropout_samples=args.mc_dropout_samples,
    )
    LOGGER.info(f"Scoring {len(smiles)} SMILES")
    if args.mc_dropout_samples > 1:
        scores_mean, scores_var = scorer.score(smiles=smiles)
        df["score_mean"] = scores_mean
        df["score_var"] = scores_var
    else:
        scores = scorer.score(smiles=smiles)
        df["score"] = scores

    # save
    if args.save_filepath is None:
        model_base = os.path.basename(os.path.dirname(os.path.dirname(args.model_path)))
        args.save_filepath = os.path.join(
            RESULTS_PATH, f"{input_base}_{model_base}_scores.csv"
        )

    os.makedirs(os.path.dirname(args.save_filepath), exist_ok=True)
    LOGGER.info(f"Saving scores to {args.save_filepath}")
    df.to_csv(args.save_filepath, index=False)
