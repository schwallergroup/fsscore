import argparse
import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from rdkit import Chem

from intuitive_sc.data.datamodule import CustomDataModule
from intuitive_sc.data.featurizer import (
    AVAILABLE_FEATURIZERS,
    Featurizer,
    get_featurizer,
)
from intuitive_sc.models.gnn import AVAILABLE_GRAPH_ENCODERS
from intuitive_sc.models.ranknet import LitRankNet
from intuitive_sc.utils.logging import get_logger

LOGGER = get_logger(__name__)


class Scorer:
    def __init__(
        self,
        model: LitRankNet = None,
        featurizer: Featurizer = None,
        num_workers: Optional[int] = None,
        mc_dropout_samples: int = 1,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.featurizer = featurizer
        self.num_workers = num_workers
        self.model.mc_dropout_samples = mc_dropout_samples

        self.trainer = Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=-1,
            logger=verbose,
            enable_progress_bar=verbose,
        )

    def score(
        self,
        smiles: List[str],
        batch_size: int = 32,
        read_fn: Callable = Chem.MolFromSmiles,
    ) -> np.ndarray[float]:
        """
        Scores a a list of SMILES string.

        Args:
            smiles: list of SMILES string

        Returns:
            np.ndarray of scores
        """
        dm = CustomDataModule(
            smiles=smiles,
            featurizer=self.featurizer,
            batch_size=batch_size,
            num_workers=self.num_workers,
            read_fn=read_fn,
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
        # TODO add a default path
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
        "--graph_encoder",
        choices=list(AVAILABLE_GRAPH_ENCODERS.keys()),
        help="Name of the graph encoder to use",
        default="GCN",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to print verbose output",
        default=False,
    )
    parser.add_argument(
        "--use_fp",
        action="store_true",
        help="Whether to use fingerprints as input",
        default=False,
    )
    args = parser.parse_args()

    # load data
    df = pd.read_csv(args.data_path)
    LOGGER.info(f"Loaded {len(df)} SMILES from {args.data_path}")
    smiles = df[args.compound_col].tolist()

    featurizer = get_featurizer(args.featurizer)

    # load model
    model = LitRankNet.load_from_checkpoint(
        args.model_path,
        input_size=featurizer.dim(),
        fp=args.use_fp,
    )

    # score
    scorer = Scorer(model=model, featurizer=featurizer, verbose=args.verbose)
    LOGGER.info(f"Scoring {len(smiles)} SMILES")
    scores = scorer.score(smiles=smiles)

    # save
    if args.save_filepath is None:
        args.save_filepath = os.path.dirname(args.model_path)
    os.makedirs(args.save_filepath, exist_ok=True)
    LOGGER.info(f"Saving scores to {args.save_filepath}")
    df["score"] = scores
    df.to_csv(os.path.join(args.save_filepath, "scores.csv"), index=False)
