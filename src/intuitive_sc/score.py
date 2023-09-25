import argparse
import multiprocessing
import os
import shutil
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from rdkit import Chem

from intuitive_sc.data.datamodule import CustomDataModule
from intuitive_sc.data.featurizer import AVAILABLE_FEATURIZERS, Featurizer
from intuitive_sc.models.ranknet import LitRankNet
from intuitive_sc.utils.logging_utils import get_logger
from intuitive_sc.utils.paths import PROCESSED_PATH, RESULTS_PATH

# TODO default modelpath

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
        dropout_p: float = 0.0,
        keep_graphs: bool = False,
    ) -> None:
        self.model = model
        self.featurizer = featurizer
        if num_workers is None:
            num_workers = multiprocessing.cpu_count() // 2
        self.num_workers = num_workers
        self.model.dropout_p = dropout_p
        self.model.mc_dropout_samples = mc_dropout_samples
        self.batch_size = batch_size
        self.keep_graphs = keep_graphs
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
    ) -> Union[np.ndarray[float], Tuple[np.ndarray[float], np.ndarray[float]]]:
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

        if not self.keep_graphs and not self.model._hparams.fp:
            os.remove(self.graph_datapath)
            # remove the processed folder even if not empty
            processed_dir = os.path.join(
                os.path.dirname(self.graph_datapath), "processed"
            )
            shutil.rmtree(processed_dir)

        if self.model.mc_dropout_samples > 1:
            scores_mean, scores_var = [pred[0] for pred in preds], [
                pred[1] for pred in preds
            ]
            return torch.cat(scores_mean).numpy(), torch.cat(scores_var).numpy()
        else:
            return torch.cat(preds).numpy()


def reverse_sigmoid(x, low, high, k=1) -> float:
    try:
        return 1 / (1 + 10 ** (k * (x - (high + low) / 2) * 10 / (high - low)))
    except OverflowError:
        return 0


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
        "--compound_cols",
        type=str,
        default=["smiles"],
        nargs="*",
        help="Column names with SMILES for each comparison.",
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
    parser.add_argument(
        "--dropout_p",
        type=float,
        help="Dropout probability",
        default=0.0,
    )
    parser.add_argument(
        "--graph_datapath",
        type=str,
        help="Path to the graph dataset",
        default=None,
    )
    parser.add_argument(
        "--keep_graphs",
        action="store_true",
        help="Whether to keep the graph dataset after scoring",
        default=False,
    )

    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    # load data
    df = pd.read_csv(args.data_path)
    LOGGER.info(f"Loaded {len(df)} SMILES from {args.data_path}")
    if len(args.compound_cols) == 1:
        output = f"{args.compound_cols[0]}_score"
        smiles = df[args.compound_cols[0]].tolist()
    else:
        output = f"{args.compound_cols[0]}_{args.compound_cols[1]}_diff"
        smiles = df[args.compound_cols].values.tolist()

    if args.mc_dropout_samples > 1 and args.dropout_p == 0.0:
        LOGGER.warning(
            "MC dropout is enabled but dropout probability is 0.0. "
            "This is not recommended."
            "Changed dropout probability to 0.2."
        )
        args.dropout_p = 0.2

    # load model
    model = LitRankNet.load_from_checkpoint(
        args.model_path,
    )
    input_base = os.path.basename(args.data_path).split(".")[0]
    if args.graph_datapath is None and not model._hparams.fp:
        args.graph_datapath = os.path.join(
            PROCESSED_PATH, f"{input_base}_graphs_score.pt"
        )

    # score
    scorer = Scorer(
        model=model,
        featurizer=args.featurizer,
        verbose=args.verbose,
        batch_size=args.batch_size,
        graph_datapath=args.graph_datapath,
        mc_dropout_samples=args.mc_dropout_samples,
        num_workers=args.num_workers,
        dropout_p=args.dropout_p,
        keep_graphs=args.keep_graphs,
    )
    LOGGER.info(f"Scoring {len(smiles)} SMILES")

    # split into subsets (hacky solution to the slowing down of dataloader)
    if len(smiles) > 100000 and not model._hparams.fp:
        LOGGER.info("Splitting into subsets")
        smiles_sub = np.array_split(smiles, len(smiles) // 100000)
    else:
        smiles_sub = [smiles]

    if args.mc_dropout_samples > 1:
        all_scores_mean, all_scores_var = [], []
        for sub in smiles_sub:
            scores_mean_sub, scores_var_sub = scorer.score(smiles=sub)
            all_scores_mean.append(scores_mean_sub)
            all_scores_var.append(scores_var_sub)
        scores_mean, scores_var = (
            np.concatenate(all_scores_mean),
            np.concatenate(all_scores_var),
        )
        df[f"{output}_mean"] = scores_mean
        df[f"{output}_var"] = scores_var
    else:
        all_scores = []
        for sub in smiles_sub:
            scores_sub = scorer.score(smiles=sub)
            all_scores.append(scores_sub)
        scores = np.concatenate(all_scores)
        df[f"{args.compound_cols[0]}_score"] = scores

        # # scale to [0,1] with sigmoid
        # if model.max_value is not None and model.min_value is not None:
        #     scores_scaled = reverse_sigmoid(scores, model.min_value, model.max_value)
        #     df[f"{args.compound_cols[0]}_score_scaled"] = scores_scaled

    # save
    if args.save_filepath is None:
        model_base = os.path.basename(os.path.dirname(os.path.dirname(args.model_path)))
        args.save_filepath = os.path.join(
            RESULTS_PATH, f"{input_base}_{model_base}_{output}.csv"
        )

    os.makedirs(os.path.dirname(args.save_filepath), exist_ok=True)
    LOGGER.info(f"Saving scores to {args.save_filepath}")
    df.to_csv(args.save_filepath, index=False)

    if args.mc_dropout_samples > 1:
        # sort by variance
        df = df.sort_values(by=[f"{output}_var"], ascending=False)
        # save sorted
        filepath_sorted = f'{args.save_filepath.split(".")[0]}_sorted.csv'
        LOGGER.info(f"Saving sorted scores to {filepath_sorted}")
        df.to_csv(filepath_sorted, index=False)
