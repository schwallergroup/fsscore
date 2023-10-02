import argparse
import os
from datetime import date
from glob import glob
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from rdkit import Chem

from fsscore.data.datamodule import CustomDataModule
from fsscore.data.featurizer import (
    AVAILABLE_FEATURIZERS,
    AVAILABLE_FP_FEATURIZERS,
    AVAILABLE_GRAPH_FEATURIZERS,
)
from fsscore.models.gnn import AVAILABLE_GRAPH_ENCODERS
from fsscore.models.nn_utils import get_new_model_and_trainer
from fsscore.models.ranknet import LitRankNet
from fsscore.utils.logging_utils import get_logger
from fsscore.utils.paths import INPUT_TRAIN_PATH, MODEL_PATH, PROCESSED_PATH

LOGGER = get_logger(__name__)

# https://github.com/facebookresearch/maskrcnn-benchmark/issues/103
torch.multiprocessing.set_sharing_strategy("file_system")


def train(
    smiles: List[Tuple[str, str]],
    target: List[float],
    graph_datapath: Optional[str] = None,
    save_dir: Optional[str] = None,
    featurizer: str = "graph_2D",
    graph_encoder: Optional[str] = None,
    use_fp: bool = False,
    lr: float = 3e-4,
    regularization_factor: float = 1e-4,
    n_epochs: int = 100,
    log_every: int = 10,
    val_size: float = 0.0,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    read_fn: Callable = Chem.MolFromSmiles,
    mc_dropout_samples: int = 1,
    dropout_p: float = 0.0,
    loss_fn: Callable = F.binary_cross_entropy_with_logits,
    resume_training: bool = False,
    arrange_layers: str = "GGLGGL",
    val_indices: Optional[List[int]] = None,
    use_geom: bool = False,  # TODO hard-coded - build option to use 3D graphs
    depth_edges: int = 1,
    reload_interval: int = 0,
    early_reloading: bool = False,
    cl_indices: Optional[List[List[int]]] = None,
) -> None:
    """
    Trains a model to rank molecules.

    Args:
        smiles: List of tuples of SMILES strings
        target: List of target values ([0, 1])
        save_dir: Directory to save the model and results
        featurizer: Featurizer to use
        graph_encoder: Graph encoder to use
        use_fp: Whether to use fingerprints
        lr: Learning rate
        regularization_factor: Regularization factor - small value is sufficient
        n_epochs: Number of epochs
        log_every: Log every n epochs
        val_size: Validation size for random split
        batch_size: Batch size
        num_workers: Number of workers
        read_fn: rdkit function to read molecules
        mc_dropout_samples: Number of MC dropout samples
        dropout_p: Dropout probability
        loss_fn: Loss function
        resume_training: Whether to resume training from last checkpoint
        arrange_layers: How to arrange graph readout layers in the model
        val_indices: List of indices to use for validation
        use_geom: Whether to use 3D geometry (coords) in the graph
        depth_edges: Number of edges to add between atoms separated by 1 bond
        reload_interval: Reload data every n epochs
        early_reloading: Whether to reload data before the end of the epoch
        cl_indices: List of indices to use for curriculum learning
    """
    if cl_indices is not None:
        num_fracs = len(cl_indices)
        n_epochs = reload_interval * num_fracs
    elif reload_interval > 0:
        num_fracs = n_epochs // reload_interval
    else:
        num_fracs = 1

    dm = CustomDataModule(
        smiles=smiles,
        target=target,
        featurizer=featurizer,
        graph_datapath=graph_datapath,
        use_fp=use_fp,
        val_size=val_size,
        batch_size=batch_size,
        num_workers=num_workers,
        read_fn=read_fn,
        use_geom=use_geom,
        random_split=True if val_indices is None else False,
        val_indices=val_indices,
        depth_edges=depth_edges,
        num_fracs=num_fracs,
        cl_indices=cl_indices,
    )

    if val_size > 0:
        use_val = 1.0
    else:
        use_val = 0.0

    # get model and trainer
    model, trainer = get_new_model_and_trainer(
        save_dir=save_dir,
        input_size=dm.dim,
        lr=lr,
        reg_factor=regularization_factor,
        n_epochs=n_epochs,
        log_every=log_every,
        loss_fn=loss_fn,
        mc_dropout_samples=mc_dropout_samples,
        dropout_p=dropout_p,
        encoder=graph_encoder,
        use_fp=use_fp,
        use_geom=use_geom,
        arrange=arrange_layers,
        reload_interval=reload_interval,
        early_reloading=early_reloading,
        use_val=use_val,
    )

    # access last checkpoint
    model_ckpt = os.path.join(save_dir, "checkpoints", "last.ckpt")
    if os.path.exists(model_ckpt) and resume_training:
        LOGGER.info(f"Found checkpoint in {model_ckpt}, resuming training.")
    else:
        model_ckpt = None

    # train model
    trainer.fit(
        model,
        dm,
        ckpt_path=model_ckpt,
    )

    # get scores for scaling
    best_ckpt = trainer.checkpoint_callback.best_model_path

    # load model
    model_best = LitRankNet.load_from_checkpoint(
        best_ckpt,
    )

    if cl_indices is not None:
        train_graphfiles = glob(dm.graph_datapath.split(".")[0] + "_cl", "*.pt")
        # sort by fraction
        train_graphfiles = sorted(
            train_graphfiles, key=lambda x: int(x.split("_cl")[-1].split(".")[0])
        )
    elif reload_interval > 0:
        train_graphfiles = glob(dm.graph_datapath.split(".")[0] + "_frac" + "*.pt")
        # sort by fraction
        train_graphfiles = sorted(
            train_graphfiles, key=lambda x: int(x.split("frac")[-1].split(".")[0])
        )
    else:
        train_graphfiles = [dm.graph_datapath]

    all_scores = []
    for idx, graphfile in enumerate(train_graphfiles):
        current_smiles = dm.smiles_train[idx]
        current_smiles_flat = [item for sublist in current_smiles for item in sublist]
        dm_scaling = CustomDataModule(
            smiles=current_smiles_flat,
            featurizer=featurizer,
            batch_size=batch_size,
            num_workers=num_workers,
            read_fn=read_fn,
            use_fp=use_fp,
            use_geom=use_geom,
            graph_datapath=graphfile,
            depth_edges=depth_edges,
        )

        scores = trainer.predict(model_best, dm_scaling)
        scores = torch.cat(scores).numpy()
        all_scores.append(scores)
    all_scores = np.concatenate(all_scores)
    max_score = np.max(all_scores)
    min_score = np.min(all_scores)

    model_best.update_min_max_values(min_score, max_score)

    # resave model
    trainer.save_checkpoint(best_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model to rank SMILES by synthetic complexity",
        prog="fsscore train",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the csv file. Must contain columns 'smiles_i', 'smiles_j'\
            and 'target'",
        default=INPUT_TRAIN_PATH,
    )
    parser.add_argument(
        "--graph_datapath",
        type=str,
        help="Path to the pt file with featurized graphs and SMILES as ID.",
        default=None,
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
        "--save_dir",
        type=str,
        help="Directory to save the model and results",
        default=os.path.join(MODEL_PATH, f"new_experiment_{date.today()}"),
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
        "--use_fp",
        action="store_true",
        help="Whether to use fingerprints",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
        default=3e-4,
    )
    parser.add_argument(
        "--reg_factor",
        type=float,
        help="Regularization factor - small value is sufficient",
        default=1e-4,
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        help="Number of epochs.",
        default=100,
    )
    parser.add_argument(
        "--log_every",
        type=int,
        help="Log every n steps",
        default=100,
    )
    parser.add_argument(
        "--val_size",
        type=float,
        help="Validation fraction for random split",
        default=0.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
        default=42,
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
        help="Number of workers",
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
        "--subsample",
        type=int,
        help="Subsample the dataset (absolute number)",
        default=None,
    )
    parser.add_argument(
        "--resume_training",
        action="store_true",
        help="Whether to resume training from last checkpoint",
    )
    parser.add_argument(
        "--arrange_layers",
        type=str,
        help="Define arrangement of GNN layers, e.g. GGLGGL. G = GATv2, L = LineEvo.",
        default="GGLGGL",
    )
    parser.add_argument(
        "--fixed_split",
        action="store_true",
        help="Whether to use a fixed train/val split",
    )
    parser.add_argument(
        "--reload_interval",
        type=int,
        help="Reload the dataset every n epochs",
        default=0,
    )
    parser.add_argument(
        "--early_reloading",
        action="store_true",
        help="Reload the dataset before the end of the epoch",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Whether to use curriculum learning.",
    )
    parser.add_argument(
        "--CL_label",
        type=str,
        help="Column name to use for curriculum learning splits.",
        default="tanimoto_bin",
    )

    args = parser.parse_args()

    if args.use_fp and args.featurizer in list(AVAILABLE_GRAPH_FEATURIZERS.keys()):
        raise ValueError(
            f"Cannot use fingerprints with graph featurizer {args.featurizer}"
        )
    elif not args.use_fp and args.featurizer in list(AVAILABLE_FP_FEATURIZERS.keys()):
        raise ValueError(
            f"Cannot use FP featurizer {args.featurizer} without fingerprints"
        )

    if args.early_reloading and args.reload_interval == 0:
        raise ValueError("Cannot use early reloading without reloading interval")

    os.makedirs(MODEL_PATH, exist_ok=True)

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    df = pd.read_csv(args.data_path)
    if args.subsample is not None:
        df = df.sample(args.subsample, random_state=args.seed).reset_index(drop=True)
    smiles_pairs = df[args.compound_cols].values.tolist()
    target = df[args.rating_col].values.tolist()

    if args.curriculum:
        # get df indices based on CL_label
        if args.CL_label not in df.columns:
            raise ValueError(f"Column {args.CL_label} not in dataframe")
        cl_indices = df.groupby(args.CL_label).apply(lambda x: x.index.tolist())
        if args.reload_interval == 0:
            args.reload_interval = args.n_epochs // len(cl_indices)

    if args.fixed_split:
        val_indices = df[df["split"] == "val"].index.tolist()

    # depth edges depends on num LineEvo layers (arrange_layers)
    depth_edges = args.arrange_layers.count("L") - 1
    if args.arrange_layers[-1] != "L":
        depth_edges += 1

    if args.graph_datapath is None and not args.use_fp:
        data_name = os.path.basename(args.data_path).split(".")[0]
        if args.subsample is not None:
            args.graph_datapath = os.path.join(
                PROCESSED_PATH,
                f"{data_name}_sub{args.subsample}_seed{args.seed}_"
                f"depth{depth_edges}_graphs.pt",
            )
        else:
            args.graph_datapath = os.path.join(
                PROCESSED_PATH, f"{data_name}_depth{depth_edges}_graphs_train.pt"
            )

    train(
        smiles=smiles_pairs,
        target=target,
        graph_datapath=args.graph_datapath,
        save_dir=args.save_dir,
        featurizer=args.featurizer,
        graph_encoder=args.graph_encoder,
        use_fp=args.use_fp,
        lr=args.lr,
        regularization_factor=args.reg_factor,
        n_epochs=args.n_epochs,
        log_every=args.log_every,
        val_size=args.val_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mc_dropout_samples=args.mc_dropout_samples,
        dropout_p=args.dropout_p,
        resume_training=args.resume_training,
        arrange_layers=args.arrange_layers,
        val_indices=val_indices if args.fixed_split else None,
        depth_edges=depth_edges,
        reload_interval=args.reload_interval,
        early_reloading=args.early_reloading,
        cl_indices=cl_indices if args.curriculum else None,
    )
