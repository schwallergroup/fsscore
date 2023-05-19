import argparse
import os
from typing import Callable, List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F
from rdkit import Chem
from sklearn.model_selection import train_test_split

from intuitive_sc.data.dataloader import get_dataloader
from intuitive_sc.data.featurizer import (
    AVAILABLE_FEATURIZERS,
    AVAILABLE_FP_FEATURIZERS,
    AVAILABLE_GRAPH_FEATURIZERS,
    get_featurizer,
)
from intuitive_sc.data.graph_dataset import GraphDataset
from intuitive_sc.models.gnn import AVAILABLE_GRAPH_ENCODERS
from intuitive_sc.models.nn_utils import get_new_model_and_trainer
from intuitive_sc.utils.logging import get_logger
from intuitive_sc.utils.paths import DATA_PATH, MODEL_PATH

LOGGER = get_logger(__name__)


# TODO add option to have different loss (so could also have hinge loss)
# TODO should have option for fixed train/validation/test split
def train(
    smiles: List[Tuple[str, str]],
    target: List[float],
    graph_datapath: Optional[str] = None,
    save_dir: Optional[str] = None,
    graph_encoder: Optional[str] = None,
    use_fp: bool = False,
    lr: float = 3e-4,
    regularization_factor: float = 1e-4,
    n_epochs: int = 100,
    log_every: int = 10,
    val_size: float = 0.0,
    seed: Optional[int] = None,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    read_fn: Callable = Chem.MolFromSmiles,
    mc_dropout_samples: int = 1,
    dropout_p: float = 0.0,
    loss_fn: Callable = F.binary_cross_entropy_with_logits,
    resume_training: bool = False,
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
        seed: Random seed
        batch_size: Batch size
        num_workers: Number of workers
        read_fn: rdkit function to read molecules
        mc_dropout_samples: Number of MC dropout samples
        dropout_p: Dropout probability
        loss_fn: Loss function
    """
    use_geom = False  # TODO build option to use 3D graphs
    if not use_fp:
        # create graph dataset so not have to compute graph features every time
        LOGGER.info("Getting graph dataset.")
        smiles_all = [s for pair in smiles for s in pair]
        smiles_all = list(set(smiles_all))
        graph_dataset = GraphDataset(
            smiles=smiles_all,
            processed_path=graph_datapath,
            # ids=None, TODO rn ids are smiles
            use_geom=use_geom,
        )
        graph_dataset.transform(depth=1)  # TODO args for this (n layers of readout)
        featurizer = get_featurizer(args.featurizer, graph_dataset=graph_dataset)
    else:
        graph_dataset = None
        featurizer = get_featurizer(args.featurizer, nbits=2048)

    # get dataloaders
    if val_size > 0:
        smiles_train, smiles_val, target_train, target_val = train_test_split(
            smiles, target, test_size=val_size, random_state=seed
        )
        dataloader_train = get_dataloader(
            smiles_train,
            target_train,
            batch_size=batch_size,
            featurizer=featurizer,
            num_workers=num_workers,
            read_fn=read_fn,
            graphset=True if graph_dataset is not None else False,
        )
        dataloader_val = get_dataloader(
            smiles_val,
            target_val,
            batch_size=batch_size,
            featurizer=featurizer,
            num_workers=num_workers,
            read_fn=read_fn,
            graphset=True if graph_dataset is not None else False,
        )
    else:
        dataloader_train = get_dataloader(
            smiles,
            target,
            batch_size=batch_size,
            featurizer=featurizer,
            num_workers=num_workers,
            read_fn=read_fn,
            graphset=True if graph_dataset is not None else False,
        )
        dataloader_val = None
        LOGGER.info("No validation set. Trains on full dataset (for production).")

    # get model and trainer
    model, trainer = get_new_model_and_trainer(
        save_dir=save_dir,
        input_size=dataloader_train.dataset.featurizer.dim(),
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
        dataloader_train,
        dataloader_val,
        ckpt_path=model_ckpt,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model to rank SMILES by synthetic complexity",
        prog="intuitive_sc train",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the csv file. Must contain columns 'smiles_i', 'smiles_j'\
            and 'target'",
        default=os.path.join(DATA_PATH, "data.csv"),
    )
    parser.add_argument(
        "--graph_datapath",
        type=str,
        help="Path to the pt file with featurized graphs and SMILES as ID.",
        default="",
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
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save the model and results",
        default=os.path.join(MODEL_PATH, "new_experiment"),
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
        help="Number of epochs",
        default=100,
    )
    parser.add_argument(
        "--log_every",
        type=int,
        help="Log every n epochs",
        default=10,
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
        "--hinge_loss",
        action="store_true",
        help="Whether to use hinge loss",
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

    args = parser.parse_args()

    if args.use_fp and args.featurizer in list(AVAILABLE_GRAPH_FEATURIZERS.keys()):
        raise ValueError(
            f"Cannot use fingerprints with graph featurizer {args.featurizer}"
        )
    elif not args.use_fp and args.featurizer in list(AVAILABLE_FP_FEATURIZERS.keys()):
        raise ValueError(
            f"Cannot use FP featurizer {args.featurizer} without fingerprints"
        )

    os.makedirs(MODEL_PATH, exist_ok=True)

    pl.seed_everything(args.seed, workers=True)

    df = pd.read_csv(args.data_path)
    if args.subsample is not None:
        df = df.sample(args.subsample)
    smiles_pairs = df[args.compound_cols].values.tolist()  # TODO check this works
    target = df[args.rating_col].values.tolist()

    if not os.path.exists(args.graph_datapath) and not args.use_fp:
        data_name = os.path.basename(args.data_path).split(".")[0]
        args.graph_datapath = os.path.join(DATA_PATH, f"{data_name}_graphs.pt")

    train(
        smiles=smiles_pairs,
        target=target,
        graph_datapath=args.graph_datapath,
        save_dir=args.save_dir,
        graph_encoder=args.graph_encoder,
        use_fp=args.use_fp,
        lr=args.lr,
        regularization_factor=args.reg_factor,
        n_epochs=args.n_epochs,
        log_every=args.log_every,
        val_size=args.val_size,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mc_dropout_samples=args.mc_dropout_samples,
        dropout_p=args.dropout_p,
        # TODO import class for hinge loss
        loss_fn="hinge" if args.hinge_loss else F.binary_cross_entropy_with_logits,
        resume_training=args.resume_training,
    )
