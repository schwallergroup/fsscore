from typing import Callable, List, Optional, Tuple

import pytorch_lightning as pl
from rdkit import Chem
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from intuitive_sc.data.dataloader import get_dataloader
from intuitive_sc.data.molgraph import NUM_NODE_FEATURES
from intuitive_sc.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


class CustomDataModule(pl.LightningDataModule):
    """
    Datamodule for easy data loading and preprocessing using pytorch lightning.
    """

    def __init__(
        self,
        smiles: List[Tuple[str, str]],
        featurizer: str,
        target: Optional[List[float]] = None,
        val_size: float = 0.2,
        batch_size: int = 32,
        use_fp: bool = False,
        use_geom: bool = False,
        graph_datapath: str = None,
        random_split: bool = True,
        val_indices: List[int] = None,
        num_workers: int = None,
        depth_edges: int = 1,
        read_fn: Callable = Chem.MolFromSmiles,
        num_fracs: int = 1,
        cl_indices: List[List[int]] = None,
        smiles_val_add: List[Tuple[str, str]] = None,
        target_val_add: List[float] = None,
    ) -> None:
        super().__init__()
        self.smiles = smiles
        self.target = target
        self.featurizer = featurizer
        self.val_size = val_size
        self.batch_size = batch_size
        self.use_fp = use_fp
        self.use_geom = use_geom
        self.graph_datapath = graph_datapath
        self.random_split = random_split
        self.val_indices = val_indices
        self.num_workers = num_workers
        self.depth_edges = depth_edges
        self.read_fn = read_fn
        self.num_fracs = num_fracs
        self.frac_index = 0
        self.dim = 2048 if self.use_fp else NUM_NODE_FEATURES  # TODO hard coded
        self.val_dataloader_instance = None
        self.cl_indices = cl_indices
        self.smiles_val_add = smiles_val_add
        self.target_val_add = target_val_add

    def setup(self, stage: str) -> None:
        """
        Split data into train, val, test, predict sets.
        This process is run on all workers and is called before training.
        Beware of memory issues when using multiple workers.
        """
        if stage == "fit" and self.random_split:
            if self.cl_indices is not None:
                cl_indices_flat = [i for indices in self.cl_indices for i in indices]
            else:
                cl_indices_flat = len(self.smiles) * [None]
            (
                self.smiles_train,
                self.smiles_val,
                self.target_train,
                self.target_val,
                self.cl_indices_train_flat,
                self.cl_indices_val_flat,
            ) = train_test_split(
                self.smiles,
                self.target,
                cl_indices_flat,
                test_size=self.val_size,
            )
            if self.num_fracs > 1:
                if self.cl_indices is not None:
                    # use predefined splits for curriculum learning
                    LOGGER.info("Using predefined splits for curriculum learning.")
                    # group by given self.cl_indices
                    cl_indices_val_set = set(self.cl_indices_val_flat)
                    cl_indices_train = [
                        [i for i in indices if i not in cl_indices_val_set]
                        for indices in self.cl_indices
                    ]
                    self.smiles_train = [
                        [self.smiles[i] for i in indices]
                        for indices in cl_indices_train
                    ]
                    self.target_train = [
                        [self.target[i] for i in indices]
                        for indices in cl_indices_train
                    ]
                    self.smiles_val = [self.smiles[i] for i in self.cl_indices_val_flat]
                    self.target_val = [self.target[i] for i in self.cl_indices_val_flat]

                else:
                    LOGGER.info(f"Splitting data into {self.num_fracs} fractions.")
                    # split into num_fracs fractions (last fraction might be smaller)
                    self.smiles_train = [
                        self.smiles_train[i :: self.num_fracs]
                        for i in range(self.num_fracs)
                    ]
                    self.target_train = [
                        self.target_train[i :: self.num_fracs]
                        for i in range(self.num_fracs)
                    ]
            else:
                self.smiles_train = [self.smiles_train]
                self.target_train = [self.target_train]

        if stage == "fit" and not self.random_split:
            self.smiles_val = [self.smiles[i] for i in self.val_indices]
            self.target_val = [self.target[i] for i in self.val_indices]
            self.smiles_train = [
                [
                    self.smiles[i]
                    for i in range(len(self.smiles))
                    if i not in self.val_indices
                ]
            ]
            self.target_train = [
                [
                    self.target[i]
                    for i in range(len(self.target))
                    if i not in self.val_indices
                ]
            ]

        if stage == "test":
            self.smiles_test = self.smiles
            self.target_test = self.target

        if stage == "predict":
            self.smiles_predict = self.smiles

    def train_dataloader(self) -> DataLoader:
        smiles_current = self.smiles_train[self.frac_index]
        target_current = self.target_train[self.frac_index]
        current_graphpath = self.graph_datapath
        if self.num_fracs > 1:
            LOGGER.info(f"Using fraction {self.frac_index+1} out of {self.num_fracs}.")
            if self.cl_indices is not None:
                current_graphpath = (
                    self.graph_datapath.split(".pt")[0] + f"_cl{self.frac_index}.pt"
                    if self.graph_datapath
                    else None
                )
            else:
                current_graphpath = (
                    self.graph_datapath.split(".pt")[0] + f"_frac{self.frac_index}.pt"
                    if self.graph_datapath
                    else None
                )
            self.frac_index += 1
        if hasattr(self, "train_dataloader_instance"):
            del self.train_dataloader_instance.dataset
            del self.train_dataloader_instance
            self.train_dataloader_instance = None
        self.train_dataloader_instance = get_dataloader(
            smiles_current,
            target_current,
            use_fp=self.use_fp,
            featurizer=self.featurizer,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            read_fn=self.read_fn,
            graph_datapath=current_graphpath,
            use_geom=self.use_geom,
            depth_edges=self.depth_edges,
        )
        return self.train_dataloader_instance

    def val_dataloader(self) -> DataLoader:
        if self.smiles_val is None:
            LOGGER.info("No validation set. Trains on full dataset (for production).")
            return None
        val_frac = int(self.val_size * 100)
        current_graphpath = (
            self.graph_datapath.split(".pt")[0]
            + f"_val{val_frac}_seed{pl.seed_everything()}.pt"
            if self.graph_datapath
            else None
        )
        if self.val_dataloader_instance is None:
            # don't want to reload val (only train) when activating reload_dataloaders
            self.val_dataloader_instance = get_dataloader(
                self.smiles_val,
                self.target_val,
                use_fp=self.use_fp,
                featurizer=self.featurizer,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                read_fn=self.read_fn,
                graph_datapath=current_graphpath,
                use_geom=self.use_geom,
                depth_edges=self.depth_edges,
            )
        if self.smiles_val_add:
            LOGGER.info("Adding additional validation data.")
            add_graphpath = (
                self.graph_datapath.split(".pt")[0] + "_second_valset.pt"
                if self.graph_datapath
                else None
            )
            dataloader_add = get_dataloader(
                self.smiles_val_add,
                self.target_val_add,
                use_fp=self.use_fp,
                featurizer=self.featurizer,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                read_fn=self.read_fn,
                graph_datapath=add_graphpath,
                use_geom=self.use_geom,
                depth_edges=self.depth_edges,
            )
            return [self.val_dataloader_instance, dataloader_add]
        else:
            return self.val_dataloader_instance

    def test_dataloader(self) -> DataLoader:
        return get_dataloader(
            self.smiles_test,
            self.target_test,
            use_fp=self.use_fp,
            featurizer=self.featurizer,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            read_fn=self.read_fn,
            graph_datapath=self.graph_datapath,
            use_geom=self.use_geom,
            depth_edges=self.depth_edges,
        )

    def predict_dataloader(self) -> DataLoader:
        return get_dataloader(
            self.smiles_predict,
            use_fp=self.use_fp,
            featurizer=self.featurizer,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            read_fn=self.read_fn,
            graph_datapath=self.graph_datapath,
            use_geom=self.use_geom,
            depth_edges=self.depth_edges,
        )

    def teardown(self, stage: str) -> None:
        return super().teardown(stage)
