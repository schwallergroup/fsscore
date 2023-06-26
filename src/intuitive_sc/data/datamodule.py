from typing import Callable, List, Optional, Tuple

import pytorch_lightning as pl
from rdkit import Chem
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from intuitive_sc.data.dataloader import get_dataloader
from intuitive_sc.data.featurizer import get_featurizer
from intuitive_sc.data.graph_dataset import GraphDatasetMem
from intuitive_sc.utils.logging import get_logger

LOGGER = get_logger(__name__)


class CustomDataModule(pl.LightningDataModule):
    """
    Datamodule for easy data loading and preprocessing using pytorch lightning.
    """

    def __init__(
        self,
        smiles: List[Tuple[str, str]],
        target: Optional[List[float]],
        featurizer: str,
        val_size: float = 0.2,
        batch_size: int = 32,
        use_fp: bool = False,
        use_geom: bool = False,
        graph_datapath: str = None,
        random_split: bool = True,
        val_indices: List[int] = None,
        seed: int = 42,
        num_workers: int = None,
        depth_edges: int = 1,
        read_fn: Callable = Chem.MolFromSmiles,
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
        self.seed = seed
        self.num_workers = num_workers
        self.depth_edges = depth_edges
        self.read_fn = read_fn

        # need to initialize these to get dimensions. \
        # pl would do this automatically when calling .fit()
        self._prepare_data()

    def _prepare_data(self) -> None:
        """
        Prepare dataset for training and testing.
        """
        if not self.use_fp:
            # create graph dataset so not have to compute graph features every time
            LOGGER.info("Getting graph dataset.")
            smiles_all = [s for pair in self.smiles for s in pair]
            smiles_all = list(set(smiles_all))
            self.graph_dataset = GraphDatasetMem(
                smiles=smiles_all,
                processed_path=self.graph_datapath,
                # ids=None, TODO rn ids are smiles
                use_geom=self.use_geom,
                depth=self.depth_edges,
            )
            self.featurizer = get_featurizer(
                self.featurizer, graph_dataset=self.graph_dataset
            )
        else:
            self.graph_dataset = None
            self.featurizer = get_featurizer(self.featurizer, nbits=2048)

    def setup(self, stage: str) -> None:
        """
        Split data into train, val, test, predict sets.
        This process is run on all workers and is called before training.
        Beware of memory issues when using multiple workers.
        """
        if stage == "fit" and self.random_split:
            # XXX uncomment if use pyg.data.Dataset instead of InMemoryDataset
            # if not self.use_fp:
            # self.graph_dataset.load_data()
            (
                self.smiles_train,
                self.smiles_val,
                self.target_train,
                self.target_val,
            ) = train_test_split(
                self.smiles,
                self.target,
                test_size=self.val_size,
                random_state=self.seed,
            )
        if stage == "fit" and not self.random_split:
            # if not self.use_fp:
            #     self.graph_dataset.load_data()
            self.smiles_train = [self.smiles[i] for i in self.val_indices]
            self.smiles_val = [self.smiles[i] for i in self.val_indices]
            self.target_train = [self.target[i] for i in self.val_indices]
            self.target_val = [self.target[i] for i in self.val_indices]

        if stage == "test":
            pass  # TODO create holdout set

        if stage == "predict":
            # if not self.use_fp:
            #     self.graph_dataset.load_data()
            self.smiles_predict = self.smiles

    def train_dataloader(self) -> DataLoader:
        return get_dataloader(
            self.smiles_train,
            self.target_train,
            featurizer=self.featurizer,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            read_fn=self.read_fn,
            graphset=True if self.graph_datapath else False,
        )

    def val_dataloader(self) -> DataLoader:
        if self.smiles_val is None:
            LOGGER.info("No validation set. Trains on full dataset (for production).")
            return None
        return get_dataloader(
            self.smiles_val,
            self.target_val,
            featurizer=self.featurizer,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            read_fn=self.read_fn,
            graphset=True if self.graph_datapath else False,
        )

    def test_dataloader(self) -> DataLoader:
        return get_dataloader(
            self.smiles_test,
            self.target_test,
            featurizer=self.featurizer,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            read_fn=self.read_fn,
            graphset=True if self.graph_datapath else False,
        )

    def predict_dataloader(self) -> DataLoader:
        return get_dataloader(
            self.smiles_predict,
            featurizer=self.featurizer,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            read_fn=self.read_fn,
            graphset=True if self.graph_datapath else False,
        )

    def teardown(self, stage: str) -> None:
        return super().teardown(stage)
