""" Dataloader infrastructure adapted from Microsoft's molskill"""
import abc
import multiprocessing
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader as GraphDataLoader

from intuitive_sc.data.featurizer import (
    Featurizer,
    FingerprintFeaturizer,
    GraphFeaturizer,
    get_featurizer,
)
from intuitive_sc.data.graph_dataset import GraphData, GraphDatasetMem
from intuitive_sc.utils.logging import get_logger

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

LOGGER = get_logger(__name__)


def get_dataloader(
    molrpr: Union[List[str], List[Tuple[str, str]]],
    target: Optional[Union[List, np.ndarray]] = None,
    batch_size: int = 32,
    shuffle: bool = False,
    featurizer: Optional[Featurizer] = None,
    num_workers: Optional[int] = None,
    read_fn: Callable = Chem.MolFromSmiles,
    graphset: bool = False,
    graph_dataset: Optional[GraphDatasetMem] = None,
) -> DataLoader:
    """Creates a pytorch dataloader from a list of molecular representations (SMILES).

    Args:
        molrpr: List (or list of pairs) of molecular representations.
        target: Target values, default to None
        batch_size (int, default: 32): batch size for the Dataloader
        shuffle: whether or not shuffling the batch at every epoch. Default to False
        featurizer: Default to 2D graph
        num_workers: Number of processes to use during dataloading.\
            Default is half of the available cores.
        read_fn: rdkit function to read molecules
    """
    if isinstance(molrpr[0], (list, tuple)) and not graphset:
        data = PairDataset(
            molrpr=molrpr,
            target=target,
            featurizer=featurizer,
            read_fn=read_fn,
        )
    elif isinstance(molrpr[0], str) and not graphset:
        data = SingleDataset(
            molrpr=molrpr,
            target=target,
            featurizer=featurizer,
            read_fn=read_fn,
        )
    elif graphset:
        data = graph_dataset
    else:
        raise ValueError(
            "Could not recognize `molrpr` data format. Please check function signature"
        )

    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 2

    if graphset:
        return GraphDataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


class BaseDataset(Dataset, abc.ABC):
    def __init__(
        self,
        molrpr: List,
        target: Optional[Union[List, np.ndarray]] = None,
        read_fn: Callable = Chem.MolFromSmiles,
        featurizer: Optional[Featurizer] = None,
    ) -> None:
        """Base dataset class

        Args:
            molrpr: A list of molecular representations (e.g. SMILES) or a list of
                           tuples (length 2) of molecular representations.
            target: A list of target values for ach molecule present in `molrpr`.\
                Defaults to None.
            read_fn: Function to use to read items in `molrpr`.\
                Defaults to MolFromSmiles.
            featurizer: Featurizer to use. Defaults to None.
        """
        self.molrpr = molrpr
        self.target = target
        self.read_fn = read_fn
        super().__init__()

        if featurizer is None:
            featurizer = get_featurizer("graph_2D")

        self.featurizer = featurizer

    def __getitem__(self, index: int):
        raise NotImplementedError()

    def __len__(self):
        return len(self.molrpr)

    def get_desc(self, smi: str):
        raise NotImplementedError()


class PairDataset(BaseDataset):
    def __init__(
        self,
        molrpr: List,
        target: Optional[Union[List, np.ndarray]] = None,
        read_fn: Callable = Chem.MolFromSmiles,
        featurizer: Optional[Featurizer] = None,
    ) -> None:
        """
        Same as `BaseDataset` but assuming that that `molrpr` is going to contain
        a list of pairs of molecular representations.
        """
        super().__init__(
            molrpr=molrpr,
            target=target,
            read_fn=read_fn,
            featurizer=featurizer,
        )

    def __getitem__(
        self, index: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Tuple[Dict, Dict],
        Tuple[Tuple[Dict, Dict], torch.Tensor],
    ]:
        molrpr_index = self.molrpr[index]
        smi_i, smi_j = molrpr_index[0], molrpr_index[1]
        desc_i, desc_j = self.get_desc(smi_i), self.get_desc(smi_j)
        if self.target is not None:
            target = torch.FloatTensor([self.target[index]])
            return (desc_i, desc_j), target
        else:
            return (desc_i, desc_j)

    def get_desc(self, smiles: str) -> Union[torch.Tensor, GraphData]:
        if isinstance(self.featurizer, FingerprintFeaturizer):
            mol = self.read_fn(smiles)
            return torch.from_numpy(self.featurizer.get_feat(mol))
        elif isinstance(self.featurizer, GraphFeaturizer):
            return self.featurizer.get_feat(smiles=smiles)


class SingleDataset(PairDataset):
    def __init__(
        self,
        molrpr: List,
        target: Optional[Union[List, np.ndarray]] = None,
        read_fn: Callable = Chem.MolFromSmiles,
        featurizer: Optional[Featurizer] = None,
    ) -> None:
        """
        Same as `BaseDataset` but assuming `molrpr` is going
        to contain a list of one molecular representation
        """
        super().__init__(
            molrpr=molrpr,
            target=target,
            read_fn=read_fn,
            featurizer=featurizer,
        )

    def __getitem__(
        self, index: int
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Dict,
        Tuple[Dict, torch.Tensor],
    ]:
        molrpr_index = self.molrpr[index]
        desc = self.get_desc(molrpr_index)
        if self.target is not None:
            target = torch.FloatTensor([self.target[index]])
            return desc, target
        else:
            return desc
