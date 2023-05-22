"""
Various featurizers for molecules. Script structure is inspired by molskill.
"""
import abc
from functools import partial
from typing import Dict, Type, Union

import numpy as np
import rdkit
from rdkit.Chem import AllChem, DataStructs

from intuitive_sc.data.graph_dataset import GraphData, GraphDataset
from intuitive_sc.data.molgraph import MolGraph


class Featurizer(abc.ABC):
    def __init__(self) -> None:
        """Base featurizer class"""
        super().__init__()

    def get_feat(self, mol: rdkit.Chem.rdchem.Mol) -> np.ndarray:
        """Base method to compute fingerprints

        Args:
            mol (rdkit.Chem.rdchem.Mol)s
        """
        raise NotImplementedError()

    def dim(self) -> int:
        """Size of the returned feature"""
        raise NotImplementedError()

    def get_name(self) -> str:
        """Name of the featurizer"""
        raise NotImplementedError()


class FingerprintFeaturizer(Featurizer):
    def __init__(self, nbits: int) -> None:
        """Base fingerprint class

        Args:
            nbits (int): Fingerprint length
        """
        self.nbits = nbits
        super().__init__()

    def dim(self) -> int:
        return self.nbits


class GraphFeaturizer(Featurizer):
    def __init__(self) -> None:
        """Base graph featurizer class"""
        super().__init__()


AVAILABLE_FP_FEATURIZERS: Dict[str, Type[FingerprintFeaturizer]] = {}
AVAILABLE_GRAPH_FEATURIZERS: Dict[str, Type[GraphFeaturizer]] = {}


def register_featurizer(name: str):
    def register_function(cls: Type[Featurizer]):
        if issubclass(cls, FingerprintFeaturizer):
            AVAILABLE_FP_FEATURIZERS[name] = partial(cls, count=False)
            AVAILABLE_FP_FEATURIZERS[name + "_count"] = partial(cls, count=True)
        elif issubclass(cls, GraphFeaturizer):
            AVAILABLE_GRAPH_FEATURIZERS[name] = cls
        else:
            raise ValueError("Not recognized descriptor type.")
        return cls

    return register_function


@register_featurizer(name="morgan")
class MorganFingerprint(FingerprintFeaturizer):
    def __init__(
        self, nbits: int = 2048, bond_radius: int = 2, count: bool = False
    ) -> None:
        """Base class for Morgan fingerprinting featurizer.

        Args:
            bond_radius (int, optional): Bond radius. Defaults to 2.
            count (bool, optional): Whether to use count fingerprints.\
                Defaults to False.
        """
        self.bond_radius = bond_radius
        self.count = count
        super().__init__(nbits=nbits)

    def get_feat(self, mol: rdkit.Chem.rdchem.Mol) -> np.ndarray:
        fp_fun = (
            AllChem.GetHashedMorganFingerprint
            if self.count
            else AllChem.GetMorganFingerprintAsBitVect
        )
        fp = fp_fun(mol, self.bond_radius, self.nbits)
        arr = np.zeros((1,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr


# TODO entry point to select specific features
@register_featurizer(name="graph_2D")
class Graph2DFeaturizer(GraphFeaturizer):
    def __init__(self, graph_dataset: GraphDataset = None) -> None:
        """Base class for 2D graph featurizer
        Args:
            graph_dataset (GraphDataset, optional): Loaded graph dataset.
        """
        self.graph_dataset = graph_dataset
        super().__init__()

    def get_feat(
        self, mol: rdkit.Chem.rdchem.Mol = None, smiles: str = None
    ) -> Union[Dict, GraphData]:
        """Base method to compute graph features. If graph_dataset is provided,
        the graph is loaded from the dataset. Otherwise, the graph is computed
        from the SMILES string (does not yet work in training).

        Args:
            mol (rdkit.Chem.rdchem.Mol): Molecule
            smiles (str): SMILES string
        """
        if self.graph_dataset is not None:
            return self.graph_dataset.get_data(ID=smiles)
        else:
            if mol is None:
                mol = rdkit.Chem.MolFromSmiles(smiles)
            graph_container = MolGraph()
            return graph_container(mol)

    def dim(self) -> int:
        """Size of the returned feature"""
        return self.graph_dataset.node_dim


@register_featurizer(name="graph_3D")
class Graph3DFeaturizer(GraphFeaturizer):
    def __init__(self, graph_dataset: GraphDataset = None) -> None:
        """Base class for 3D graph featurizer

        Args:
            graph_dataset (GraphDataset, optional): Loaded graph dataset.
        """
        self.graph_dataset = graph_dataset
        super().__init__()

    def get_feat(
        self, mol: rdkit.Chem.rdchem.Mol = None, smiles: str = None
    ) -> Union[Dict, GraphData]:
        """Base method to compute graph features

        Args:
            mol (rdkit.Chem.rdchem.Mol): Molecule
            smiles (str): SMILES string
        """
        if self.graph_dataset is not None:
            return self.graph_dataset.get_data(smiles)
        else:
            if mol is None:
                mol = rdkit.Chem.MolFromSmiles(smiles)
            graph_container = MolGraph(use_geometry=True)
            return graph_container(mol)

    def dim(self) -> int:
        """Size of the returned feature"""
        return self.graph_dataset.node_dim


def get_featurizer(featurizer_name: str, **kwargs) -> Featurizer:
    """Basic factory function for fp featurizers"""
    return AVAILABLE_FEATURIZERS[featurizer_name](**kwargs)


AVAILABLE_FEATURIZERS = AVAILABLE_FP_FEATURIZERS | AVAILABLE_GRAPH_FEATURIZERS

if __name__ == "__main__":
    print(AVAILABLE_FEATURIZERS)
    smi = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    mol = rdkit.Chem.MolFromSmiles(smi)
    morgan = MorganFingerprint()
    print(morgan.get_feat(mol))
    graph = GraphFeaturizer()
    print(graph.get_feat(mol))
