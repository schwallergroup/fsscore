import abc
from functools import partial
from typing import Dict, Type

import numpy as np
import rdkit
from rdkit.Chem import AllChem, DataStructs


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
        """Base class for Morgan fingerprinting featurizer

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


AVAILABLE_FEATURIZERS = AVAILABLE_FP_FEATURIZERS | AVAILABLE_GRAPH_FEATURIZERS

if __name__ == "__main__":
    print(AVAILABLE_FEATURIZERS)
    smi = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    mol = rdkit.Chem.MolFromSmiles(smi)
    morgan = MorganFingerprint()
    print(morgan.get_feat(mol))
