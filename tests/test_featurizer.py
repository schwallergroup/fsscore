from typing import List

import pytest
from rdkit import Chem

from fsscore.data.featurizer import (
    AVAILABLE_FEATURIZERS,
    FingerprintFeaturizer,
    get_featurizer,
)


@pytest.fixture
def smiles() -> List[str]:
    return ["CCO", "CNC", "CCN", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]


@pytest.fixture
def nbits() -> int:
    return 2048


@pytest.mark.parametrize("featurizer_name", AVAILABLE_FEATURIZERS.keys())
def test_featurizer(featurizer_name: str, smiles: List[str], nbits: int):
    """Test that featurizers can be instantiated and used."""
    featurizer = get_featurizer(featurizer_name)
    assert featurizer
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    for mol in mols:
        feat = featurizer.get_feat(mol)
        if isinstance(featurizer, FingerprintFeaturizer):
            assert len(feat) == nbits
            assert featurizer.dim() == nbits
        else:
            assert type(feat) == dict
            assert feat["x"].shape[1] == featurizer.dim()
