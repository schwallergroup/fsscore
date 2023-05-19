from typing import Dict, List

import pytest
from rdkit import Chem

from intuitive_sc.data.featurizer import (
    AVAILABLE_FEATURIZERS,
    FingerprintFeaturizer,
    get_featurizer,
)


@pytest.mark.parametrize("featurizer_name", AVAILABLE_FEATURIZERS.keys())
def test_featurizer(featurizer_name: str, smiles: List[str], nbits: int):
    """Test that featurizers can be instantiated and used."""
    featurizer = get_featurizer(featurizer_name)
    assert featurizer
    assert featurizer.get_name() == featurizer_name
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    for mol in mols:
        feat = featurizer.get_feat(mol)
        if isinstance(featurizer, FingerprintFeaturizer):
            assert len(feat) == nbits
            assert featurizer.dim() == nbits
        else:
            assert type(feat) == Dict
