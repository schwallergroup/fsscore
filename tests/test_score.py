import os

import numpy as np
import pytest

from fsscore.models.ranknet import LitRankNet
from fsscore.score import Scorer
from fsscore.utils.paths import PRETRAIN_MODEL_PATH, PROCESSED_PATH


@pytest.fixture(scope="module")
def scorer():
    model = LitRankNet.load_from_checkpoint(PRETRAIN_MODEL_PATH)
    graphpath = os.path.join(PROCESSED_PATH, "graph_data_test.pt")
    scorer = Scorer(model=model, featurizer="graph_2D", graph_datapath=graphpath)
    return scorer


def test_score(scorer):
    smiles = ["CCO", "CCC", "CCCC"]
    scores = scorer.score(smiles)
    assert len(scores) == len(smiles)
    assert isinstance(scores, np.ndarray)
