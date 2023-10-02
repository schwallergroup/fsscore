import pytest

from intuitive_sc.models.ranknet import LitRankNet
from intuitive_sc.score import Scorer
from intuitive_sc.utils.paths import BEST_MODEL_PATH


@pytest.fixture(scope="module")
def scorer():
    model = LitRankNet.load_from_checkpoint(BEST_MODEL_PATH)
    scorer = Scorer(model=model, featurizer="graph_2D")
    return scorer


def test_score(scorer):
    smiles = ["CCO", "CCC", "CCCC"]
    scores = scorer.score(smiles)
    assert len(scores) == len(smiles)
    assert all(isinstance(score, float) for score in scores)