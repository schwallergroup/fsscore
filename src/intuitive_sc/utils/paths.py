import os

ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATA_PATH = os.path.join(ROOT_PATH, "data")
MODEL_PATH = os.path.join(ROOT_PATH, "models")
RESULTS_PATH = os.path.join(ROOT_PATH, "results")
# TODO add default model checkpoint path

INPUT_TRAIN_PATH = os.path.join(
    DATA_PATH, "combo", "uspto_cjhif_combo_train_reorder.csv"
)
INPUT_TEST_PATH = os.path.join(DATA_PATH, "combo", "uspto_cjhif_combo_test_reorder.csv")
