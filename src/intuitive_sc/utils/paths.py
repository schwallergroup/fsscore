import os

ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATA_PATH = os.path.join(ROOT_PATH, "data")
MODEL_PATH = os.path.join(ROOT_PATH, "models")
RESULTS_PATH = os.path.join(ROOT_PATH, "results")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")
BEST_MODEL_PATH = os.path.join(
    MODEL_PATH,
    "combo_train_graph_GGLGGL_ep25x10_2023-09-16",
    "checkpoints",
    "ranknet-epoch=242-best_val_loss.ckpt",
)
# TODO add default model checkpoint path

INPUT_TRAIN_PATH = os.path.join(
    DATA_PATH, "combo", "uspto_cjhif_combo_train_reorder.csv"
)
INPUT_TEST_PATH = os.path.join(DATA_PATH, "combo", "uspto_cjhif_combo_test_reorder.csv")
