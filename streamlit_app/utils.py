import datetime
import glob
import json
import os
import re
import shutil
import uuid
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
import torch

from fsscore.finetuning import finetune
from fsscore.models.ranknet import LitRankNet
from fsscore.score import Scorer
from fsscore.utils.clustering_mols import ClusterMols
from fsscore.utils.paths import PRETRAIN_MODEL_PATH

ROOT_PATH = "streamlit_app"
UNLABELED_PATH = os.path.join(ROOT_PATH, "data", "unlabeled")
LABELED_PATH = os.path.join(ROOT_PATH, "data", "labeled")
MODELS_PATH = os.path.join(ROOT_PATH, "data", "models")
UNSCORED_PATH = os.path.join(ROOT_PATH, "data", "scoring")
SCORED_PATH = os.path.join(ROOT_PATH, "data", "scored")
os.makedirs(UNLABELED_PATH, exist_ok=True)
os.makedirs(LABELED_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

swap_dict = {0: 0, 1: 2, 2: 1}
side_dict = {0: "", 1: "left", 2: "right"}

dict_initial_values = {
    "state_of_site": "normal",
    "number_pair_feedback": 100,
    "tab_of_site": "Home",
    "number_current_pair": 0,
    "final_validation": False,
    "database_already_shuffled": False,
    "abbreviation": "off",
    "current_pair_hardest": None,
    "highest_pair_seen": 0,
    "pair_number_last_images": -1,
    "database_df": None,
    "dataset_complete": None,
    "dataset_name": None,
}


######################################################


def convertToNumber(s):
    return int.from_bytes(s.encode(), "little")


def init_important_variables(restart=False):
    global dict_initial_values
    for key in dict_initial_values.keys():
        if key not in st.session_state or restart:
            st.session_state[key] = dict_initial_values[key]
    if st.session_state.current_pair_hardest is None:
        st.session_state.current_pair_hardest = np.zeros(
            st.session_state.number_pair_feedback
        )
    if restart:
        st.session_state["session_name"] = str(uuid.uuid4())
        st.session_state.random_state_dataset = (
            convertToNumber(st.session_state.session_name)
        ) % (2**30)
        st.session_state.feedback_already_sent = True
    st.session_state.swap_molecules = np.random.randint(2, size=1000)
    # don't want to shuffle rows as order based on variance


def update_database(filepath):
    dict_initial_values["database_df"] = pd.read_csv(filepath)
    dict_initial_values["dataset_name"] = os.path.basename(filepath)
    st.session_state.update(
        {
            "database_df": dict_initial_values["database_df"],
            "dataset_name": dict_initial_values["dataset_name"],
        }
    )

    st.session_state.number_pair_feedback = min(
        len(dict_initial_values["database_df"]), 100
    )

    init_important_variables()

    # make sure that columns smiles_i and smiles_j are there
    if st.session_state.tab_of_site != "Score molecules":
        if (
            "smiles_i" not in st.session_state.database_df.columns
            or "smiles_j" not in st.session_state.database_df.columns
        ):
            # print warning and exit
            st.error(
                "Warning: File requires 'smiles_i' and 'smiles_j' as column names."
            )
            st.stop()
    else:
        if "smiles" not in st.session_state.database_df.columns:
            # print warning and exit
            st.error("Warning: File requires 'smiles' as column name.")
            st.stop()


def unlabeled_fraction():
    if st.session_state.number_current_pair == 0:
        if "has_label" not in st.session_state.database_df.columns:
            st.session_state.database_df["has_label"] = False
            st.session_state.database_df.to_csv(
                os.path.join(UNLABELED_PATH, st.session_state.dataset_name), index=False
            )

    # st.session_state.database_df = st.session_state.database_df[
    #     ~st.session_state.database_df["has_label"]
    # ]

    st.session_state.number_current_pair = len(
        st.session_state.database_df[st.session_state.database_df["has_label"]]
    )

    if len(st.session_state.database_df) == 0:
        st.session_state.state_of_site = "end_labeling"


def get_pair(number):
    swap = st.session_state.swap_molecules[number]

    pair = (
        st.session_state.database_df["smiles_i"].values[number],
        st.session_state.database_df["smiles_j"].values[number],
    )
    if int(swap):
        return pair[::-1]
    else:
        return pair


def rxn_to_image(smiles, theme="dark"):
    """
    Get a depiction of some smiles.
    """
    url = (
        "https://www.simolecule.com/cdkdepict/depict/wot/png"
        if theme == "dark"
        else "https://www.simolecule.com/cdkdepict/depict/bot/png"
    )

    headers = {"Content-Type": "application/json"}
    response = requests.get(
        url,
        headers=headers,
        params={
            "smi": re.sub("~", ".", smiles),
            "annotate": "cip&r=0",  # "colmap",
            "disp": "bridgehead",
            "zoom": 4,  # 0.74,
            "w": -1,
            "h": -1,
            "abbr": st.session_state.abbreviation,
            "sma": "",
        },
    )
    img = Image.open(BytesIO(response.content))
    st.session_state.pair_number_last_images = st.session_state.number_current_pair
    return img


def update_current_pair_hardest(i):
    st.session_state.current_pair_hardest[st.session_state.number_current_pair] = i


def set_current_pair(i):
    st.session_state.number_current_pair = i
    get_feedback_json()
    if st.session_state.highest_pair_seen < i:
        st.session_state.highest_pair_seen = i


def get_feedback_json(return_dict=False):
    label_path = os.path.join(ROOT_PATH, "data", "labeled")
    os.makedirs(label_path, exist_ok=True)

    if not os.path.exists(os.path.join(label_path, "feedback.json")):
        feedback_json = {}
        with open(os.path.join(label_path, "feedback.json"), "w") as json_file:
            json.dump(feedback_json, json_file)
    else:
        with open(os.path.join(label_path, "feedback.json"), "r") as json_file:
            feedback_json = json.load(json_file)

    if len(st.session_state.database_df) == 0:
        if return_dict:
            return feedback_json
        else:
            return

    feedback_json[f"feedback_{st.session_state.session_name}"] = {}

    feedback_pairs = {}
    number_pairs_completed = 0
    for i in range(st.session_state.highest_pair_seen + 1):
        if st.session_state.current_pair_hardest[i] in [1, 2]:
            number_pairs_completed += 1

        feedback_one_pair = {
            "proposed_smiles_i": st.session_state.database_df["smiles_i"].values[i],
            "proposed_smiles_j": st.session_state.database_df["smiles_j"].values[i],
            "hardest": swap_dict[int(st.session_state.current_pair_hardest[i])]
            if st.session_state.swap_molecules[i]
            else int(st.session_state.current_pair_hardest[i]),
            "side": side_dict[int(st.session_state.current_pair_hardest[i])],
        }
        feedback_pairs[f"pair_{i}"] = feedback_one_pair
        st.session_state.database_df["has_label"].iloc[i] = (
            True if feedback_one_pair["hardest"] != 0 else False
        )

    dict_feedback = {
        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": st.session_state.dataset_name,
        "random_seed": st.session_state.random_state_dataset,
        "feedback_pairs": feedback_pairs,
        "final_validation": st.session_state.final_validation,
        "number_pairs_completed_seen": st.session_state.highest_pair_seen + 1,
        "number_pairs_completed": number_pairs_completed,
    }

    df_complete = pd.read_csv(
        os.path.join(UNLABELED_PATH, st.session_state.dataset_name)
    )

    for i, row in st.session_state.database_df.iterrows():
        idx = row.name
        df_complete.loc[idx, "has_label"] = st.session_state.database_df.loc[
            idx, "has_label"
        ]

    df_complete.to_csv(
        os.path.join(UNLABELED_PATH, st.session_state.dataset_name), index=False
    )

    for key in dict_feedback.keys():
        feedback_json[f"feedback_{st.session_state.session_name}"][key] = dict_feedback[
            key
        ]

    with open(os.path.join(label_path, "feedback.json"), "w") as outfile:
        json.dump(feedback_json, outfile)

    if return_dict:
        return feedback_json


def make_results_df():
    st.session_state.final_validation = True
    feedback_json = get_feedback_json(True)

    output_name = f"{st.session_state.dataset_name.split('.')[0]}_labeled.csv"
    output_path = os.path.join(LABELED_PATH, output_name)

    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
    else:
        df = pd.DataFrame(columns=["smiles_i", "smiles_j", "target"])

    dfs = []
    for key in feedback_json:
        if feedback_json[key]["dataset"] != st.session_state.dataset_name:
            continue
        for subkey in feedback_json[key]["feedback_pairs"]:
            current_dict = feedback_json[key]["feedback_pairs"][subkey]
            if current_dict["hardest"] != 0:
                dfs.append(
                    pd.DataFrame(
                        {
                            "smiles_i": [current_dict["proposed_smiles_i"]],
                            "smiles_j": [current_dict["proposed_smiles_j"]],
                            "target": [current_dict["hardest"] - 1],
                        }
                    )
                )
    if dfs:
        df = pd.concat(dfs, ignore_index=True)

    df.to_csv(output_path, index=False)


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode("utf-8")


def kickoff_finetuning():
    st.session_state.final_validation = True
    st.session_state.state_of_site = "normal"
    st.session_state.tab_of_site = "Fine-tune"
    make_results_df()


def end_of_labeling():
    make_results_df()
    st.session_state.state_of_site = "end_labeling"


def go_to_loading():
    st.session_state.state_of_site = "loading"


def go_to_score_loading():
    st.session_state.state_of_site = "loading_scoring"


def _go_to_labeling():
    datapath = os.path.join(UNLABELED_PATH, f"{st.session_state.paired_ds}.csv")
    go_to_labeling(datapath)


def go_to_labeling(path):
    update_database(path)
    unlabeled_fraction()
    st.session_state.tab_of_site = "Label molecules"
    st.session_state.state_of_site = "_labeling"


def restart_labeling(path):
    make_results_df()
    update_database(path)
    unlabeled_fraction()
    st.session_state.tab_of_site = "Label molecules"
    st.session_state.state_of_site = "_labeling"


def pair_molecules(filepath):
    df = pd.read_csv(filepath)
    assert "smiles" in df.columns, "File requires 'smiles' as column name."

    smiles = df["smiles"].values.tolist()

    # cluster molecules
    cluster_mols = ClusterMols(
        smiles=smiles,
    )
    pairs, _ = cluster_mols.optimize_clustering_parallel()

    pairs_smi = []
    for pair in pairs:
        pairs_smi.append((smiles[pair[0]], smiles[pair[1]]))
    df_pairs = pd.DataFrame(
        {
            "smiles_i": [pair[0] for pair in pairs_smi],
            "smiles_j": [pair[1] for pair in pairs_smi],
        }
    )

    return df_pairs


def rank_by_uncertainty(df, output_path):
    model = LitRankNet.load_from_checkpoint(
        PRETRAIN_MODEL_PATH,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scorer = Scorer(
        model=model,
        featurizer="graph_2D",
        batch_size=32,
        graph_datapath=None,
        mc_dropout_samples=100,
        dropout_p=0.2,
        keep_graphs=False,
        num_workers=None if device == "cuda" else 0,
        device=device,
    )

    smiles = df[["smiles_i", "smiles_j"]].values.tolist()
    scores_mean, scores_var = scorer.score(smiles=smiles)

    df["score_diff_mean"] = scores_mean
    df["score_diff_var"] = scores_var

    df = df.sort_values(by="score_diff_var", ascending=False)
    df.to_csv(output_path, index=False)


def fine_tune():
    save_dir = os.path.join(
        MODELS_PATH,
        f"ft_{st.session_state.dataset_name.split('.')[0]}_"
        f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}",
    )

    df = st.session_state.database_df.copy()

    n_samples = st.session_state.num_samples

    if n_samples > len(df):
        st.error(f"Number of samples exceeds number of available samples ({len(df)}).")
        st.stop()
    if n_samples > -1:
        df = df.sample(n=st.session_state.num_samples, random_state=42)
    else:
        n_samples = len(df)

    smiles_test = df[["smiles_i", "smiles_j"]].values.tolist()
    target_test = df["target"].values.tolist()

    if "target" not in df.columns:
        # print warning and exit
        st.error("Warning: File requires 'target' as column name.")
        st.stop()

    smiles_pairs = df[["smiles_i", "smiles_j"]].values.tolist()
    target = df["target"].values.tolist()

    bs = st.session_state.batch_size
    lr = st.session_state.lr
    epochs = st.session_state.num_epochs

    # TODO not sure if earlystopping is working
    finetune(
        smiles_pairs,
        target,
        save_dir=save_dir,
        filename=st.session_state.dataset_name,
        featurizer="graph_2D",
        model_path=st.session_state.model_path,
        batch_size=bs,
        n_epochs=epochs,
        lr=lr,
        track_improvement=False,
        smiles_val_add=smiles_test,
        target_val_add=target_test,
        earlystopping=True,
        patience=st.session_state.patience,
        val_size=0,  # production
        wandb_mode="disabled",
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=None if torch.cuda.is_available() else 0,
    )

    # move trained model ckpt to models folder
    model_path = glob.glob(os.path.join(save_dir, "checkpoints", "run_*", "last.ckpt"))[
        0
    ]

    st.session_state.ft_model_path = os.path.join(
        MODELS_PATH,
        f'ft_model_{st.session_state.dataset_name.split(".")[0]}_n{n_samples}_bs{bs}.ckpt',  # noqa
    )
    shutil.move(model_path, st.session_state.ft_model_path)


def score():
    df = st.session_state.database_df.copy()
    assert "smiles" in df.columns, "File requires 'smiles' as column name."
    smiles = df["smiles"].values.tolist()

    model = LitRankNet.load_from_checkpoint(
        st.session_state.model_path_scoring,
    )

    # tempoarary filedrop (deleted at end of call)
    graph_datapath = os.path.join(
        SCORED_PATH, f"{st.session_state.dataset_name.split('.')[0]}_graphs_score.pt"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scorer = Scorer(
        model=model,
        featurizer="graph_2D",
        batch_size=32,
        graph_datapath=graph_datapath,
        keep_graphs=False,
        num_workers=None if device == "cuda" else 0,
        device=device,
    )

    scores = scorer.score(smiles)

    df["score"] = scores
    # drop column has_label
    if "has_label" in df.columns:
        df = df.drop(columns=["has_label"])

    df.to_csv(
        os.path.join(
            SCORED_PATH, f"{st.session_state.dataset_name.split('.')[0]}_score.csv"
        ),
        index=False,
    )
