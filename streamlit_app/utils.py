import datetime
import glob
import json
import os
import re
import time
import uuid
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

######################################################
####################  VARIABLES  ##################### # noqa
######################################################

ROOT_PATH = "streamlit_app"
UNLABELED_PATH = os.path.join(ROOT_PATH, "data", "unlabeled")
LABELED_PATH = os.path.join(ROOT_PATH, "data", "labeled")

swap_dict = {0: 0, 1: 2, 2: 1}

side_dict = {0: "", 1: "left", 2: "right"}

AVAIL_UNLABELED_PATHS = glob.glob(os.path.join(UNLABELED_PATH, "*.csv"))
AVAIL_LABELED_PATHS = glob.glob(os.path.join(LABELED_PATH, "*.csv"))

dict_initial_values = {
    "state_of_site": "normal",
    "number_pair_feedback": 100,
    "tab_of_site": "Label molecules",
    "number_current_pair": 0,
    "final_validation": False,
    "database_already_shuffled": False,
    "abbreviation": "off",
    "current_pair_hardest": None,
    "highest_pair_seen": 0,
    "pair_number_last_images": -1,
    "database_df": pd.read_csv(AVAIL_UNLABELED_PATHS[0]),
    "dataset_complete": None,
    "dataset_name": os.path.basename(AVAIL_UNLABELED_PATHS[0]),
    "df_update_required": False,
}


######################################################
####################  FUNCTIONS  ##################### # noqa
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


def change_requirement():
    st.session_state.df_update_required = True


def update_database(filepath):
    dict_initial_values["database_df"] = pd.read_csv(filepath)
    dict_initial_values["dataset_name"] = os.path.basename(filepath)
    st.session_state.database_df = dict_initial_values["database_df"]
    st.session_state.dataset_name = dict_initial_values["dataset_name"]

    init_important_variables(restart=True)

    st.session_state.df_update_required = False

    # make sure that columns smiles_i and smiles_j are there
    if (
        "smiles_i" not in st.session_state.database_df.columns
        or "smiles_j" not in st.session_state.database_df.columns
    ):
        # print warning and exit
        st.error("Warning: File requires 'smiles_i' and 'smiles_j' as column names.")
        st.stop()


def unlabeled_fraction():
    if "has_label" not in st.session_state.database_df.columns:
        st.session_state.database_df["has_label"] = False
        st.session_state.database_df.to_csv(
            os.path.join(UNLABELED_PATH, st.session_state.dataset_name), index=False
        )

    st.session_state.database_df = st.session_state.database_df[
        ~st.session_state.database_df["has_label"]
    ]

    if len(st.session_state.database_df) == 0:
        st.session_state.state_of_site = "end_labeling"

    if len(st.session_state.database_df) < st.session_state.number_pair_feedback:
        st.session_state.number_pair_feedback = len(st.session_state.database_df)


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

    st.session_state.state_of_site = "get_dataset"


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode("utf-8")


def kickoff_finetuning():
    st.session_state.final_validation = True
    make_results_df()
    st.session_state.state_of_site = "loading"


def end_of_labeling():
    st.session_state.state_of_site = "end_labeling"


def fine_tune_here():
    # Create a model on location 'streamlit_app/data/models/fine_tuned_model.pt'

    # TODO call make_results_df?
    # also remark somewhere that dataset is avail?

    time_i = time.time()
    while time.time() - time_i < 5:
        pass


def restart_labeling(path):
    make_results_df()
    update_database(path)
    unlabeled_fraction()
    st.session_state.tab_of_site = "Label molecules"
    st.session_state.state_of_site = "normal"
