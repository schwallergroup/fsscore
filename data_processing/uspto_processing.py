import argparse
import glob
import os

import gdown
import networkx as nx
import pandas as pd
from pathos import multiprocessing as mp
from rdkit import Chem, RDLogger
from remove_cycle_edges_by_dfs import dfs_remove_back_edges
from tqdm import tqdm

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

URLS = [
    (
        "https://drive.google.com/uc?id=1PbHoIYbm7-69yPOvRA0CrcjojGxVCJCj",
        "src-train.txt",
    ),
    (
        "https://drive.google.com/uc?id=1RRveZmyXAxufTEix-WRjnfdSq81V9Ud9",
        "tgt-train.txt",
    ),
    ("https://drive.google.com/uc?id=1jOIA-20zFhQ-x9fco1H7Q10R6CfxYeZo", "src-val.txt"),
    ("https://drive.google.com/uc?id=19ZNyw7hLJaoyEPot5ntKBxz_o-_R14QP", "tgt-val.txt"),
    (
        "https://drive.google.com/uc?id=1ErtNB29cpSld8o_gr84mKYs51eRat0H9",
        "src-test.txt",
    ),
    (
        "https://drive.google.com/uc?id=1kV9p1_KJm8EqK6OejSOcqRsO8DwOgjL_",
        "tgt-test.txt",
    ),
]

ACCEPT_ELEMS = ["Se", "B", "I", "P", "F", "H", "C", "Br", "N", "O", "S", "Si", "Cl"]


def chunked_parallel(input_list, function, chunks=100, max_cpu=16):
    """chunked_parallel

    Args:
        input_list : list of objects to apply function
        function : Callable with 1 input and returning a single value
        chunks: number of hcunks
        max_cpu: Max num cpus
    """

    cpus = min(mp.cpu_count(), max_cpu)
    pool = mp.Pool(processes=cpus)

    def batch_func(list_inputs):
        outputs = []
        for i in list_inputs:
            outputs.append(function(i))
        return outputs

    list_len = len(input_list)
    num_chunks = min(list_len, chunks)
    step_size = len(input_list) // num_chunks

    chunked_list = [
        input_list[i : i + step_size] for i in range(0, len(input_list), step_size)
    ]

    list_outputs = list(tqdm(pool.imap(batch_func, chunked_list), total=num_chunks))
    full_output = [j for i in list_outputs for j in i]
    return full_output


def get_raw_uspto(data_dir):
    for url, fn in URLS:
        ofn = os.path.join(data_dir, fn)
        if not os.path.exists(ofn):
            gdown.download(url, ofn, quiet=False)
            assert os.path.exists(ofn)
        else:
            print(f"{ofn} exists, skip downloading")


def split_uspto(data_dir):
    raw_filepaths = glob.glob(os.path.join(data_dir, "*.txt"))
    uspto_data_prod = []
    uspto_data_react = []
    for filepath in raw_filepaths:
        if "train" in filepath:
            split = "train"
        elif "val" in filepath:
            split = "val"
        elif "test" in filepath:
            split = "test"
        if "src" in filepath:
            moltype = "product"
            with open(filepath, "r") as f:
                lines = f.readlines()
            # remove spaces between characters
            lines = ["".join(line.strip().split()) for line in lines]
            sub_df = pd.DataFrame(lines, columns=[moltype])
            sub_df["split"] = split
            uspto_data_prod.append(sub_df)
        elif "tgt" in filepath:
            moltype = "reactant"
            with open(filepath, "r") as f:
                lines = f.readlines()
            lines = ["".join(line.strip().split()) for line in lines]
            sub_df = pd.DataFrame(lines, columns=[moltype])
            sub_df["split"] = split
            uspto_data_react.append(sub_df)
        print(f"USPTO: Finished processing {filepath}: n = {len(sub_df)}")
    uspto_data_prod = pd.concat(uspto_data_prod)
    uspto_data_react = pd.concat(uspto_data_react)

    return uspto_data_prod, uspto_data_react


def split_reacs(data):
    # split datapoints in several points if there are multiple reactants
    data_split = []
    for i, row in data.iterrows():
        if "." in row["reactant"]:
            reactants = row["reactant"].split(".")
            for reactant in reactants:
                sub_df = pd.DataFrame(
                    {"reactant": reactant, "product": row["product"]}, index=[i]
                )
                data_split.append(sub_df)
        else:
            sub_df = pd.DataFrame(
                {"reactant": row["reactant"], "product": row["product"]}, index=[i]
            )
            data_split.append(sub_df)
    data_split = pd.concat(data_split)
    return data_split


def remove_back_cycles(df):
    reac_prod = list(zip(df["reactant"], df["product"]))
    combo_digraph = nx.DiGraph()
    combo_digraph.add_edges_from(reac_prod)
    edges_to_remove = dfs_remove_back_edges(combo_digraph)
    # remove edges and lonely nodes
    combo_digraph.remove_edges_from(edges_to_remove)
    combo_digraph.remove_nodes_from(
        [node for node in combo_digraph.nodes if combo_digraph.degree(node) == 0]
    )

    df_lin = df[~df.set_index(["reactant", "product"]).index.isin(edges_to_remove)]
    return df_lin


def clean_uspto(data):
    print("USPTO: all unfiltered reactions: ", len(data))
    # remove rows with nan values
    data = data.dropna()
    print("USPTO: after removing nan values: ", len(data))
    # remove rows with empty strings
    data = data[data["reactant"] != ""]
    data = data[data["product"] != ""]
    print("USPTO: after removing empty strings: ", len(data))
    # remove rows with reactant and product that are the same
    data = data[data["reactant"] != data["product"]]
    print("USPTO: after removing reactant and product that are the same: ", len(data))
    # remove duplicates
    data.drop_duplicates(keep="first", inplace=True)
    print("USPTO: after removing duplicates: ", len(data))
    data.reset_index(drop=True, inplace=True)

    uspto_data_split = split_reacs(data)

    # canonicalize smiles
    print(f"USPTO: after splitting reactants: {len(uspto_data_split)}")
    uspto_data_split.drop_duplicates(keep="first", inplace=True)
    print(f"USPTO: after removing duplicates: {len(uspto_data_split)}")
    uspto_data_split["reactant"] = uspto_data_split["reactant"].apply(
        lambda x: Chem.CanonSmiles(x)
    )
    uspto_data_split["product"] = uspto_data_split["product"].apply(
        lambda x: Chem.CanonSmiles(x)
    )

    uspto_data_split = uspto_data_split[
        uspto_data_split["reactant"] != uspto_data_split["product"]
    ]
    print(
        "USPTO: after removing reactant and product that are the same: ",
        len(uspto_data_split),
    )
    # remove duplicates
    uspto_data_split.drop_duplicates(keep="first", inplace=True)
    print("USPTO after removing canonical duplicates: ", len(uspto_data_split))
    uspto_data_split.reset_index(drop=True, inplace=True)

    return uspto_data_split


def filter_process(both_smiles):
    dropped = False
    reac_smi = both_smiles[0]
    prod_smi = both_smiles[1]
    reactant = Chem.MolFromSmiles(reac_smi)
    product = Chem.MolFromSmiles(prod_smi)
    if reactant is None or product is None:
        dropped = True
    elif reactant.GetNumHeavyAtoms() < 4 or product.GetNumHeavyAtoms() < 4:
        dropped = True
    # canonicalize smiles
    if not dropped:
        new_reac_smi = Chem.CanonSmiles(reac_smi)
        new_prod_smi = Chem.CanonSmiles(prod_smi)
        return (new_reac_smi, new_prod_smi)
    else:
        return (None, None)


def remove_unwanted_elems(both_smiles):
    reac_smi = both_smiles[0]
    prod_smi = both_smiles[1]
    reactant = Chem.MolFromSmiles(reac_smi)
    product = Chem.MolFromSmiles(prod_smi)
    if reactant is None or product is None:
        return (None, None)
    reactant_elements = [atom.GetSymbol() for atom in reactant.GetAtoms()]
    product_elements = [atom.GetSymbol() for atom in product.GetAtoms()]

    # drop if reactant or product contains elements not in accepted_elems
    if not set(reactant_elements).issubset(set(ACCEPT_ELEMS)) or not set(
        product_elements
    ).issubset(set(ACCEPT_ELEMS)):
        return (None, None)
    else:
        return (reac_smi, prod_smi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory to save the data"
    )

    args = parser.parse_args()

    # 1. Get raw USPTO data
    uspto_raw = os.path.join(args.data_dir, "uspto_raw")
    os.makedirs(uspto_raw, exist_ok=True)
    get_raw_uspto(uspto_raw)
    print("1. Finished downloading raw USPTO data")

    # 2. Split USPTO data into products and reactants
    uspto_data_prod, uspto_data_react = split_uspto(uspto_raw)

    # 3. Combine the the original splits
    uspto_prod_train = uspto_data_prod[uspto_data_prod["split"] == "train"]
    uspto_prod_val = uspto_data_prod[uspto_data_prod["split"] == "val"]
    uspto_prod_test = uspto_data_prod[uspto_data_prod["split"] == "test"]
    uspto_react_train = uspto_data_react[uspto_data_react["split"] == "train"]
    uspto_react_val = uspto_data_react[uspto_data_react["split"] == "val"]
    uspto_react_test = uspto_data_react[uspto_data_react["split"] == "test"]

    uspto_train = pd.concat(
        [uspto_react_train.drop(columns=["split"]), uspto_prod_train], axis=1
    )
    uspto_val = pd.concat(
        [uspto_react_val.drop(columns=["split"]), uspto_prod_val], axis=1
    )
    uspto_test = pd.concat(
        [uspto_react_test.drop(columns=["split"]), uspto_prod_test], axis=1
    )
    # combine train, test, val
    uspto_data = pd.concat([uspto_train, uspto_val, uspto_test])
    print(f"3. USPTO: n pairs = {len(uspto_data)}")

    # 4. Clean and filter USPTO
    uspto_data_clean = clean_uspto(uspto_data)

    # 5. Save intermediate data
    uspto_data.to_csv(os.path.join(uspto_raw, "uspto_raw_combo.csv"), index=False)
    uspto_data_clean.to_csv(
        os.path.join(uspto_raw, "uspto_raw_split_combo.csv"), index=False
    )

    # 6. Remove molecules with less than 4 heavy atoms
    smiles_list = list(
        zip(uspto_data_clean["reactant"].tolist(), uspto_data_clean["product"].tolist())
    )

    # filter smiles
    smiles_list_filtered = chunked_parallel(smiles_list, filter_process)
    df_uspto_filtered = pd.DataFrame(
        smiles_list_filtered, columns=["reactant", "product"]
    )
    df_uspto_filtered = df_uspto_filtered.dropna()
    print("6. USPTO: after removing nan values: ", len(df_uspto_filtered))

    # 7. Remove unwanted elements
    smiles_list = list(
        zip(
            df_uspto_filtered["reactant"].tolist(),
            df_uspto_filtered["product"].tolist(),
        )
    )
    # filter smiles
    smiles_list_filtered = chunked_parallel(smiles_list, remove_unwanted_elems)
    # make df from filtered smiles
    df_uspto_filtered2 = pd.DataFrame(
        smiles_list_filtered, columns=["reactant", "product"]
    )
    df_uspto_filtered2 = df_uspto_filtered2.dropna()
    print("7. USPTO: after removing unwanted elements: ", len(df_uspto_filtered2))

    # 8. Save next USPTO version
    df_uspto_filtered2.to_csv(
        os.path.join(uspto_raw, "uspto_split_combo_fil_withloops2.csv"), index=False
    )

    # 9. Create cycles in graph
    df_uspto_filtered2_lin = remove_back_cycles(df_uspto_filtered2)
    print("9. USPTO: after removing back edges: ", len(df_uspto_filtered2_lin))

    # 10. save final USPTO data
    df_uspto_filtered2_lin.to_csv(
        os.path.join(uspto_raw, "uspto_split_combo_fil_deloop.csv")
    )

    print("Done with USPTO")
