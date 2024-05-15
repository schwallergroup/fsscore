import argparse
import os

import pandas as pd
from rdkit import RDLogger
from uspto_processing import (
    chunked_parallel,
    filter_process,
    remove_back_cycles,
    remove_unwanted_elems,
    split_reacs,
)

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None, help="Input file")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory to save the data"
    )

    args = parser.parse_args()

    if args.input_file is None:
        # stop execution if no input file is provided
        print("No input file provided. Exiting Chempapers processing.")
        exit()

    chempapers_raw = os.path.join(args.data_dir, "chempapers_raw")
    os.makedirs(chempapers_raw, exist_ok=True)

    # 1. Load the data
    assert os.path.exists(
        args.input_file
    ), f"Input file {args.input_file} does not exist"

    df = pd.read_table(
        args.input_file,
        header=None,
        sep=r"\§",
        engine="python",
        names=[
            "rxn_smiles",
            "mapped_rxn_smiles",
            "reagents1",
            "reagents2",
            "solvent",
            "catalyst",
            "yield",
        ],
    )
    print(f"1. Chempapers data loaded: n = {len(df)}")

    # 2. split into reactants and products
    df_rxns = pd.DataFrame()
    df_rxns["reactant"] = df["rxn_smiles"].apply(lambda x: x.split(">>")[0])
    df_rxns["product"] = df["rxn_smiles"].apply(lambda x: x.split(">>")[1])
    df_rxns.drop_duplicates(keep="first", inplace=True)
    print(f"2. Chempapers data split into reactants and products: n = {len(df_rxns)}")

    # 3. Split into individual reactants from every reaction
    df_rxns_split = split_reacs(df_rxns)
    print(
        f"3. Chempapers data split into individual reactants: n = {len(df_rxns_split)}"
    )

    # 4. Drop rows with mulitple products
    df_rxns_split = df_rxns_split[
        ~df_rxns_split["product"].str.contains(".", regex=False)
    ]
    print(
        f"4. Chempapers data with multiple products removed: n = {len(df_rxns_split)}"
    )

    # 5. Pre-cleaning
    df_rxns_split = df_rxns_split[df_rxns_split["reactant"] != df_rxns_split["product"]]
    print(
        "5. Chempapers after removing reactant and product that are the same: ",
        len(df_rxns_split),
    )
    # remove duplicates
    df_rxns_split = df_rxns_split.drop_duplicates(keep="first")
    print("5. Chempapers after removing duplicates: ", len(df_rxns_split))
    df_rxns_split.reset_index(drop=True, inplace=True)
    df_rxns_split.to_csv(
        os.path.join(chempapers_raw, "chempapers_preproc.csv"), index=False
    )

    # 6. Remove molecules with less than 4 heavy atoms
    smiles_list = list(
        zip(df_rxns_split["reactant"].tolist(), df_rxns_split["product"].tolist())
    )
    # filter smiles
    smiles_list_filtered = chunked_parallel(smiles_list, filter_process)
    df_rxns_filtered = pd.DataFrame(
        smiles_list_filtered, columns=["reactant", "product"]
    )
    print("6. Chempapers all unfiltered reactions: ", len(df_rxns_filtered))
    df_rxns_filtered = df_rxns_filtered.dropna()
    print("6. Chempapers after removing nan values: ", len(df_rxns_filtered))
    df_rxns_filtered = df_rxns_filtered[
        df_rxns_filtered["reactant"] != df_rxns_filtered["product"]
    ]
    print(
        "6. Chempapers after removing reactant and product that are the same: ",
        len(df_rxns_filtered),
    )
    df_rxns_filtered = df_rxns_filtered.drop_duplicates(keep="first")
    print("6. Chempapers after removing duplicates: ", len(df_rxns_filtered))

    # 7. Remove unwanted elements
    smiles_list = list(
        zip(
            df_rxns_filtered["reactant"].tolist(),
            df_rxns_filtered["product"].tolist(),
        )
    )
    # filter smiles
    smiles_list_filtered = chunked_parallel(smiles_list, remove_unwanted_elems)
    # make df from filtered smiles
    df_rxns_filtered2 = pd.DataFrame(
        smiles_list_filtered, columns=["reactant", "product"]
    )
    df_rxns_filtered2 = df_rxns_filtered2.dropna()
    print("7. Chempapers all unfiltered reactions: ", len(df_rxns_filtered2))
    df_rxns_filtered2.to_csv(
        os.path.join(chempapers_raw, "chempapers_preproc2.csv"), index=False
    )

    # 8. Remove cycles
    df_rxns_filtered2_lin = remove_back_cycles(df_rxns_filtered2)
    print("8. Chempapers after removing cycles: ", len(df_rxns_filtered2_lin))

    df_rxns_filtered2_lin.to_csv(
        os.path.join(chempapers_raw, "chempapers_preproc3_deloop.csv"), index=False
    )

    print("Done processing Chempapers data.")
