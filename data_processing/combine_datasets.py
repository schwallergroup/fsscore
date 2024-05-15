import argparse
import os
import random

import networkx as nx
import pandas as pd
from uspto_processing import remove_back_cycles

random.seed(42)


def format_rows(row):
    # randomize reactants and products
    if random.random() > 0.5:
        return row["reactant"], row["product"], 1
    else:
        return row["product"], row["reactant"], 0


def grow_from_random_edge(digraph, subgraph, focus_edge, n_max):
    # get edges from connected nodes
    connected_edges = []
    for node in focus_edge:
        new_edges = list(digraph.edges(node))
        if new_edges:
            connected_edges += new_edges
        else:
            # select new random focus edge if node has no edges
            new_focus_edge = random.choice(list(digraph.edges))
            grow_from_random_edge(digraph, subgraph, new_focus_edge, n_max)

    # remove edges that are already in subgraph
    connected_edges = [edge for edge in connected_edges if edge not in subgraph.edges]

    # add all edges that are not already in subgraph
    subgraph.add_edges_from(connected_edges)

    # start recursion if subgraph is not yet big enough
    print("subgraph size:", len(subgraph), "n_max:", n_max)
    if len(subgraph) < n_max:
        for edge in connected_edges:
            # check if the subgraph has reached the desired size before recursion
            if len(subgraph) >= n_max:
                break
            grow_from_random_edge(digraph, subgraph, edge, n_max)

    # return the subgraph when the desired size is reached
    return subgraph


def find_border_edge(graph):
    for edge in graph.edges():
        node1, node2 = edge
        if len(list(graph.neighbors(node1))) < graph.degree(node1) or len(
            list(graph.neighbors(node2))
        ) < graph.degree(node2):
            return edge
    return None


def split_data(digraph):
    #  Get the weakly connected components of the graph
    components = list(nx.weakly_connected_components(digraph))

    # Calculate the total number of edges in the graph
    total_edges = len(digraph.edges())

    # Calculate the target number of test edges
    target_test_edges = int(total_edges * 0.1)

    # get the biggest component and initialize the test graph with the rest
    biggest_component = max(components, key=len)
    train_graph_init = digraph.subgraph(biggest_component).copy()
    test_graph_init = digraph.copy()
    test_graph_init.remove_nodes_from(biggest_component)

    # Get the number of edges in the train and test graphs
    num_train_edges = len(train_graph_init.edges())
    num_test_edges = len(test_graph_init.edges())
    # calculate how many are missing to get to target
    missing_test_edges = target_test_edges - num_test_edges

    # grow test graph from random edge until target is reached
    # random_focus_edge = random.choice(list(train_graph_init.edges))
    random_focus_edge = find_border_edge(train_graph_init)

    focus_subgraph = nx.DiGraph()
    focus_subgraph.add_edge(random_focus_edge[0], random_focus_edge[1])
    test_graph_add = grow_from_random_edge(
        train_graph_init, focus_subgraph, random_focus_edge, missing_test_edges
    )

    # add the additional test graph to the initial test graph
    test_graph = nx.compose(test_graph_init, test_graph_add)

    # Create the train graph
    train_graph = digraph.copy()
    train_graph.remove_nodes_from(test_graph.nodes())

    # Get the number of edges in the train and test graphs
    num_train_edges = len(train_graph.edges())
    num_test_edges = len(test_graph.edges())

    # Print the number of edges in the train and test graphs
    print("Number of edges in train graph:", num_train_edges)
    print("Number of edges in test graph:", num_test_edges)
    # get actual split ratio
    print("Actual test ratio:", num_test_edges / (num_train_edges + num_test_edges))

    return train_graph, test_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uspto_file", type=str, required=True, help="USPTO filepath")
    parser.add_argument(
        "--chempapers_file", type=str, required=True, help="Chempapers filepath"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )

    args = parser.parse_args()

    assert os.path.exists(
        args.uspto_file
    ), f"USPTO file {args.uspto_file} does not exist"

    if not os.path.exists(args.chempapers_file):
        cp_avail = False
        insert = ""
        print(
            f"Chempapers file {args.chempapers_file} does not exist. \n",
            "Continuing without it.",
        )
    else:
        cp_avail = True
        print("Combining datasets: ", args.uspto_file, args.chempapers_file)
        insert = "chempapers_combo_"

    output_intermediate_dir = os.path.join(args.output_dir, "combined_intermediate")
    os.makedirs(output_intermediate_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    df_uspto = pd.read_csv(args.uspto_file, usecols=["reactant", "product"])
    if cp_avail:
        df_cp = pd.read_csv(args.chempapers_file, usecols=["reactant", "product"])

    # 1. combine dataframes
    df_uspto["origin"] = "uspto"
    if cp_avail:
        df_cp["origin"] = "Chempapers"
        df_combined = pd.concat([df_uspto, df_cp], ignore_index=True)
    else:
        df_combined = df_uspto

    print("Combined set before deduplication: ", len(df_combined))
    df_combined = df_combined.drop_duplicates(
        keep="first", subset=["reactant", "product"]
    )
    print("Combined set after deduplication: ", len(df_combined))

    # 2. remove cycles
    df_combined_lin = remove_back_cycles(df_combined)
    print("Combined set after cycle removal: ", len(df_combined_lin))

    # save intermediate file
    df_combined_lin.to_csv(
        os.path.join(output_intermediate_dir, f"uspto_{insert}.csv"), index=False
    )

    # 3. Split data
    reac_prod = list(zip(df_combined_lin["reactant"], df_combined_lin["product"]))
    combo_digraph = nx.DiGraph()
    combo_digraph.add_edges_from(reac_prod)

    train_graph, test_graph = split_data(combo_digraph)
    assert len(set(train_graph.nodes()).intersection(set(test_graph.nodes()))) == 0

    # save train and test graphs
    nx.write_edgelist(
        train_graph,
        os.path.join(output_intermediate_dir, f"uspto_{insert}train.edgelist"),
    )
    nx.write_edgelist(
        test_graph,
        os.path.join(output_intermediate_dir, f"uspto_{insert}test.edgelist"),
    )

    # split the df accordingly
    test_edges = list(test_graph.edges())
    train_edges = list(train_graph.edges())
    df_train = df_combined_lin[
        df_combined_lin.set_index(["reactant", "product"]).index.isin(train_edges)
    ]
    df_test = df_combined_lin[
        df_combined_lin.set_index(["reactant", "product"]).index.isin(test_edges)
    ]

    # save train and test dfs
    df_train.to_csv(
        os.path.join(output_intermediate_dir, f"uspto_{insert}train.csv"), index=False
    )
    df_test.to_csv(
        os.path.join(output_intermediate_dir, f"uspto_{insert}test.csv"), index=False
    )

    # 4. Randomize position of reactants and products
    df_train_reorder = pd.DataFrame(
        [format_rows(row) for _, row in df_train.iterrows()],
        columns=["smiles_i", "smiles_j", "target"],
    )
    df_test_reorder = pd.DataFrame(
        [format_rows(row) for _, row in df_test.iterrows()],
        columns=["smiles_i", "smiles_j", "target"],
    )
    # save the reordered dfs
    df_train_reorder.to_csv(
        os.path.join(args.output_dir, "train_reorder.csv"),
        index=False,
    )
    df_test_reorder.to_csv(
        os.path.join(args.output_dir, "test_reorder.csv"), index=False
    )

    print("Done!")
