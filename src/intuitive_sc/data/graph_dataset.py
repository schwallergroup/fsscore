"""
Create a graph dataset class that inherits from torch_geometric.data.Dataset.
Code adapted from LineEvo (https://github.com/fate1997/LineEvo/tree/main)
"""
import os
from collections import defaultdict
from itertools import chain, combinations
from typing import List, Union

# from torch_sparse import SparseTensor
import networkx as nx
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, Dataset, InMemoryDataset
from tqdm import tqdm

from intuitive_sc.data.molgraph import NUM_NODE_FEATURES, MolGraph
from intuitive_sc.utils.logging import get_logger

LOGGER = get_logger(__name__)


class GraphData(Data):
    def __init__(self, ID: Union[str, int] = None):
        super().__init__()
        self.ID = ID
        self.num_nodes = None
        self.num_bonds = None
        self.num_nodes_features = None
        self.num_bonds_features = None
        self.evolve_edges = 0
        self.flat_index = None

    def __repr__(self):
        return f"GraphData(ID={self.ID}, num_node={self.num_nodes}, \
            num_bond={self.num_bonds}, num_node_features={self.num_nodes_features})"

    def __inc__(self, key, value, *args, **kwargs):
        if "edges_" in key:
            return (
                self.num_nodes
                if key[-1] == "0"
                else getattr(self, "edges_" + str(int(key[-1]) - 1)).size(0)
            )
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDatasetMem(InMemoryDataset):
    def __init__(
        self,
        processed_path: str,
        smiles: List[str] = None,
        ids: List[str] = None,
        use_geom: bool = False,  # TODO add option to use confs instead of smiles
        depth: int = 1,
    ):
        self.processed_path = processed_path
        self.smiles = smiles
        self.data_list = []
        self.use_geom = use_geom
        self.depth = depth
        if ids is None:
            self.ids = self.smiles
        else:
            self.ids = ids
        self.featurizer = MolGraph(use_geometry=self.use_geom)
        self.name = os.path.basename(processed_path).split(".")[0]
        super().__init__(
            root=os.path.dirname(processed_path),
            transform=None,
            pre_filter=None,
            pre_transform=self.pre_transform,
        )
        self.data, self.slices = torch.load(self.processed_path)

    @property
    def processed_file_names(self) -> Union[str, List[Union[str, int]]]:
        return [self.processed_path]

    def process_mol(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        feature_dict = self.featurizer(mol)
        data = GraphData(ID=smiles)

        # Positions
        if "pos" not in feature_dict.keys() and self.use_geom:
            print("Warning: No positions found for {}".format(smiles))
        if self.use_geom:
            data.pos = torch.tensor(feature_dict["pos"], dtype=torch.float16)

        # Edge index and edges
        data.edge_index = feature_dict["edge_index"]

        # Edge_attr
        if "edge_attr" in feature_dict.keys():
            data.edge_attr = feature_dict["edge_attr"].clone().detach()
            data.num_bonds_features = (
                data.edge_attr.shape[1] if data.edge_attr.shape[0] != 0 else 0
            )

        data.x = feature_dict["x"].clone().detach()

        # Representation info
        data.num_bonds = data.edge_index.shape[1]
        data.num_nodes_features = data.x.shape[1]
        data.num_nodes = data.x.shape[0]

        return data

    def process(self):
        for i, smi in enumerate(tqdm(self.smiles, desc="graph data processing")):
            mol = Chem.MolFromSmiles(smi)
            feature_dict = self.featurizer(mol)
            data = GraphData(ID=self.ids[i])

            # postions
            if "pos" not in feature_dict.keys() and self.use_geom:
                print("Warning: No positions found for molecule {}".format(self.ids[i]))
            if self.use_geom:
                data.pos = torch.tensor(feature_dict["pos"], dtype=torch.float16)

            # edge index and edges
            data.edge_index = feature_dict["edge_index"]

            # edge_attr
            if "edge_attr" in feature_dict.keys():
                data.edge_attr = feature_dict["edge_attr"].clone().detach()
                data.num_bonds_features = (
                    data.edge_attr.shape[1] if data.edge_attr.shape[0] != 0 else 0
                )

            data.x = feature_dict["x"].clone().detach()

            # repr info
            data.num_bonds = data.edge_index.shape[1]
            data.num_nodes_features = data.x.shape[1]
            data.num_nodes = data.x.shape[0]

            self.data_list.append(data)

        # for speed edges already transformed before saving. --> careful with filenames
        # XXX could also use transform() method, which is called per batch
        self.pre_transform()

        data, slices = self.collate(self.data_list)
        torch.save((data, slices), self.processed_path)

    def pre_transform(self):
        for data in self.data_list:
            edges = torch.LongTensor(
                np.array(nx.from_edgelist(data.edge_index.T.tolist()).edges)
            )

            num_nodes = data.x.shape[0]
            isolated_nodes = set(range(num_nodes)).difference(
                set(edges.flatten().tolist())
            )
            edges = torch.cat(
                [edges, torch.LongTensor([[i, i] for i in isolated_nodes])], dim=0
            ).to(torch.long)

            setattr(data, f"edges_{0}", edges)

            for i in range(self.depth):
                num_nodes = edges.shape[0]
                edges = evolve_edges_generater(edges)

                # create edges for isolated nodes
                isolated_nodes = set(range(num_nodes)).difference(
                    set(edges.flatten().tolist())
                )
                edges = torch.cat(
                    [
                        edges,
                        torch.LongTensor(
                            [[i, i] for i in isolated_nodes], device=edges.device
                        ),
                    ],
                    dim=0,
                )

                setattr(data, f"edges_{i+1}", edges)

    def __getitem__(self, index):
        return self.data_list[index]

    def len(self):
        return len(self.data_list)

    def __repr__(self):
        return f"GraphDataset({self.name}, num_mols={self.__len__()})"

    def get(self, ID: Union[int, str]) -> Data:
        return self.data_list[self.ids.index(ID)]

    @property
    def node_dim(self) -> int:
        if self.data_list:
            return self.data_list[0].num_nodes_features
        else:
            return NUM_NODE_FEATURES


# XXX not used: very slow with big dataset but if mem becomes an issue, can use this
# requires loading data (load_data()) in the datamodule!
class GraphDataset(Dataset):
    def __init__(
        self,
        processed_path: str,
        smiles: List[str] = None,
        ids: List[str] = None,
        use_geom: bool = False,  # TODO add option to use confs instead of smiles
        depth: int = 1,
    ):
        self.data_list = []
        self.smiles = smiles
        self.processed_path = processed_path
        self.use_geom = use_geom
        self.depth = depth
        if ids is None:
            self.ids = self.smiles
        else:
            self.ids = ids
        self.featurizer = MolGraph(use_geometry=self.use_geom)
        self.name = os.path.basename(processed_path).split(".")[0]

    def process_mol(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        feature_dict = self.featurizer(mol)
        data = GraphData(ID=smiles)

        # Positions
        if "pos" not in feature_dict.keys() and self.use_geom:
            print("Warning: No positions found for {}".format(smiles))
        if self.use_geom:
            data.pos = torch.tensor(feature_dict["pos"], dtype=torch.float16)

        # Edge index and edges
        data.edge_index = feature_dict["edge_index"]

        # Edge_attr
        if "edge_attr" in feature_dict.keys():
            data.edge_attr = feature_dict["edge_attr"].clone().detach()
            data.num_bonds_features = (
                data.edge_attr.shape[1] if data.edge_attr.shape[0] != 0 else 0
            )

        data.x = feature_dict["x"].clone().detach()

        # Representation info
        data.num_bonds = data.edge_index.shape[1]
        data.num_nodes_features = data.x.shape[1]
        data.num_nodes = data.x.shape[0]

        return data

    def process(self):
        for i, smi in enumerate(tqdm(self.smiles, desc="graph data processing")):
            mol = Chem.MolFromSmiles(smi)
            feature_dict = self.featurizer(mol)
            data = GraphData(ID=self.ids[i])

            # postions
            if "pos" not in feature_dict.keys() and self.use_geom:
                print("Warning: No positions found for molecule {}".format(self.ids[i]))
            if self.use_geom:
                data.pos = torch.tensor(feature_dict["pos"], dtype=torch.float16)

            # edge index and edges
            data.edge_index = feature_dict["edge_index"]

            # edge_attr
            if "edge_attr" in feature_dict.keys():
                data.edge_attr = feature_dict["edge_attr"].clone().detach()
                data.num_bonds_features = (
                    data.edge_attr.shape[1] if data.edge_attr.shape[0] != 0 else 0
                )

            data.x = feature_dict["x"].clone().detach()

            # repr info
            data.num_bonds = data.edge_index.shape[1]
            data.num_nodes_features = data.x.shape[1]
            data.num_nodes = data.x.shape[0]

            self.data_list.append(data)

        torch.save(self.data_list, self.processed_path)

    def load_data(self):
        self.data_list = torch.load(self.processed_path)
        self.transform()

    def transform(self):
        for data in self.data_list:
            edges = torch.LongTensor(
                np.array(nx.from_edgelist(data.edge_index.T.tolist()).edges)
            )

            num_nodes = data.x.shape[0]
            isolated_nodes = set(range(num_nodes)).difference(
                set(edges.flatten().tolist())
            )
            edges = torch.cat(
                [edges, torch.LongTensor([[i, i] for i in isolated_nodes])], dim=0
            ).to(torch.long)

            setattr(data, f"edges_{0}", edges)

            for i in range(self.depth):
                num_nodes = edges.shape[0]
                edges = evolve_edges_generater(edges)

                # create edges for isolated nodes
                isolated_nodes = set(range(num_nodes)).difference(
                    set(edges.flatten().tolist())
                )
                edges = torch.cat(
                    [
                        edges,
                        torch.LongTensor(
                            [[i, i] for i in isolated_nodes], device=edges.device
                        ),
                    ],
                    dim=0,
                )

                setattr(data, f"edges_{i+1}", edges)

    def __getitem__(self, index):
        return self.data_list[index]

    def len(self):
        return len(self.data_list)

    def __repr__(self):
        return f"GraphDataset({self.name}, num_mols={self.__len__()})"

    def get(self, ID: Union[int, str]) -> Data:
        return self.data_list[self.ids.index(ID)]

    @property
    def node_dim(self) -> int:
        if self.data_list:
            return self.data_list[0].num_nodes_features
        else:
            return NUM_NODE_FEATURES


def evolve_edges_generater(edges):
    long = edges[:, 0].tolist() + edges[:, 1].tolist()
    tally = defaultdict(list)
    for i, item in enumerate(long):
        tally[item].append(i if i < len(long) // 2 else i - len(long) // 2)

    output = []
    for _, locs in tally.items():
        if len(locs) > 1:
            output.append(list(combinations(locs, 2)))

    return torch.LongTensor(list(chain(*output)), device=edges.device)


if __name__ == "__main__":
    # use this to generate and save the processed data (e.g. for holdout set)
    df = pd.read_csv("path/to/smiles.csv")
    smiles = df["smiles"].tolist()
    graphset = GraphDataset(
        smiles=smiles, processed_path="path/to/save.pt", use_geom=False
    )
    graphset.transform(1)
