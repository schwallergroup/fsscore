"""
Create a graph dataset class that inherits from torch_geometric.data.Dataset.
Code adapted from LineEvo (https://github.com/fate1997/LineEvo/tree/main)
"""
import copy
import os
from collections import defaultdict
from itertools import chain, combinations
from typing import List, Tuple, Union

# from torch_sparse import SparseTensor
import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.separate import separate
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
        return f"GraphData(ID={self.ID}, num_node={self.num_nodes},\
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
        targets: List[float] = None,
    ):
        self.processed_path = processed_path
        self.smiles = smiles
        if isinstance(smiles[0], str):
            self.smiles_flat = smiles
        elif isinstance(smiles[0], (list, tuple)):
            self.smiles_flat = list(chain(*smiles))
        self.data_list = []
        self.use_geom = use_geom
        self.depth = depth
        self.targets = targets
        if ids is None:
            self.ids = self.smiles_flat
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
        self.data, self.slices = torch.load(self.processed_file_names[0])

    @property
    def processed_file_names(self) -> Union[str, List[Union[str, int]]]:
        return [self.processed_path]

    def process(self):
        for i, smi in enumerate(tqdm(self.smiles_flat, desc="graph data processing")):
            mol = Chem.MolFromSmiles(smi)
            feature_dict = self.featurizer(mol)
            data = GraphData(ID=self.ids[i])

            # postions
            if "pos" not in feature_dict.keys() and self.use_geom:
                print("Warning: No positions found for molecule {}".format(self.ids[i]))
            if self.use_geom:
                data.pos = torch.tensor(feature_dict["pos"])

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

    def len(self):
        return len(self.smiles)

    def __repr__(self):
        return f"GraphDataset({self.name}, num_mols={len(self.smiles_flat)},\
              num_points={self.__len__()})"

    def __getitem__(
        self, index: int
    ) -> Union[
        GraphData,
        Tuple[GraphData, GraphData],
        Tuple[Tuple[GraphData, GraphData], torch.Tensor],
    ]:
        if isinstance(self.smiles[0], (list, Tuple)):
            smi_i, smi_j = self.smiles[index]
            data_i = self.get(smi_i)
            data_j = self.get(smi_j)
            if self.targets is None:
                return (data_i, data_j)
            target = torch.FloatTensor([self.targets[index]])
            return (data_i, data_j), target
        elif isinstance(self.smiles[0], str):
            smi = self.smiles[index]
            data = self.get(smi)
            if self.targets is None:
                return data
            target = torch.FloatTensor([self.targets[index]])
            return data, target

    def get(self, ID: Union[int, str]) -> Data:
        """
        Copied from torch_geometric.data.InMemoryDataset and adapt indexing based on ID.
        """
        idx = self._data["ID"].index(ID)

        if len(self.smiles_flat) == 1:
            return copy.copy(self._data)

        if not hasattr(self, "_data_list") or self._data_list is None:
            self._data_list = len(self.smiles_flat) * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        self._data_list[idx] = copy.copy(data)

        assert data["ID"] == ID

        return data

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
