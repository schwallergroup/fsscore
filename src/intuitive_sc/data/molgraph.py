"""
Code required to transform a RDKit molecule into a graph with features.
"""
import abc
from typing import Dict, List

import networkx as nx
import numpy as np
import rdkit
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.sparse import coo_matrix

from intuitive_sc.utils.conversions import one_hot_encoding

# TODO adapt
ATOM_TYPES = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I", "Se"]
CHARGES = list(range(-4, 5))
DEGREES = list(range(5))
HYBRIDIZATIONS = list(range(len(Chem.HybridizationType.names) - 1))
CHIRAL_TAGS = ["S", "R", "?"]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
NUM_NODE_FEATURES = (
    len(ATOM_TYPES)
    + 1
    + len(CHARGES)
    + 1
    + 2 * (len(DEGREES) + 1)
    + len(HYBRIDIZATIONS)
    + 1
    + len(CHIRAL_TAGS)
    + 1
    + 2
)


class MolGraph(abc.ABC):
    """
    Class to represent a molecule as a graph.
    """

    def __init__(self, use_geometry: bool = False) -> None:
        """Initialize MolGraph"""
        self.mol = None
        self.n_atoms = None
        self.node_features = None
        self.edge_features = None
        self.edge_indices = None
        self.edges = None
        self.feature_dict = {}
        self.use_geometry = use_geometry

    def _mol2graph(self):
        """Convert RDKit molecule to graph with features"""
        if not self.mol:
            raise ValueError("No molecule to convert to graph.")
        self.n_atoms = self.mol.GetNumAtoms()

        self._get_node_features()
        self._get_edge_features()

    def _get_node_features(self):
        """Instantiates node (atom) features"""
        atoms = map(self.mol.GetAtomWithIdx, range(self.n_atoms))
        self.chiral_centers = Chem.FindMolChiralCenters(
            self.mol, includeUnassigned=True
        )
        self.node_features = torch.tensor(
            np.array([self._atom_featurizer(atom) for atom in atoms]),
            dtype=torch.float,
        )

    def _get_edge_features(self):
        """Instantiates edge (bond) features"""
        adj = Chem.GetAdjacencyMatrix(self.mol)
        # sparse matrix
        coo_adj = coo_matrix(adj)
        # indices of edges in adj matrix
        self.edge_indices = torch.tensor(
            np.array([coo_adj.row, coo_adj.col]), dtype=torch.long
        )
        # atom indices connected by an edge
        self.edges = torch.as_tensor(
            np.array(nx.from_edgelist(self.edge_indices.T.tolist()).edges)
        )

        # add attr
        if self.edges.nelement() != 0:
            bond_features = []
            for i, j in zip(self.edges[:, 0], self.edges[:, 1]):
                bond = self.mol.GetBondBetweenAtoms(int(i), int(j))
                bond_features.append(self._bond_featurizer(bond))
        else:
            bond_features = []
        self.edge_features = torch.tensor(
            np.array(bond_features),  # dtype=torch.float16
        )

    def _atom_featurizer(self, atom: rdkit.Chem.rdchem.Atom):
        """Create node feature vector for a given atom of the graph.
        Descriptors used:
        TODO: Add descriptors and maybe make modular for ablations (dict)
            - Atom type (one-hot)
            - Formal charge (one-hot)
            - Implicit Hs (one-hot)
            - Degree (one-hot)
            - In ring (bool)
            - Is aromatic (bool)
            - Hybridization (one-hot)
            - Chiral tag (one-hot)

        Args:
            atom (rdkit.Chem.rdchem.Atom): Atom of the graph
        Returns:
            np.ndarray: Node feature vector
        """
        atom_types_one_hot = one_hot_encoding(atom.GetSymbol(), ATOM_TYPES)
        charge_one_hot = one_hot_encoding(atom.GetFormalCharge(), CHARGES)
        implicit_h_one_hot = one_hot_encoding(atom.GetTotalNumHs(), DEGREES)
        degree_one_hot = one_hot_encoding(atom.GetDegree(), DEGREES)
        in_ring = 1 if atom.IsInRing() else 0
        is_aromatic = 1 if atom.GetIsAromatic() else 0
        hybrid_one_hot = one_hot_encoding(atom.GetHybridization(), HYBRIDIZATIONS)
        chiral_tag_one_hot = one_hot_encoding("not_chiral", CHIRAL_TAGS)
        for center in self.chiral_centers:
            if center[0] == atom.GetIdx():
                chiral_tag_one_hot = one_hot_encoding(center[1], CHIRAL_TAGS)
                break

        return np.concatenate(
            [
                atom_types_one_hot,
                charge_one_hot,
                implicit_h_one_hot,
                degree_one_hot,
                [in_ring],
                [is_aromatic],
                hybrid_one_hot,
                chiral_tag_one_hot,
            ]
        )

    def _bond_featurizer(self, bond: rdkit.Chem.rdchem.Bond):
        """Create edge feature vector for a given bond of the graph.
        Descriptors used:
            - Bond type (one-hot)
            - Is conjugated (bool)
            - Is in ring (bool)
        Args:
            bond (rdkit.Chem.rdchem.Bond): Bond of the graph
        Returns:
            np.ndarray: Edge feature vector
        """
        bond_type_one_hot = one_hot_encoding(bond.GetBondType(), BOND_TYPES)
        is_conjugated = 1 if bond.GetIsConjugated() else 0
        is_in_ring = 1 if bond.IsInRing() else 0

        return np.concatenate([bond_type_one_hot, [is_conjugated], [is_in_ring]])

    def _get_postions(self, mol: rdkit.Chem.rdchem.Mol):
        params = AllChem.ETKDGv3()
        params.useSmallRingTorsions = True
        num_atoms = mol.GetNumAtoms()
        mol = Chem.AddHs(mol)
        success = AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
        if success == -1:
            return None
        elif success == 0:
            return mol.GetConformer().GetPositions()[:num_atoms, :]

    @property
    def feature_keys(self) -> List:
        return self.feature_dict.keys()

    def __call__(self, mol: rdkit.Chem.rdchem.Mol) -> Dict:
        """Convert RDKit molecule to graph with features.
        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule
        Returns:
            Dict: Dictionary with graph information
        """
        self.mol = mol
        self._mol2graph()

        self.feature_dict["x"] = self.node_features
        self.feature_dict["edge_index"] = self.edge_indices
        self.feature_dict["edge_attr"] = self.edge_features
        self.feature_dict["edges"] = self.edges
        if self.use_geometry:
            self.feature_dict["pos"] = self._get_postions(mol)

        return self.feature_dict


if __name__ == "__main__":
    molgraph = MolGraph()
    smi = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    mol = rdkit.Chem.MolFromSmiles(smi)
    print(molgraph.feature_keys)
    new_graph = molgraph(mol)
    print(molgraph.feature_keys)
    print(
        new_graph["x"].size(),
        new_graph["edge_index"].size(),
        new_graph["edge_attr"].size(),
        new_graph["edges"].size(),
    )
    print("end")
