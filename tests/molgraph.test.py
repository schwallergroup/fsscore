import unittest

from rdkit import Chem

from fsscore.data.molgraph import MolGraph


class TestMolGraph(unittest.TestCase):
    def setUp(self):
        self.smiles = "CCO"
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.molgraph = MolGraph()

    def test_mol2graph(self):
        self.molgraph(self.mol)
        self.assertEqual(self.molgraph.n_atoms, 3)
        self.assertEqual(self.molgraph.node_features.shape, (3, 25))
        self.assertEqual(self.molgraph.edge_indices.shape, (2, 2))
        self.assertEqual(self.molgraph.edge_features.shape, (2, 5))
        self.assertEqual(self.molgraph.edges.shape, (2, 2))

    def test_atom_featurizer(self):
        atom = self.mol.GetAtomWithIdx(0)
        features = self.molgraph._atom_featurizer(atom)
        self.assertEqual(features.shape, (25,))

    def test_bond_featurizer(self):
        bond = self.mol.GetBondWithIdx(0)
        features = self.molgraph._bond_featurizer(bond)
        self.assertEqual(features.shape, (5,))

    def test_get_postions(self):
        positions = self.molgraph._get_postions(self.mol)
        self.assertEqual(positions.shape, (3, 3))

    def test_call(self):
        features = self.molgraph(self.mol)
        self.assertIsInstance(features, dict)
        self.assertIn("x", features)
        self.assertIn("edge_index", features)
        self.assertIn("edge_attr", features)
        self.assertIn("edges", features)
        self.assertEqual(features["x"].shape, (3, 25))
        self.assertEqual(features["edge_index"].shape, (2, 2))
        self.assertEqual(features["edge_attr"].shape, (2, 5))
        self.assertEqual(features["edges"].shape, (2, 2))
