"""
Script to cluster molecules based on their fingerprints in order to
pair them up for fine-tuning.
"""
import argparse
import multiprocessing
import os
import random
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from intuitive_sc.utils.logging_utils import get_logger
from intuitive_sc.utils.paths import DATA_PATH

LOGGER = get_logger(__name__)


class ClusterMols:
    def __init__(
        self,
        smiles: List[str],
        labels: List[int] = [],
        k_clusters: int = 10,
        dissim_cutoff: float = 0.2,
        cluster_method: str = "kmeans",
        buffer: float = None,
    ) -> None:
        self.smiles = smiles
        self.labels = labels
        self.k_clusters = k_clusters
        self.dissim_cutoff = dissim_cutoff
        self.cluster_method = cluster_method
        self.buffer = buffer
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        self.fp = self.get_fingerprints(mols)
        if cluster_method == "kmeans":
            LOGGER.info("Clustering molecules using k-means.")
            self.dists = self.tanimoto_dist_mat2()
            self.cluster_fn = self.kmeans_cluster_fps
        elif cluster_method == "butina":
            # buggy
            raise NotImplementedError
            LOGGER.info("Clustering molecules using Butina.")
            self.dists = self.tanimoto_dist_mat()
            self.cluster_fn = self.butina_cluster_fps
        else:
            raise ValueError("Invalid cluster method.")

    def get_fingerprints(
        self,
        mols: List[Chem.rdchem.Mol],
    ) -> List[DataStructs.cDataStructs.ExplicitBitVect]:
        """
        Get fingerprints for a list of molecules.
        """
        fp = []
        for mol in tqdm(mols, desc="Generating fingerprints"):
            fp.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        return fp

    def cluster_mols(
        self,
    ) -> List[tuple]:
        """
        Cluster molecules based on their fingerprints.
        """
        pairs = self.create_unique_pairs(self.cluster_fn())

        return pairs

    def tanimoto_dist_mat(
        self,
    ) -> List[float]:
        dist_mat = []
        for i in tqdm(
            range(1, len(self.fp)), desc="Calculating distances", total=len(self.fp)
        ):
            sims = DataStructs.BulkTanimotoSimilarity(self.fp[i], self.fp[:i])
            dist_mat.extend(np.array([1 - x for x in sims], dtype=np.float16))
        return dist_mat

    def calculate_dissimilarity(self, i):
        sims = DataStructs.BulkTanimotoSimilarity(self.fp[i], self.fp[:])
        return np.array([1 - x for x in sims], dtype=np.float16)

    def tanimoto_dist_mat2(
        self,
    ) -> np.ndarray:
        n = len(self.fp)
        dissim = (
            np.array(
                [
                    1 - x
                    for x in DataStructs.BulkTanimotoSimilarity(self.fp[i], self.fp[:])
                ],
                dtype=np.float16,
            )
            for i in tqdm(range(n), desc="Calculating distances", total=n)
        )
        dissim = np.array(list(dissim), dtype=np.float16)
        return dissim

    def tanimoto_dist_mat2_parallel(
        self,
    ) -> np.ndarray:
        pairwise_dissimilarity = Parallel(n_jobs=multiprocessing.cpu_count() // 4)(
            delayed(self.calculate_dissimilarity)(i)
            for i in tqdm(
                range(len(self.fp)), desc="Calculating distances", total=len(self.fp)
            )
        )

        return pairwise_dissimilarity

    def optimize_clustering(
        self,
    ) -> List[tuple]:
        if self.cluster_method == "kmeans":
            if len(self.smiles) > 500:
                cluster_params = range(5, 30)
            else:
                cluster_params = range(3, 10)
        elif self.cluster_method == "butina":
            cluster_params = np.arange(0.1, 1, 0.1)
        else:
            raise ValueError("Invalid cluster method.")
        all_scores_combined = []
        for param in tqdm(cluster_params, desc="Optimizing clustering"):
            clusters = self.cluster_fn(param)
            pairs = self.create_unique_pairs(clusters)
            max_pair_coverage = len(pairs) / (len(self.smiles) / 2)
            if len(set(clusters)) == len(self.smiles):
                sil_score = 0
            else:
                sil_score = silhouette_score(self.dists, clusters)
            score_combined = max_pair_coverage * sil_score
            all_scores_combined.append(score_combined)
        best_param = cluster_params[np.argmax(all_scores_combined)]
        clusters = self.cluster_fn(best_param)
        pairs = self.create_unique_pairs(clusters)
        return pairs, best_param

    def optimize_clustering_parallel(
        self,
    ) -> List[tuple]:
        if self.cluster_method == "kmeans":
            if len(self.smiles) > 500:
                cluster_params = range(5, 30)
            else:
                cluster_params = range(3, 10)
        elif self.cluster_method == "butina":
            cluster_params = np.arange(0.1, 1, 0.1)
        else:
            raise ValueError("Invalid cluster method.")
        all_scores = Parallel(n_jobs=multiprocessing.cpu_count() // 2)(
            delayed(self.get_score_combined)(param)
            for param in tqdm(cluster_params, desc="Optimizing clustering")
        )
        best_param = cluster_params[np.argmax(all_scores)]
        clusters = self.cluster_fn(best_param)
        pairs = self.create_unique_pairs(clusters)
        return pairs, best_param

    def get_score_combined(
        self,
        param: float,
    ) -> float:
        clusters = self.cluster_fn(param)
        pairs = self.create_unique_pairs(clusters)
        max_pair_coverage = len(pairs) / (len(self.smiles) / 2)
        if len(set(clusters)) == len(self.smiles):
            sil_score = 0
        else:
            sil_score = silhouette_score(self.dists, clusters)
        score_combined = max_pair_coverage * sil_score
        return score_combined

    def butina_cluster_fps(
        self,
        dissim_cutoff: float = None,
    ) -> List[int]:
        """
        Cluster fingerprints using Butina clustering.
        The threshold is the minimum distance between two clusters.
        """
        if not dissim_cutoff:
            dissim_cutoff = self.dissim_cutoff
        clusters = Butina.ClusterData(
            self.dists, len(self.fp), dissim_cutoff, isDistData=True
        )
        # reorder so that cluster index is in order of molecules
        reordered_clusters = np.zeros(len(self.fp), dtype=int)
        for i, cluster in enumerate(clusters):
            for member in cluster:
                reordered_clusters[member] = i
        return reordered_clusters

    def kmeans_cluster_fps(
        self,
        k_clusters: int = None,
    ) -> List[int]:
        """
        Cluster fingerprints using k-means clustering.
        """
        if not k_clusters:
            k_clusters = self.k_clusters
        # Cluster molecules using k-means
        kmeans = KMeans(n_clusters=k_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.dists)
        return clusters

    def pair_molecules_direct(
        self,
    ) -> List[tuple]:
        """
        Pair molecules with different labels based on their fingerprints.
        """
        pairs = []
        n = len(self.labels)
        for i in range(n):
            for j in range(i + 1, n):
                dissimilarity = 1 - DataStructs.TanimotoSimilarity(
                    self.fp[i], self.fp[j]
                )
                if (
                    dissimilarity > self.dissim_cutoff
                    and self.labels[i] != self.labels[j]
                ):
                    pairs.append((i, j))
        return pairs

    def create_unique_pairs(
        self,
        clusters: List[int],
    ) -> List[tuple]:
        """
        Create a list of pairs of molecules with different labels (if available)
          and cluster IDs.

        Args:
        clusters (list): A list of cluster IDs for each molecule.

        Returns:
        List[Tuple]: A list of pairs of molecules
        """
        # Create a dictionary mapping clusters to molecules
        # keys: cluster index, values: list of mol indices in cluster
        cluster_dict = {}
        for i, cluster_i in enumerate(clusters):
            if cluster_i not in cluster_dict:
                cluster_dict[cluster_i] = []
            cluster_dict[cluster_i].append(i)

        # Create a list of pairs of molecules with different labels
        pairs = []
        mols_added = []
        for cluster_i in cluster_dict:
            # iterate through all remaining clusters
            for cluster_j in cluster_dict:
                if cluster_i != cluster_j:
                    for i in cluster_dict[cluster_i]:
                        for j in cluster_dict[cluster_j]:
                            if self.labels and not self.buffer:
                                if (
                                    self.labels[i] != self.labels[j]
                                    and i not in mols_added
                                    and j not in mols_added
                                ):
                                    pairs.append((i, j))
                                    mols_added.append(i)
                                    mols_added.append(j)
                            elif self.labels and self.buffer:
                                # pairing based on continuous labels
                                # TODO check
                                if (
                                    abs(self.labels[i] - self.labels[j]) >= self.buffer
                                    and i not in mols_added
                                    and j not in mols_added
                                ):
                                    pairs.append((i, j))
                                    mols_added.append(i)
                                    mols_added.append(j)
                            else:
                                if i not in mols_added and j not in mols_added:
                                    pairs.append((i, j))
                                    mols_added.append(i)
                                    mols_added.append(j)
        return pairs


def randomize_cols(row):
    LOGGER.info("Position of columns randomized.")
    if random.random() < 0.5:
        return (
            row["smiles_i"],
            row["smiles_j"],
            row["label_i"],
            row["label_j"],
            row["target"],
        )
    else:
        if row["target"] == 0:
            new_target = 1
        elif row["target"] == 1:
            new_target = 0
        return (
            row["smiles_j"],
            row["smiles_i"],
            row["label_j"],
            row["label_i"],
            new_target,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to a csv file containing SMILES strings.",
    )
    parser.add_argument(
        "--k_clusters",
        type=int,
        default=10,
        help="Number of clusters to use for k-means clustering.",
    )
    parser.add_argument(
        "--dissim_cutoff",
        type=float,
        default=0.2,
        help="Minimum dissimilarity between two clusters.",
    )
    parser.add_argument(
        "--cluster_method",
        type=str,
        default="kmeans",
        help="Method to use for clustering. Options: kmeans, butina.",
    )
    parser.add_argument(
        "--smi_col",
        type=str,
        default="smiles",
        help="Name of the column containing SMILES strings.",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Name of the column containing labels.",
    )
    parser.add_argument(
        "--complex_label",
        type=str,
        default=None,
        help="Name of the label for the more complex molecules. \
            Options: specific label or high/low for continuous labels.",
    )
    parser.add_argument(
        "--use_labels",
        action="store_true",
        help="Whether to use labels to pair molecules.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Whether to optimize clustering parameters.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save pairs.",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=None,
        help="Buffer to use for pairing based on continuous labels.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    smiles = df[args.smi_col].tolist()
    if args.use_labels:
        LOGGER.info("Using labels to pair molecules.")
        labels = df[args.label_col].tolist()
    else:
        LOGGER.info("Not using labels to pair molecules.")
        labels = []

    cluster_mols = ClusterMols(
        smiles=smiles,
        labels=labels,
        k_clusters=args.k_clusters,
        dissim_cutoff=args.dissim_cutoff,
        cluster_method=args.cluster_method,
        buffer=args.buffer,
    )

    # TODO include option to do simple pairing? (when dataset size veery small)
    if args.optimize:
        LOGGER.info("Optimizing clustering parameters.")
        pairs, cluster_param = cluster_mols.optimize_clustering_parallel()
    else:
        LOGGER.info("Using provided clustering parameter.")
        pairs = cluster_mols.cluster_mols()
        cluster_param = (
            args.k_clusters if args.cluster_method == "kmeans" else args.dissim_cutoff
        )

    LOGGER.info(f"Number of pairs: {len(pairs)}")
    # get SMILES strings and labels (if avail) for each pair
    pairs_smi = []
    pairs_labels = []
    for pair in pairs:
        pairs_smi.append((smiles[pair[0]], smiles[pair[1]]))
        if labels:
            pairs_labels.append((labels[pair[0]], labels[pair[1]]))

    # save pairs to csv
    if args.use_labels:
        df_pairs = pd.DataFrame(
            {
                "smiles_i": [pair[0] for pair in pairs_smi],
                "smiles_j": [pair[1] for pair in pairs_smi],
                "label_i": [pair[0] for pair in pairs_labels],
                "label_j": [pair[1] for pair in pairs_labels],
            }
        )
        if args.complex_label:
            if args.buffer:
                if args.complex_label == "high":
                    # 1 if label_j bigger than label_i
                    df_pairs["target"] = df_pairs.apply(
                        lambda row: 1 if row["label_j"] > row["label_i"] else 0, axis=1
                    )
                elif args.complex_label == "low":
                    # 1 if label_j smaller than label_i
                    df_pairs["target"] = df_pairs.apply(
                        lambda row: 1 if row["label_j"] < row["label_i"] else 0, axis=1
                    )
                else:
                    raise ValueError("Invalid complex label.")
            else:
                df_pairs["target"] = df_pairs.apply(
                    lambda row: 1 if args.complex_label == row["label_j"] else 0, axis=1
                )
            # randomize order of pairs in columns
            df_pairs = df_pairs.apply(randomize_cols, axis=1, result_type="broadcast")

    else:
        df_pairs = pd.DataFrame(
            {
                "smiles_i": [pair[0] for pair in pairs_smi],
                "smiles_j": [pair[1] for pair in pairs_smi],
            }
        )

    if args.output_path is None:
        base_input_path = os.path.basename(args.data_path)
        args.output_path = os.path.join(
            DATA_PATH, f'{base_input_path.split(".")[0]}_k{cluster_param}_pairs.csv'
        )
    df_pairs.to_csv(args.output_path, index=False)
