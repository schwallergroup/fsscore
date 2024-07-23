import argparse
from pathlib import Path

import pandas as pd
import torch

from fsscore.models.ranknet import LitRankNet
from fsscore.score import Scorer
from fsscore.utils.clustering_mols import ClusterMols
from fsscore.utils.paths import PRETRAIN_MODEL_PATH


def pair_molecules(filepath, smi_col="smiles"):
    df = pd.read_csv(filepath)

    assert smi_col in df.columns, f"File requires {smi_col} as column name."

    smiles = df[smi_col].values.tolist()

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


def rank_by_uncertainty(
    df, output_path, featurizer="graph_2D", batch_size=32, ckpt_path=PRETRAIN_MODEL_PATH
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LitRankNet.load_from_checkpoint(ckpt_path, map_location=device)

    scorer = Scorer(
        model=model,
        featurizer=featurizer,
        batch_size=batch_size,
        graph_datapath=None,
        mc_dropout_samples=100,
        dropout_p=0.2,
        keep_graphs=False,
        num_workers=0,
        device=device,
    )

    smiles = df[["smiles_i", "smiles_j"]].values.tolist()
    scores_mean, scores_var = scorer.score(smiles=smiles)

    df["score_diff_mean"] = scores_mean
    df["score_diff_var"] = scores_var

    df = df.sort_values(by="score_diff_var", ascending=False)
    df.to_csv(output_path, index=False)


parser = argparse.ArgumentParser(description="Pair molecules and rank by uncertainty.")
parser.add_argument(
    "--filepath", type=str, help="Path to input csv file.", required=True
)
parser.add_argument(
    "--outdir", type=str, help="Path to output directory.", required=True
)
parser.add_argument(
    "--featurizer", type=str, default="graph_2D", help="Featurizer to use."
)
parser.add_argument(
    "--smi_col", type=str, default="smiles", help="Column name for SMILES strings."
)
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument(
    "--ckpt_path",
    type=str,
    default=PRETRAIN_MODEL_PATH,
    help="Path to pre-trained model checkpoint.",
)

args = parser.parse_args()

assert Path(args.filepath).exists(), "File does not exist."
Path(args.outdir).mkdir(parents=True, exist_ok=True)

print("Clustering and pairing molecules...")
df_pairs = pair_molecules(args.filepath, smi_col=args.smi_col)

out_csvfilename = Path(args.filepath).stem + "_pairs.csv"
out_csvpath = Path(args.outdir) / out_csvfilename

print("Ranking by uncertainty...")
rank_by_uncertainty(
    df_pairs,
    output_path=out_csvpath,
    featurizer=args.featurizer,
    batch_size=args.batch_size,
    ckpt_path=args.ckpt_path,
)
print(f"Output saved to {out_csvpath}")
