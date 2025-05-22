import pandas as pd
import numpy as np
from multiprocessing import Pool
import dgl.backend as F
from dgl.data.utils import save_graphs
from dgllife.utils.io import pmap
from rdkit import Chem
from scipy import sparse as sp
import argparse
from loguru import logger

from src.data.featurizer import smiles_to_graph_tune
from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--path_length", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=24)
    args = parser.parse_args()
    return args


def preprocess_dataset(args):
    df = pd.read_csv(f"{args.data_path}/{args.dataset}/{args.dataset}.csv")
    cache_file_path = (
        f"{args.data_path}/{args.dataset}/{args.dataset}_{args.path_length}.pkl"
    )
    smiless = df.smiles.values.tolist()
    solvent_smiless = df.solvent.values.tolist()

    # Task names was specified here
    task_names = [args.dataset]

    logger.info("constructing graphs")
    graphs = pmap(
        smiles_to_graph_tune,
        smiless,
        max_length=args.path_length,
        n_virtual_nodes=3,
        n_jobs=args.n_jobs,
    )

    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)
        else:
            logger.warning(f"graph {i} is None, smiles: {smiless[i]}")

    _label_values = df[task_names].values
    labels = F.zerocopy_from_numpy(_label_values.astype(np.float32))[valid_ids]
    logger.info("saving graphs")
    save_graphs(cache_file_path, valid_graphs, labels={"labels": labels})

    logger.info("extracting fingerprints")
    FP_list = []
    for smiles in tqdm(smiless):
        mol = Chem.MolFromSmiles(smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    FP_arr = np.array(FP_list)
    FP_sp_mat = sp.csc_matrix(FP_arr)
    logger.info("saving fingerprints")
    sp.save_npz(f"{args.data_path}/{args.dataset}/rdkfp1-7_512.npz", FP_sp_mat)

    logger.info("extracting molecular descriptors")
    generator = RDKit2DNormalized()
    features_map = Pool(args.n_jobs).imap(generator.process, tqdm(smiless))
    arr = np.array(list(features_map))
    np.savez_compressed(
        f"{args.data_path}/{args.dataset}/molecular_descriptors.npz", md=arr[:, 1:]
    )

    logger.info("extracting solvent descriptors")
    generator = RDKit2DNormalized()
    solvent_features_map = Pool(args.n_jobs).imap(
        generator.process, tqdm(solvent_smiless)
    )
    solvent_arr = np.array(list(solvent_features_map))
    np.savez_compressed(
        f"{args.data_path}/{args.dataset}/solvent_descriptors.npz",
        sd=solvent_arr[:, 1:],
    )

    logger.info("extracting splits")
    train_idx = np.where(df["split"] == "train")[0]
    valid_idx = np.where(df["split"] == "valid")[0]
    test_idx = np.where(df["split"] == "test")[0]

    split_idx = np.array([train_idx, valid_idx, test_idx])
    np.save(f"{args.data_path}/{args.dataset}/splits.npy", split_idx)


if __name__ == "__main__":
    args = parse_args()
    preprocess_dataset(args)
