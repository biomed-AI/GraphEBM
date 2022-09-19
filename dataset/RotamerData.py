# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import pickle
import random

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import special_ortho_group
from torch_geometric.data import Data, Batch

import gemmi
import torch
from .constants import test_rotamers, name2symbol
from .math_utils import rotate_v1_v2
from .mmcif_utils import (
    compute_dihedral,
    exhaustive_sample,
    interpolated_sample_normal,
    load_rotamor_library,
    mixture_sample_normal,
    parse_dense_format,
    reencode_dense_format,
    rotate_dihedral_fast,
    parse_cif,
)
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
import gzip
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBIO, Select


def generate_PDB(seq_id, mmcif_file):
    cif_file = mmcif_file.split("/")
    cif_file[-1] = cif_file[-1].split(".")[0].lower() + ".cif.gz"
    parser = MMCIFParser()
    with gzip.open("/".join(cif_file), "rt") as f:
        structure = parser.get_structure(seq_id[:-1], f)

    parse_cif("/".join(cif_file))

    class ChainSelect(Select):
        def accept_chain(self, chain):
            if chain.get_id() == seq_id[-1]:
                return True
            else:
                return False

        def accept_residue(self, residue):

            if residue.id[0] == " " and not [False for atom in ["N", "CA", "C", "O"] if atom not in residue.child_dict]:
                return True
            else:
                return False

    io = PDBIO()
    io.set_structure(structure)
    io.save(f'./data/Rotamer/pdb/{seq_id}.pdb', ChainSelect())


def generate_BackBonePDB(seq_id, mmcif_file):
    cif_file = mmcif_file.split("/")
    cif_file[-1] = cif_file[-1].split(".")[0].lower() + ".cif.gz"
    parser = MMCIFParser()
    with gzip.open("/".join(cif_file), "rt") as f:
        structure = parser.get_structure(seq_id[:-1], f)

    class ChainSelect(Select):
        def accept_chain(self, chain):
            if chain.get_id() == seq_id[-1]:
                return True
            else:
                return False

        def accept_residue(self, residue):
            if residue.id[0] == " " and not [False for atom in ["N", "CA", "C", "O"] if atom not in residue.child_dict]:
                return True
            else:
                return False

        def accept_atom(self, atom):
            if atom.name in ["N", "CA", "C", "O"]:
                return True
            else:
                return False

    io = PDBIO()
    io.set_structure(structure)
    io.save(f'./data/Rotamer/pdb_backbone/{seq_id}_backbone.pdb', ChainSelect())



class MMCIFTransformer(Dataset):
    def __init__(
            self,
            FLAGS,
            mmcif_path="./data/Rotamer/mmcif",
            split="train",
            uniform=True,
            weighted_gauss=False,
            gmm=False,
            chi_mean=False,
            valid=False,
    ):
        files = []
        dirs = os.listdir(osp.join(mmcif_path, "mmCIF"))

        self.mmcif_path = mmcif_path
        self.split = split
        self.so3 = special_ortho_group(3)
        self.chi_mean = chi_mean
        self.weighted_gauss = weighted_gauss
        self.gmm = gmm
        self.uniform = uniform

        self.name2seq = {}
        name = None
        fasta_file = open("data/Rotamer/mmcif/cullpdb_pc90_res1.8_R0.25_d190807_chains14857.fasta", "r")
        for line in fasta_file.readlines():
            line = line.strip()
            if line == "":
                write = False
            elif line[0] == ">":
                write = False
                name = line.split(" ")[0][1:]
            else:
                write = True
            if write:
                seq = self.name2seq.setdefault(name, "") + line
                self.name2seq[name] = seq

        # Filter out proteins in test dataset
        for d in tqdm(dirs):
            directory = osp.join(mmcif_path, "mmCIF", d)
            d_files = os.listdir(directory)
            files_tmp = [osp.join(directory, d_file) for d_file in d_files if ".p" in d_file]

            for f in files_tmp:
                name = f.split("/")[-1]
                name = name.split(".")[0]
                if name in test_rotamers and self.split == "test":
                    files.append(f)
                elif name not in test_rotamers and self.split in ["train", "val"]:
                    files.append(f)

        self.files = files
        self.seqId = []

        self.alphabet = []
        for letter in range(65, 91):
            self.alphabet.append(chr(letter))

        if split in ["train", "val"]:
            duplicate_seqs = set()

            # Remove proteins in the train dataset that are too similar to the test dataset
            with open(osp.join(mmcif_path, "duplicate_sequences.txt"), "r") as f:
                for line in f:
                    duplicate_seqs.add(line.strip())

            fids = set()

            # Remove low resolution proteins
            with open(
                    osp.join(mmcif_path, "cullpdb_pc90_res1.8_R0.25_d190807_chains14857"), "r"
            ) as f:
                i = 0
                for line in f:
                    if i is not 0:
                        fid = line.split()[0]
                        if fid not in duplicate_seqs:
                            fids.add(fid)

                    i += 1

            files_new = []

            for f in files:
                tup = (f.split("/")[-1]).split(".")

                if int(tup[1]) >= len(self.alphabet):
                    continue

                seq_id = tup[0].upper() + self.alphabet[int(tup[1])]

                if seq_id in fids:
                    files_new.append(f)

            self.files = files_new
        elif split == "test":
            fids = set()

            # Remove low resolution proteins
            with open(
                    osp.join(mmcif_path, "cullpdb_pc90_res1.8_R0.25_d190807_chains14857"), "r"
            ) as f:
                i = 0
                for line in f:
                    if i is not 0:
                        fid = line.split()[0]
                        fids.add(fid)

                    i += 1

            files_new = []

            for f in files:
                tup = (f.split("/")[-1]).split(".")

                if int(tup[1]) >= len(self.alphabet):
                    continue

                seq_id = tup[0].upper() + self.alphabet[int(tup[1])]

                if seq_id in fids:
                    self.seqId.append(seq_id)
                    files_new.append(f)
                    # if seq_id == "2H5CA":
                    #     generate_PDB(seq_id, f)
                    #     generate_BackBonePDB(seq_id, f)

            self.files = files_new

        self.FLAGS = FLAGS
        self.db = load_rotamor_library()
        print(f"Loaded {len(self.files)} files for {split} dataset split")

        self.split = split

        self.names = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index, forward=False):
        FLAGS = self.FLAGS

        if FLAGS.single and not forward:
            index = 0

        FLAGS = self.FLAGS
        pickle_file = self.files[index]
        pdb_id, chain_id, _ = pickle_file.split("/")[-1].split(".")
        name = pdb_id.upper() + self.alphabet[int(chain_id)]
        self.names.append(name)
        # node_embed: D x 6
        (node_embed,) = pickle.load(open(pickle_file, "rb"))
        node_embed_original = node_embed

        # Remove proteins with small numbers of atoms
        if node_embed.shape[0] < 20:
            return self.__getitem__((index + 1) % len(self.files), forward=True)

        # Remove invalid proteins
        if (
                node_embed.max(axis=0)[2] >= 21
                or node_embed.max(axis=0)[0] >= 20
                or node_embed.max(axis=0)[1] >= 5
        ):
            return self.__getitem__((index + 1) % len(self.files), forward=True)

        par, child, pos, pos_exist, res, chis_valid = parse_dense_format(node_embed)

        parent = par.copy()
        childs = child.copy()
        edges = []
        idx_exist = 0
        last_atom = 0
        idx2id = {}
        for i, residue in enumerate(pos_exist):
            atoms = sum(residue != 0) - 2
            parent[i][parent[i] == -18] = -last_atom
            childs[i][childs[i] == 18] = atoms
            idx2id[i] = list(range(idx_exist, idx_exist + atoms + 2))
            last_atom = atoms
            for j, idx in enumerate(residue):
                if idx == 0:
                    continue
                if childs[i][j] != 0:
                    edges.append((idx_exist, idx_exist + childs[i][j]))
                if parent[i][j] != 0:
                    edges.append((idx_exist + parent[i][j], idx_exist))
                idx_exist += 1
        parentDict = {}
        childDict = {}
        for p, c in edges:
            parentList = parentDict.setdefault(p, [])
            childList = childDict.setdefault(c, [])
            if c not in parentList:
                parentList.append(c)
            if p not in childList:
                childList.append(p)
        for c, p in edges:
            parentList = parentDict.setdefault(p, [])
            childList = childDict.setdefault(c, [])
            if c not in parentList:
                parentList.append(c)
            if p not in childList:
                childList.append(p)

        if par is None:
            return self.__getitem__((index + 1) % len(self.files), forward=True)

        if len(res) < 5:
            return self.__getitem__((index + 1) % len(self.files), forward=True)

        angles = compute_dihedral(par, child, pos, pos_exist)
        select_idxs = []
        if FLAGS.train:
            tries = 0
            perm = np.random.permutation(np.arange(1, len(res) - 1))
            while True:
                # Randomly sample an amino acid that are not the first and last amino acid
                idx = perm[tries]
                if res[idx] == "gly" or res[idx] == "ala":
                    idx = random.randint(1, len(res) - 2)
                else:
                    select_idxs.append(idx)

                    if len(select_idxs) == FLAGS.multisample:
                        break

                tries += 1

                if tries > 1000 or tries == perm.shape[0]:
                    return self.__getitem__((index + 1) % len(self.files), forward=True)
        else:
            for idx in range(1, len(res) - 1):
                if res[idx] == "gly" or res[idx] == "ala":
                    continue
                select_idxs.append(idx)

        pos_graph = []
        neg_graph = []
        select_gt_chis = []
        select_neg_chis = []
        select_res = []
        select_types = []

        for idx in select_idxs:
            neg_samples = []
            gt_chis = [(angles[idx, 4:8], chis_valid[idx, :4])]
            neg_chis = []

            # Choose number of negative samples
            if FLAGS.train and self.split in ["val", "test"]:
                neg_sample = 150
            else:
                neg_sample = FLAGS.neg_sample

            atom_idxs = []
            atoms_mask = []
            chis_valids = []
            ancestors = []

            if self.split == "test":
                dist = np.sqrt(np.square(pos[idx: idx + 1, 2] - pos[:, 2]).sum(axis=1))
                neighbors = (dist < 10).sum()

                if neighbors >= 24:
                    residue_tpye = "buried"
                elif neighbors < 16:
                    residue_tpye = "surface"
                else:
                    residue_tpye = "neutral"

                # Choose different tresholds of sampling dependent on whether an atom is dense
                # or not
                if neighbors < 24:
                    tresh = 0.95
                else:
                    tresh = 0.98

                if self.weighted_gauss:
                    chis_list = interpolated_sample_normal(
                            self.db,
                            angles[idx, 1],
                            angles[idx, 2],
                            res[idx],
                            neg_sample * 2,
                            uniform=self.uniform,
                    )
                elif self.gmm:
                    chis_list = mixture_sample_normal(
                            self.db,
                            angles[idx, 1],
                            angles[idx, 2],
                            res[idx],
                            neg_sample * 2,
                            uniform=self.uniform,
                    )
                else:
                    chis_list = exhaustive_sample(
                            self.db,
                            angles[idx, 1],
                            angles[idx, 2],
                            res[idx],
                            tresh=tresh,
                            chi_mean=self.chi_mean,
                    )
                    random.shuffle(chis_list)

            else:
                dist = np.sqrt(np.square(pos[idx: idx + 1, 2] - pos[:, 2]).sum(axis=1))
                neighbors = (dist < 10).sum()

                residue_tpye = "neutral"

                if neighbors < 24:
                    tresh = 1.0
                else:
                    tresh = 1.0

                if self.weighted_gauss:
                    chis_list = interpolated_sample_normal(
                            self.db,
                            angles[idx, 1],
                            angles[idx, 2],
                            res[idx],
                            30 if self.split == "train" else neg_sample * 2,
                            uniform=self.uniform,
                    )
                elif self.gmm:
                    chis_list = mixture_sample_normal(
                            self.db,
                            angles[idx, 1],
                            angles[idx, 2],
                            res[idx],
                            30 if self.split == "train" else neg_sample * 2,
                            uniform=self.uniform,
                    )
                else:
                    chis_list = exhaustive_sample(
                            self.db,
                            angles[idx, 1],
                            angles[idx, 2],
                            res[idx],
                            tresh=tresh,
                            chi_mean=self.chi_mean,
                    )
                    random.shuffle(chis_list)

            i = 0
            while i < neg_sample:
                chis_target = angles[:, 4:8].copy()
                if len(chis_list) == 0:
                    return self.__getitem__((index + 1) % len(self.files), forward=True)
                select_i = np.random.randint(len(chis_list))
                chis = chis_list[select_i]

                chis_target[idx] = (
                        chis * chis_valid[idx, :4] + (1 - chis_valid[idx, :4]) * chis_target[idx]
                )
                pos_new = rotate_dihedral_fast(
                        angles, par, child, pos, pos_exist, chis_target, chis_valid, idx
                )

                distMatrix = F.pdist(torch.from_numpy(pos_new[pos_exist == 1]))
                if torch.min(distMatrix) <= 1 and FLAGS.restrict and self.split == "train":
                    chis_list.pop(select_i)
                    continue
                i += 1
                node_neg_embed = reencode_dense_format(node_embed, pos_new, pos_exist)
                neg_samples.append(node_neg_embed)
                neg_chis.append((chis_target[idx], chis_valid[idx, :4]))
                nelem = pos_exist[:idx].sum()
                offset = pos_exist[idx].sum()
                mask = np.zeros(20)
                mask[:offset] = 1

                atom_idxs.append(
                        np.concatenate(
                                [np.arange(nelem, nelem + offset), np.ones(20 - offset) * (nelem)]
                        )
                )
                atoms_mask.append(mask)
                chis_valids.append(chis_valid[idx, :4].copy())
                ancestors.append(np.stack([par[idx], child[idx]], axis=0))

            node_embed_negative = np.array(neg_samples)

            pos_chosen = pos[idx][pos_exist[idx] == 1]

            # Choose the closest atoms to the chosen location:
            distance = np.square(node_embed[:, np.newaxis, 3:6] - pos_chosen[np.newaxis, :, :]).sum(axis=-1).min(
                    axis=-1)
            close_idx = np.argsort(distance)
            short_distance = np.sort(distance)
            closest_idx = close_idx[short_distance <= FLAGS.max_distance]
            node_embed_short = node_embed[closest_idx].copy()

            if not ((short_distance <= 1e-4).sum() == (pos_exist[idx] == 1).sum()):
                return self.__getitem__((index + 1) % len(self.files), forward=True)
            exist_pos = (pos_exist[idx] == 1).sum()

            pos_closest_edges = []
            atomIdSet = set(closest_idx)
            for atomId in closest_idx:
                atomParent = set(parentDict[atomId]) & atomIdSet
                atomChild = set(childDict[atomId]) & atomIdSet
                while len(atomParent):
                    p = atomParent.pop()
                    pos_closest_edges.append((np.where(closest_idx == atomId)[0].item(),
                                              np.where(closest_idx == p)[0].item()))
                while len(atomChild):
                    c = atomChild.pop()
                    pos_closest_edges.append((np.where(closest_idx == atomId)[0].item(),
                                              np.where(closest_idx == c)[0].item()))
            pos_closest_edges = torch.LongTensor(pos_closest_edges).T
            # Compute the corresponding indices for atom_idxs
            # Get the position of each index ik
            # pos_code = np.argsort(close_idx_neg, axis=1)
            # choose_idx = np.take_along_axis(pos_code, atom_idxs.astype(np.int32), axis=1)

            # if choose_idx.max() >= FLAGS.max_size:
            #     return self.__getitem__((index + 1) % len(self.files), forward=True)

            node_sample_negative = []
            node_sample_neg_edges = []

            for i in range(neg_sample):
                node_embed_neg = node_embed_negative[i]
                distance = np.square(node_embed_neg[:, np.newaxis, 3:6] - pos_chosen[np.newaxis, :, :]).sum(
                        axis=-1).min(
                        axis=-1)
                close_idx = np.argsort(distance)
                short_distance = np.sort(distance)
                closest_idx = close_idx[short_distance <= FLAGS.max_distance]
                closest_edges = []
                atomIdSet = set(closest_idx)
                for atomId in closest_idx:
                    atomParent = set(parentDict[atomId]) & atomIdSet
                    atomChild = set(childDict[atomId]) & atomIdSet
                    while len(atomParent):
                        p = atomParent.pop()
                        closest_edges.append((np.where(closest_idx == atomId)[0].item(),
                                              np.where(closest_idx == p)[0].item()))
                    while len(atomChild):
                        c = atomChild.pop()
                        closest_edges.append((np.where(closest_idx == atomId)[0].item(),
                                              np.where(closest_idx == c)[0].item()))
                closest_edges = torch.LongTensor(closest_edges).T
                node_sample_neg_edges.append(closest_edges)
                embed_neg = node_embed_negative[i, closest_idx]
                embed_neg[:, 3:6] = embed_neg[:, 3:6] - np.mean(
                        node_embed_short[:, -3:], axis=0
                )
                node_sample_negative.append(embed_neg)

            # Normalize each coordinate of node_embed to have x, y, z coordinate to be equal 0
            node_embed_short[:, -3:] = node_embed_short[:, -3:] - np.mean(
                    node_embed_short[:, -3:], axis=0
            )
            # node_embed_negative[:, :, -3:] = node_embed_negative[:, :, -3:] - np.mean(
            #         node_embed_negative[:, :, -3:], axis=1, keepdims=True
            # )

            if FLAGS.augment:
                # Now rotate all elements
                rot_matrix = self.so3.rvs(FLAGS.rotations)
                if FLAGS.rotations == 1:
                    node_embed_short[:, -3:] = np.matmul(node_embed_short[:, -3:], rot_matrix)

                    rot_matrix_neg = self.so3.rvs(len(node_sample_negative)).reshape(-1, 3, 3)
                    for i, embed_neg in enumerate(node_sample_negative):
                        node_sample_negative[i][:, 3:6] = np.matmul(embed_neg[:, 3:6], rot_matrix_neg[i])
                else:
                    for i, embed_neg in enumerate(node_sample_negative):
                        node_sample_negative[i] = np.expand_dims(node_sample_negative[i], 0).repeat(FLAGS.rotations,
                                                                                                    axis=0)
                        node_sample_negative[i][:, :, -3:] = np.matmul(node_sample_negative[i][:, :, -3:], rot_matrix)

            # # Additionally scale values to be in the same scale
            # node_embed_short[:, -3:] = node_embed_short[:, -3:] / 10.0
            # node_embed_negative[:, :, -3:] = node_embed_negative[:, :, -3:] / 10.0

            # Augment the data with random rotations
            node_embed_short = torch.from_numpy(node_embed_short).float()

            residue_index = torch.ones(exist_pos)

            pos_residue_index = torch.zeros(node_embed_short.shape[0])
            pos_residue_index[:exist_pos] = residue_index

            pos_data = Data(x=torch.cat([node_embed_short[:, :3],
                                         pos_residue_index.reshape(-1, 1)], dim=-1),
                            pos=node_embed_short[:, 3:6],
                            edge_index=pos_closest_edges)
            pos_graph.append(pos_data)
            select_gt_chis.append(gt_chis)
            select_neg_chis.append(neg_chis)
            select_res.append(res[idx])
            if self.split == "test":
                select_types.append(residue_tpye)

            for i in range(neg_sample):
                node_sample_negative[i] = torch.from_numpy(node_sample_negative[i]).float()
                neg_residue_index = torch.zeros(node_sample_negative[i].shape[0])
                neg_residue_index[:exist_pos] = residue_index
                neg_data = Data(x=torch.cat([node_sample_negative[i][:, :3],
                                             neg_residue_index.reshape(-1, 1)], dim=-1),
                                pos=node_sample_negative[i][:, 3:6],
                                edge_index=node_sample_neg_edges[i])
                # distMatrix = F.pdist(neg_data.pos)
                # # if self.split is "train":
                # if torch.min(distMatrix) <= 1:
                #     return self.__getitem__((index + 1) % len(self.files), forward=True)
                neg_graph.append(neg_data)
            if self.split in ["val", "test"] and FLAGS.train:
                return pos_graph, neg_graph, gt_chis, neg_chis, res[idx], residue_tpye
        if FLAGS.train:
            return pos_graph, neg_graph
        else:
            return pos_graph, neg_graph, select_gt_chis, select_neg_chis, select_res, select_types


def collate_fn_transformer(inp):
    node_embed, node_embed_neg = zip(*inp)
    node_embed, node_embed_neg = sum(node_embed, []), sum(node_embed_neg, [])
    return Batch.from_data_list(node_embed), Batch.from_data_list(node_embed_neg)


def collate_fn_transformer_test(inp):
    node_embed, node_embed_neg, gt_chis, neg_chis, res, types = zip(*inp)
    node_embed, node_embed_neg = sum(node_embed, []), sum(node_embed_neg, [])
    return Batch.from_data_list(node_embed), Batch.from_data_list(node_embed_neg), gt_chis, neg_chis, res, types
