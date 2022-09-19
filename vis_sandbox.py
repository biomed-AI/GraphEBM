# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Quantiative and qualtitative metrics of performance on rotamer recovery.
Additional scripts for visualization are in scripts/

Several of the visualizations are helper functions that can easily be
imported into a jupyter notebook for further analysis.
"""

import argparse
import itertools
import os
import os.path as osp
import pickle
import random
import numpy as np
import torch_geometric
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group

import torch
from dataset.amino_acid_config import kvs
from dataset.config import MMCIF_PATH
# from constants import atom_names, residue_names
from dataset import MMCIFDataset, collate_fn_transformer_test
from easydict import EasyDict
from dataset.mmcif_utils import (
    compute_dihedral,
    compute_rotamer_score_planar,
    exhaustive_sample,
    interpolated_sample_normal,
    load_rotamor_library,
    mixture_sample_normal,
    parse_dense_format,
    reencode_dense_format,
    rotate_dihedral_fast,
)
from model.ebm import RotomerDimeNet, RotomerLdq, RotomerTransformer, DimeNetPlus, DimeNetPlusPlus, DimeNetPlusPlusGraph
from torch import nn
from tqdm import tqdm

from utils.GPU_Manager import GPUManager, set_rand_seed, setCpu

from train import rebatch


def add_args(parser):
    parser.add_argument(
            "--prot",
            default=False,
            type=bool,
            help="using ProtTrans Features or not.(default: False)"
    )
    parser.add_argument(
            "--logdir",
            default="cachedir",
            type=str,
            help="location where log of experiments will be stored",
    )
    parser.add_argument(
            "--no-cuda", default=False, action="store_true", help="do not use GPUs for computations"
    )
    parser.add_argument(
            "--exp",
            default="transformer_gmm_uniform",
            type=str,
            help="name of experiment to run" "for pretrained model in the code, exp can be pretrained",
    )
    parser.add_argument(
            "--resume-iter",
            default=130000,
            type=int,
            help="list the iteration from which to continue training",
    )
    parser.add_argument(
            "--task",
            default="pair_atom",
            type=str,
            help="use a series of different tasks"
                 "pair_atom for measuring differences between atoms"
                 "sequence_recover for sequence recovery "
                 "pack_rotamer for sequence recovery "
                 "rotamer_trials for rotamer trials ",
    )
    parser.add_argument(
            "--pdb-name", default="6mdw.0", type=str, help="pdb on which to run analysis"
    )
    parser.add_argument(
            "--outdir", default="rotation_energies", type=str, help="output for experiments"
    )

    #############################
    ##### Analysis hyperparameters
    #############################

    parser.add_argument(
            "--rotations", default=1, type=int, help="number of rotations to use when evaluating"
    )
    parser.add_argument(
            "--sample-mode", default="rosetta", type=str, help="gmm or weighted_gauss or rosetta"
    )
    parser.add_argument(
            "--ensemble", default=1, type=int, help="number of ensembles to use for evaluation"
    )
    parser.add_argument(
            "--neg-sample",
            default=500,
            type=int,
            help="number of negative rotamer samples for rotamer trials (1-1 ratio)",
    )
    return parser


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]


def rotamer_trials(model, FLAGS, test_dataset):
    rotamer_scores_total = []
    rotamer_rmsd_total = []
    surface_scores_total = []
    surface_rmsd_total = []
    buried_scores_total = []
    buried_rmsd_total = []
    neutral_scores_total = []
    neutral_rmsd_total = []
    amino_recovery_total = {}
    for k, v in kvs.items():
        amino_recovery_total[k.lower()] = []
    amino_rmsd_total = {}
    for k, v in kvs.items():
        amino_rmsd_total[k.lower()] = []

    counter = 0
    rotations = FLAGS.rotations
    FLAGS.data_workers = 8
    FLAGS.batch_size = FLAGS.batch_size

    test_dataloader = DataLoader(
            test_dataset,
            num_workers=0,
            collate_fn=collate_fn_transformer_test,
            batch_size=1,
            # prefetch_factor=2,
            shuffle=False,
            pin_memory=False,
    )
    index = -1
    for node_pos, node_neg, gt_chis, neg_chis, res, types in tqdm(test_dataloader):
        index += 1
        name = test_dataset.names[index]
        gt_chis, neg_chis, res, types = gt_chis[0], neg_chis[0], res[0], types[0]
        amino_recovery = {}
        for k, v in kvs.items():
            amino_recovery[k.lower()] = []
        amino_rmsd = {}
        for k, v in kvs.items():
            amino_rmsd[k.lower()] = []
        rotamer_scores = []
        rotamer_rmsd = []
        surface_scores = []
        surface_rmsd = []
        buried_scores = []
        buried_rmsd = []
        neutral_scores = []
        neutral_rmsd = []

        n_entries = node_pos.num_graphs
        old_node_neg = node_neg

        if node_neg.num_graphs >= FLAGS.batch_size:
            node_neg = rebatch(node_neg, FLAGS)

        with torch.no_grad():
            if type(node_neg) == torch_geometric.data.batch.DataBatch:
                energy_neg = model.forward(node_neg.to(device).to("cpu"))
            elif type(node_neg) == list:
                energy_neg = []
                for neg in node_neg:
                    energy_neg.append(model.forward(neg.to(device)).to("cpu"))
                energy_neg = torch.cat(energy_neg)
            else:
                raise NotImplementedError

        energy = energy_neg
        energy = energy.view(n_entries, -1, rotations).mean(dim=2).nan_to_num(float('inf'))
        select_idx = torch.argmin(energy, dim=1).numpy()

        chis_pred = []
        for i in range(n_entries):
            neg_emb = old_node_neg.get_example(i * energy.size(1) + select_idx[i])
            pos_emb = node_pos.get_example(i)
            select_idx_i = select_idx[i]
            valid_chi_idx = gt_chis[i][0][1]
            rotamer_score, _ = compute_rotamer_score_planar(
                    gt_chis[i][0][0], neg_chis[i][select_idx_i][0], valid_chi_idx[:4], res[i], delta_angle=20
            )
            chis_pred.append([gt_chis[i][0][0], neg_chis[i][select_idx_i][0], valid_chi_idx[:4], res[i]])
            rmsd = (torch.square(pos_emb.pos[pos_emb.x[:, 3] == 1] - neg_emb.pos[neg_emb.x[:, 3] == 1]).sum() /
                    pos_emb.pos[pos_emb.x[:, 3] == 1].size(0)).numpy()
            rotamer_scores.append(rotamer_score)
            rotamer_rmsd.append(rmsd)

            amino_recovery[str(res[i])] = amino_recovery[str(res[i])] + [rotamer_score]
            amino_rmsd[str(res[i])] = amino_rmsd[str(res[i])] + [rmsd]

            if types[i] == "buried":
                buried_scores.append(rotamer_score)
                buried_rmsd.append(rmsd)
            elif types[i] == "surface":
                surface_scores.append(rotamer_score)
                surface_rmsd.append(rmsd)
            elif types[i] == "neutral":
                neutral_scores.append(rotamer_score)
                neutral_rmsd.append(rmsd)
        # torch.save(chis_pred, "DimeNet_every_result/{}/{}.pt".format(FLAGS.sample_mode, name))

        rotamer_scores_total.extend(rotamer_scores)
        rotamer_rmsd_total.extend(rotamer_rmsd)
        buried_scores_total.extend(buried_scores)
        buried_rmsd_total.extend(buried_rmsd)
        surface_scores_total.extend(surface_scores)
        surface_rmsd_total.extend(surface_rmsd)
        neutral_scores_total.extend(neutral_scores)
        neutral_rmsd_total.extend(neutral_rmsd)

        for k, v in amino_recovery.items():
            if len(v) > 0:
                amino_recovery_total[k] = amino_recovery_total[k] + v
        for k, v in amino_rmsd.items():
            if len(v) > 0:
                amino_rmsd_total[k] = amino_rmsd_total[k] + v

        print(
                "Obtained {} rotamer recovery score of {:.4f}({:.4f}) rotamer rmsd of {:.4f}({:.4f})".format(
                        len(rotamer_scores_total),
                        np.mean(rotamer_scores_total),
                        np.std(rotamer_scores_total) / len(rotamer_scores_total) ** 0.5,
                        np.mean(rotamer_rmsd_total),
                        np.std(rotamer_rmsd_total),
                )
        )
        print(
                "Obtained {} buried recovery score of {:.4f}({:.4f}) buried rmsd of {:.4f}({:.4f})".format(
                        len(buried_scores_total),
                        np.mean(buried_scores_total),
                        np.std(buried_scores_total) / len(buried_scores_total) ** 0.5,
                        np.mean(buried_rmsd_total),
                        np.std(buried_rmsd_total),
                )
        )
        print(
                "Obtained {} neutral recovery score of {:.4f}({:.4f}) neutral rmsd of {:.4f}({:.4f})".format(
                        len(neutral_scores_total),
                        np.mean(neutral_scores_total),
                        np.std(neutral_scores_total) / len(neutral_scores_total) ** 0.5,
                        np.mean(neutral_rmsd_total),
                        np.std(neutral_rmsd_total),
                )
        )
        print(
                "Obtained {} surface recovery score of {:.4f}({:.4f}) surface rmsd of {:.4f}({:.4f})".format(
                        len(surface_scores_total),
                        np.mean(surface_scores_total),
                        np.std(surface_scores_total) / len(surface_scores_total) ** 0.5,
                        np.mean(surface_rmsd_total),
                        np.std(surface_rmsd_total),
                )
        )
        for k, v in amino_recovery_total.items():
            print(
                    "{} amino acid recovery of {} score of {:.4f}({:.4f}) rmsd of {:.4f}({:.4f})".format(
                            len(v),
                            k,
                            np.mean(v),
                            np.std(v) / len(v) ** 0.5,
                            np.mean(amino_rmsd_total[k]),
                            np.std(amino_rmsd_total[k]),
                    )
            )


def rotamer_trials_transformer(model, FLAGS, test_dataset):
    test_files = test_dataset.files
    random.shuffle(test_files)
    db = load_rotamor_library()
    so3 = special_ortho_group(3)

    node_embed_evals = []
    nminibatch = 4

    if FLAGS.ensemble > 1:
        models = model

    rotamer_scores_total = []
    surface_scores_total = []
    buried_scores_total = []
    neutral_scores_total = []
    amino_recovery_total = {}
    for k, v in kvs.items():
        amino_recovery_total[k.lower()] = []

    counter = 0
    rotations = FLAGS.rotations

    for test_i, test_file in enumerate(test_files):
        (node_embed,) = pickle.load(open(test_file, "rb"))
        node_embed_original = node_embed
        par, child, pos, pos_exist, res, chis_valid = parse_dense_format(node_embed)
        angles = compute_dihedral(par, child, pos, pos_exist)

        amino_recovery = {}
        for k, v in kvs.items():
            amino_recovery[k.lower()] = []

        if node_embed is None:
            continue

        rotamer_scores = []
        surface_scores = []
        buried_scores = []
        neutral_scores = []
        types = []

        gt_chis = []
        node_embed_evals = []
        neg_chis = []
        valid_chi_idxs = []
        res_names = []

        neg_sample = FLAGS.neg_sample

        n_amino = pos.shape[0]
        amino_recovery_curr = {}
        for idx in tqdm(range(1, n_amino - 1), desc="test: {}/{}".format(test_i, len(test_files))):
            res_name = res[idx]
            if res_name == "gly" or res_name == "ala":
                continue

            res_names.append(res_name)

            gt_chis.append(angles[idx, 4:8])
            valid_chi_idxs.append(chis_valid[idx, :4])

            hacked_pos = np.copy(pos)
            swap_hacked_pos = np.swapaxes(hacked_pos, 0, 1)  # (20, 59, 3)
            idxs_to_change = swap_hacked_pos[4] == [0, 0, 0]  # (59, 3)
            swap_hacked_pos[4][idxs_to_change] = swap_hacked_pos[3][idxs_to_change]
            hacked_pos_final = np.swapaxes(swap_hacked_pos, 0, 1)

            neighbors = np.linalg.norm(pos[idx: idx + 1, 4] - hacked_pos_final[:, 4], axis=1) < 10
            neighbors = neighbors.astype(np.int32).sum()

            if neighbors >= 24:
                types.append("buried")
            elif neighbors < 16:
                types.append("surface")
            else:
                types.append("neutral")

            if neighbors >= 24:
                tresh = 0.98
            else:
                tresh = 0.95

            if FLAGS.sample_mode == "weighted_gauss":
                chis_list = interpolated_sample_normal(
                        db, angles[idx, 1], angles[idx, 2], res[idx], neg_sample, uniform=False
                )
            elif FLAGS.sample_mode == "gmm":
                chis_list = mixture_sample_normal(
                        db, angles[idx, 1], angles[idx, 2], res[idx], neg_sample, uniform=False
                )
            elif FLAGS.sample_mode == "rosetta":
                chis_list = exhaustive_sample(
                        db, angles[idx, 1], angles[idx, 2], res[idx], tresh=tresh
                )

            neg_chis.append(chis_list)

            node_neg_embeds = []
            length_chis = len(chis_list)
            for i in range(neg_sample):
                chis_target = angles[:, 4:8].copy()

                if i >= len(chis_list):
                    node_neg_embed_copy = node_neg_embed.copy()
                    node_neg_embeds.append(node_neg_embeds[i % length_chis])
                    neg_chis[-1].append(chis_list[i % length_chis])
                    continue

                chis = chis_list[i]

                chis_target[idx] = (
                        chis * chis_valid[idx, :4] + (1 - chis_valid[idx, :4]) * chis_target[idx]
                )
                pos_new = rotate_dihedral_fast(
                        angles, par, child, pos, pos_exist, chis_target, chis_valid, idx
                )

                node_neg_embed = reencode_dense_format(node_embed, pos_new, pos_exist)
                node_neg_embeds.append(node_neg_embed)

            node_neg_embeds = np.array(node_neg_embeds)
            dist = np.square(node_neg_embeds[:, :, -3:] - pos[idx: idx + 1, 4:5, :]).sum(axis=2)
            close_idx = np.argsort(dist)
            short_distance = np.sort(dist)
            closest_idx = close_idx[short_distance <= FLAGS.max_distance]
            node_neg_embeds = np.take_along_axis(node_neg_embeds, close_idx[:, :64, None], axis=1)
            node_neg_embeds[:, :, -3:] = node_neg_embeds[:, :, -3:] / 10.0
            node_neg_embeds[:, :, -3:] = node_neg_embeds[:, :, -3:] - np.mean(
                    node_neg_embeds[:, :, -3:], axis=1, keepdims=True
            )

            node_embed_evals.append(node_neg_embeds)

            if len(node_embed_evals) == nminibatch or idx == (n_amino - 2):
                n_entries = len(node_embed_evals)
                node_embed_evals = np.concatenate(node_embed_evals)
                s = node_embed_evals.shape

                # For sample rotations per batch
                node_embed_evals = np.tile(node_embed_evals[:, None, :, :], (1, rotations, 1, 1))
                rot_matrix = so3.rvs(rotations)

                if rotations == 1:
                    rot_matrix = rot_matrix[None, :, :]

                node_embed_evals[:, :, :, -3:] = np.matmul(
                        node_embed_evals[:, :, :, -3:], rot_matrix[None, :, :, :]
                )
                node_embed_evals = node_embed_evals.reshape((-1, *s[1:]))

                node_embed_feed = torch.from_numpy(node_embed_evals).float().to(device)

                with torch.no_grad():
                    energyList = []
                    dataList = []
                    for sample_i in range(node_embed_feed.shape[0]):
                        data = Data(x=node_embed_feed[sample_i][:, :3], pos=node_embed_feed[sample_i][:, 3:6] * 10)
                        dataList.append(data)
                        if ((sample_i + 1) % FLAGS.batch_size == 0) or ((sample_i + 1) == node_embed_feed.shape[0]):
                            if FLAGS.ensemble > 1:
                                energy = 0
                                for model in models:
                                    energy_tmp = model.forward(Batch().from_data_list(dataList))
                                    energy = energy + energy_tmp
                            else:
                                energy = model.forward(Batch().from_data_list(dataList))
                            energyList.append(energy)
                            dataList = []

                energy = torch.cat(energyList)
                energy = energy.view(n_entries, -1, rotations).mean(dim=2).nan_to_num(float('inf'))
                select_idx = torch.argmin(energy, dim=1).cpu().numpy()

                for i in range(n_entries):
                    select_idx_i = select_idx[i]
                    valid_chi_idx = valid_chi_idxs[i]
                    rotamer_score, _ = compute_rotamer_score_planar(
                            gt_chis[i], neg_chis[i][select_idx_i], valid_chi_idx[:4], res_names[i]
                    )
                    rotamer_scores.append(rotamer_score)

                    amino_recovery[str(res_names[i])] = amino_recovery[str(res_names[i])] + [
                        rotamer_score
                    ]

                    if types[i] == "buried":
                        buried_scores.append(rotamer_score)
                    elif types[i] == "surface":
                        surface_scores.append(rotamer_score)
                    elif types[i] == "neutral":
                        neutral_scores.append(rotamer_score)

                gt_chis = []
                node_embed_evals = []
                neg_chis = []
                valid_chi_idxs = []
                res_names = []
                types = []

            counter += 1

        rotamer_scores_total.extend(rotamer_scores)

        buried_scores_total.extend(buried_scores)
        surface_scores_total.extend(surface_scores)
        neutral_scores_total.extend(neutral_scores)

        for k, v in amino_recovery.items():
            if len(v) > 0:
                amino_recovery_total[k] = amino_recovery_total[k] + v

        print(
                "Obtained {} rotamer recovery score of ".format(len(rotamer_scores_total)),
                np.mean(rotamer_scores_total),
                np.std(rotamer_scores_total) / len(rotamer_scores_total) ** 0.5,
        )
        print(
                "Obtained {} buried recovery score of ".format(len(buried_scores_total)),
                np.mean(buried_scores_total),
                np.std(buried_scores_total) / len(buried_scores_total) ** 0.5,
        )
        print(
                "Obtained {} neutral recovery score of ".format(len(neutral_scores_total)),
                np.mean(neutral_scores_total),
                np.std(neutral_scores_total) / len(neutral_scores_total) ** 0.5,
        )
        print(
                "Obtained {} surface recovery score of ".format(len(surface_scores_total)),
                np.mean(surface_scores_total),
                np.std(surface_scores_total) / len(surface_scores_total) ** 0.5,
        )
        for k, v in amino_recovery_total.items():
            print(
                    "{} amino acid recovery of {} score of ".format(len(v), k),
                    np.mean(v),
                    np.std(v) / len(v) ** 0.5,
            )


def new_model(model, FLAGS, pdb_name):
    BATCH_SIZE = 128
    (node_embed,) = pickle.load(open(pdb_name, "rb"))
    par, child, pos, pos_exist, res, chis_valid = parse_dense_format(node_embed)
    angles = compute_dihedral(par, child, pos, pos_exist)

    parent = par.copy()
    childs = child.copy()
    edges = []
    idx_exist = 0
    last_atom = 0
    for i, residue in enumerate(pos_exist):
        atoms = sum(residue != 0) - 2
        parent[i][parent[i] == -18] = -last_atom
        childs[i][childs[i] == 18] = atoms
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

    chis_target_initial = angles[
                          :, 4:8
                          ].copy()  # dihedral for backbone (:4); dihedral for sidechain (4:8)

    NUM_RES = len(res)
    all_energies = np.zeros((NUM_RES, 4, 360), dtype=np.float32)  # 4 is number of possible chi angles

    surface_core_type = []
    for idx in range(NUM_RES):
        dist = np.sqrt(np.square(pos[idx: idx + 1, 2] - pos[:, 2]).sum(axis=1))
        neighbors = (dist < 10).sum()
        if neighbors >= 24:
            surface_core_type.append("core")
        elif neighbors <= 16:
            surface_core_type.append("surface")
        else:
            surface_core_type.append("unlabeled")

    for idx in tqdm(range(NUM_RES), desc=pdb_name.split("/")[-1]):
        for chi_num in range(4):
            if not chis_valid[idx, chi_num]:
                continue
            pos_chosen = pos[idx][pos_exist[idx] == 1]
            distance = np.square(node_embed[:, np.newaxis, 3:6] - pos_chosen[np.newaxis, :, :]).sum(axis=-1).min(
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

            exist_pos = (pos_exist[idx] == 1).sum()
            closest_edges = torch.LongTensor(closest_edges).T

            # init_angle = chis_target[idx, chi_num]
            for angle_deltas in batch(range(-180, 180, 3), BATCH_SIZE):
                pre_rot_node_embed_short = []
                for angle_delta in angle_deltas:
                    chis_target = chis_target_initial.copy()  # make a local copy

                    # modify the angle by angle_delta amount. rotate to chis_target
                    chis_target[
                        idx, chi_num
                    ] += angle_delta  # Set the specific chi angle to be the sampled value

                    # pos_new is n residues x 20 atoms x 3 (xyz)
                    pos_new = rotate_dihedral_fast(
                            angles, par, child, pos, pos_exist, chis_target, chis_valid, idx
                    )
                    node_neg_embed = reencode_dense_format(node_embed, pos_new, pos_exist)

                    # sort the atoms by how far away they are
                    # sort key is the first atom on the sidechain
                    # pos_chosen = pos_new[idx, 4]
                    # close_idx = np.argsort(
                    #         np.square(node_neg_embed[:, -3:] - pos_chosen).sum(axis=1)
                    # )

                    # Grab the 64 closest atoms
                    node_embed_short = node_neg_embed[closest_idx].copy()

                    # Normalize each coordinate of node_embed to have x, y, z coordinate to be equal 0
                    node_embed_short[:, -3:] = node_embed_short[:, -3:] - np.mean(
                            node_embed_short[:, -3:], axis=0
                    )
                    node_embed_short = torch.from_numpy(node_embed_short).float()

                    residue_index = torch.ones(exist_pos)

                    pos_residue_index = torch.zeros(node_embed_short.shape[0])
                    pos_residue_index[:exist_pos] = residue_index

                    pos_data = Data(x=torch.cat([node_embed_short[:, :3],
                                                 pos_residue_index.reshape(-1, 1)], dim=-1),
                                    pos=node_embed_short[:, 3:6],
                                    edge_index=closest_edges)
                    pre_rot_node_embed_short.append(pos_data.to(device))

                # Now rotate all elements
                # n_rotations = 100
                # so3 = special_ortho_group(3)
                # rot_matrix = so3.rvs(n_rotations)  # n x 3 x 3
                # node_embed_short = pre_rot_node_embed_short.repeat(1, n_rotations, 1, 1)
                # rot_matrix = torch.from_numpy(rot_matrix).float().cuda()
                # node_embed_short[:, :, :, -3:] = torch.matmul(
                #         node_embed_short[:, :, :, -3:], rot_matrix
                # )  # (batch_size, n_rotations, 64, 20)

                # Compute the energies for the n_rotations * batch_size for this window of 64 atoms.
                # Batch the first two dimensions, then pull them apart aftewrads.
                # node_embed_short = node_embed_short.reshape(
                #         node_embed_short.shape[0] * node_embed_short.shape[1],
                #         *node_embed_short.shape[2:],
                # )
                energies = model.forward(Batch().from_data_list(pre_rot_node_embed_short))  # (12000, 1)

                # divide the batch dimension by the 10 things we just did
                energies = energies.reshape(-1)  # (10, 200)
                # Average the energy across the n_rotations, but keeping batch-wise seperate
                # energies = energies.mean(1)  # (10, 1)

                # Save the result
                all_energies[idx, chi_num, angle_deltas] = energies.cpu().numpy()

    # Can use these for processing later.
    avg_chi_angle_energy = (all_energies * chis_valid[:NUM_RES, :4, None]).sum(0) / np.expand_dims(
            chis_valid[:NUM_RES, :4].sum(0), 1
    )  # normalize by how many times each chi angle occurs
    output = {
        "all_energies": all_energies[:, :, range(-180, 180, 3)],
        "chis_valid": chis_valid,
        "chis_target_initial": chis_target_initial,
        "avg_chi_angle_energy": avg_chi_angle_energy,  # make four plots from this (4, 360),
        "res": res,
        "surface_core_type": surface_core_type,
    }
    # Dump the output
    output_path = osp.join(FLAGS.outdir, "{}_rot_energies.p".format(pdb_name.split("/")[-1][:-2]))
    if not osp.exists(FLAGS.outdir):
        os.makedirs(FLAGS.outdir)
    pickle.dump(output, open(output_path, "wb"))


def main_single(FLAGS):
    FLAGS_OLD = FLAGS

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)

    if FLAGS.resume_iter != 0:
        model_path = osp.join(logdir, "model_{}".format(FLAGS.resume_iter))
        checkpoint = torch.load(model_path, map_location='cpu')
        try:
            FLAGS = checkpoint["FLAGS"]

            FLAGS.resume_iter = FLAGS_OLD.resume_iter
            FLAGS.neg_sample = FLAGS_OLD.neg_sample
            FLAGS.rotations = FLAGS_OLD.rotations

            for key in FLAGS.keys():
                if "__" not in key:
                    FLAGS_OLD[key] = getattr(FLAGS, key)

            FLAGS = FLAGS_OLD
            FLAGS.train = False
        except Exception as e:
            print(e)
            print("Didn't find keys in checkpoint'")

    models = []
    if FLAGS.ensemble > 1:
        for i in range(FLAGS.ensemble):
            # if FLAGS.model == "transformer":
            #     model = RotomerTransformerModel(FLAGS).eval()
            # elif FLAGS.model == "fc":
            #     model = RotomerFCModel(FLAGS).eval()
            # elif FLAGS.model == "s2s":
            #     model = RotomerSet2SetModel(FLAGS).eval()
            # elif FLAGS.model == "graph":
            #     model = RotomerGraphModel(FLAGS).eval()
            # elif FLAGS.model == "s2s":
            #     model = RotomerSet2SetModel(FLAGS).eval()
            if FLAGS.model == "GAT":
                model = RotomerLdq(8, 5.0, 4, 256, 8, 6).eval()
            elif FLAGS.model == "Transformer":
                model = RotomerTransformer(256, 8, 3).eval()
            elif FLAGS.model == "DimeNet":
                model = RotomerDimeNet(128, 1, 4, 4, 4, 4, cutoff=5.0, max_num_neighbors=64).eval()
            elif FLAGS.model == "DimeNetPlus":
                model = DimeNetPlus(128, 1, 4, 4, 4, 4, cutoff=5.0, max_num_neighbors=128).eval()
            elif FLAGS.model == "DimeNetPlusPlus":
                model = DimeNetPlusPlus(512, 1, 6, 256, 256, 256, 7, 7, cutoff=5.0, max_num_neighbors=128).eval()
            else:
                raise NotImplementedError
            models.append(model)
    else:
        # if FLAGS.model == "transformer":
        #     model = RotomerTransformerModel(FLAGS).eval()
        # elif FLAGS.model == "fc":
        #     model = RotomerFCModel(FLAGS).eval()
        # elif FLAGS.model == "s2s":
        #     model = RotomerSet2SetModel(FLAGS).eval()
        if FLAGS.model == "GAT":
            model = RotomerLdq(8, 5.0, 4, 256, 8, 6).eval()
        elif FLAGS.model == "Transformer":
            model = RotomerTransformer(256, 8, 3).eval()
        elif FLAGS.model == "DimeNet":
            model = RotomerDimeNet(128, 1, 4, 4, 4, 4, cutoff=5.0, max_num_neighbors=64).eval()
        elif FLAGS.model == "DimeNetPlus":
            model = DimeNetPlus(256, 1, 8, 256, 8, 4,
                                smooth_factor=FLAGS.smooth_factor,
                                cutoff=10.0, max_num_neighbors=256).eval()
        elif FLAGS.model == "DimeNetPlusPlus":
            model = DimeNetPlusPlus(256, 1, 8, 256, 256, 256, 8, 4,
                                    smooth_factor=FLAGS.smooth_factor,
                                    cutoff=10.0,
                                    max_num_neighbors=256).eval()
        elif FLAGS.model == "DimeNetPlusPlusGraph":
            model = DimeNetPlusPlusGraph(256, 1, 8, 256, 256, 256, 8, 4, smooth_factor=FLAGS.smooth_factor,
                                         heads=4, bond_channels=8, cutoff=10.0, max_num_neighbors=256).eval()
        # elif FLAGS.model == "DimeNetPlusPlusGraph":
        #     model = DimeNetPlusPlusGraph(512, 1, 4, 512, 512, 256, 9, 4,
        #                                  heads=8, bond_channels=32, cutoff=10.0, max_num_neighbors=256)
        else:
            raise NotImplementedError

    gpu = 0
    world_size = 0

    it = FLAGS.resume_iter

    if not osp.exists(logdir):
        os.makedirs(logdir)

    checkpoint = None

    if FLAGS.ensemble > 1:
        for i, model in enumerate(models):
            if FLAGS.resume_iter != 0:
                model_path = osp.join(logdir, "model_{}".format(FLAGS.resume_iter - i * 1000))
                checkpoint = torch.load(model_path)
                try:
                    model.load_state_dict(checkpoint["model_state_dict"])
                except Exception as e:
                    print("Transfer between distributed to non-distributed")

                    if world_size > 1:
                        model_state_dict = {
                            k.replace("module.", ""): v
                            for k, v in checkpoint["model_state_dict"].items()
                        }
                    else:
                        model_state_dict = {
                            k.replace("module.", ""): v
                            for k, v in checkpoint["model_state_dict"].items()
                        }
                    model.load_state_dict(model_state_dict)

            models[i] = model  # nn.DataParallel(model)
        model = models
    else:
        if FLAGS.resume_iter != 0:
            model_path = osp.join(logdir, "model_{}".format(FLAGS.resume_iter))
            checkpoint = torch.load(model_path, map_location='cpu')
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except Exception as e:
                print("Transfer between distributed to non-distributed")

                if world_size > 1:
                    model_state_dict = {
                        k.replace("module.", ""): v
                        for k, v in checkpoint["model_state_dict"].items()
                    }
                else:
                    model_state_dict = {
                        k.replace("module.", ""): v
                        for k, v in checkpoint["model_state_dict"].items()
                    }
                model.load_state_dict(model_state_dict)
            # model = nn.DataParallel(model)

    if FLAGS.cuda:
        if FLAGS.ensemble > 1:
            for i, model in enumerate(models):
                models[i] = model.to(device)

            model = models
        else:
            model = model.to(device)

    FLAGS.multisample = 1
    print("New Values of args: ", FLAGS)

    del checkpoint

    with torch.no_grad():
        if FLAGS.task == "rotamer_trial":
            test_dataset = MMCIFDataset(FLAGS,
                                        mmcif_path=MMCIF_PATH,
                                        split="test",
                                        uniform=False,
                                        gmm=True if FLAGS.sample_mode == "gmm" else False,
                                        )
            rotamer_trials(model, FLAGS, test_dataset)
        elif FLAGS.task == "rotamer_trial_tran":
            test_dataset = MMCIFDataset(FLAGS, mmcif_path=MMCIF_PATH, split="test")
            rotamer_trials_transformer(model, FLAGS, test_dataset)
        elif FLAGS.task == "new_model":
            test_dataset = MMCIFDataset(FLAGS, mmcif_path=MMCIF_PATH, split="test")
            for pdb_name in test_dataset.files:
                new_model(model, FLAGS, pdb_name)
        else:
            assert False


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    FLAGS = parser.parse_args()

    # convert to easy_dict; this is what is saved with model checkpoints and used in logic above
    keys = dir(FLAGS)
    flags_dict = EasyDict()
    for key in keys:
        if "__" not in key:
            flags_dict[key] = getattr(FLAGS, key)

    # postprocess arguments
    FLAGS.cuda = not FLAGS.no_cuda
    device = GPUManager().auto_choice()

    # set seeds
    set_rand_seed(1)
    setCpu(16)
    main_single(flags_dict)
