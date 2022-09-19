import copy
import json
import os
import pickle
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
from Bio.PDB.PDBParser import PDBParser

from dataset.constants import test_rotamers
from dataset.mmcif_utils import (
    compute_dihedral,
    load_rotamor_library,
    mixture_sample_normal,
    parse_dense_format,
    reencode_dense_format,
    rotate_dihedral_fast,
)

from model.ebm import DimeNetPlusPlusGraph

from utils.GPU_Manager import GPUManager

import warnings

warnings.filterwarnings('ignore')

device = GPUManager().auto_choice()


class test_dataset(Dataset):
    def __init__(
            self,
            neg_samples=200,
            mmcif_path="data/Rotamer/mmcif",
    ):
        files = []
        dirs = os.listdir(os.path.join(mmcif_path, "mmCIF"))

        self.split = "test"
        self.mmcif_path = mmcif_path
        self.neg_sample = neg_samples

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

        for d in tqdm(dirs):
            directory = os.path.join(mmcif_path, "mmCIF", d)
            d_files = os.listdir(directory)
            files_tmp = [os.path.join(directory, d_file) for d_file in d_files if ".p" in d_file]

            for f in files_tmp:
                name = f.split("/")[-1]
                name = name.split(".")[0]
                if name in test_rotamers:
                    files.append(f)

        self.files = files
        self.alphabet = []
        for letter in range(65, 91):
            self.alphabet.append(chr(letter))

        # Filter out proteins in test dataset
        fids = set()

        # Remove low resolution proteins
        with open(
                os.path.join(mmcif_path, "cullpdb_pc90_res1.8_R0.25_d190807_chains14857"), "r"
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
                files_new.append(f)
                # generate_pdb(seq_id, f)

        self.files = files_new
        self.db = load_rotamor_library()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        pickle_file = self.files[index]
        (node_embed,) = pickle.load(open(pickle_file, "rb"))

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

        edges = []
        for atomId in range(idx_exist):
            atomParent = parentDict[atomId]
            atomChild = childDict[atomId]
            while len(atomParent):
                p = atomParent.pop()
                edges.append((atomId, p))
            while len(atomChild):
                c = atomChild.pop()
                edges.append((atomId, c))
        edges = torch.LongTensor(edges).T

        angles = compute_dihedral(par, child, pos, pos_exist)

        select_types = []
        select_chis = []
        select_res = []
        select_idxs = []
        id2pos = {}
        id2chi = {}
        for idx in tqdm(range(len(res)), desc="protein {:3d}".format(index)):
            neg_pos = []
            neg_chis = []
            if res[idx] == "gly" or res[idx] == "ala":
                neg_pos.append(node_embed[idx2id[idx]][:, 3:])
                neg_chis.append((angles[:, 4:8][idx], chis_valid[idx, :4]))
                id2pos[idx] = torch.from_numpy(np.stack(neg_pos)).float()
                id2chi[idx] = torch.from_numpy(np.stack(neg_chis)).float()
                continue
            # Choose number of negative samples
            neg_sample = self.neg_sample

            dist = np.sqrt(np.square(pos[idx: idx + 1, 2] - pos[:, 2]).sum(axis=1))
            neighbors = (dist < 10).sum()

            if neighbors >= 24:
                residue_tpye = "buried"
            elif neighbors < 16:
                residue_tpye = "surface"
            else:
                residue_tpye = "neutral"

            select_idxs.append(idx)
            select_types.append(residue_tpye)
            select_chis.append(np.concatenate([angles[idx, 4:8], chis_valid[idx, :4]]))
            select_res.append(node_embed[idx, 0])

            chis_list = mixture_sample_normal(
                    self.db,
                    angles[idx, 1],
                    angles[idx, 2],
                    res[idx],
                    neg_sample * 2,
                    uniform=False,
            )

            i = 0
            while i < neg_sample:
                chis_target = angles[:, 4:8].copy()
                select_i = np.random.randint(len(chis_list))
                chis = chis_list[select_i]

                chis_target[idx] = (
                        chis * chis_valid[idx, :4] + (1 - chis_valid[idx, :4]) * chis_target[idx]
                )
                pos_new = rotate_dihedral_fast(
                        angles, par, child, pos, pos_exist, chis_target, chis_valid, idx
                )

                i += 1
                node_neg_embed = reencode_dense_format(node_embed, pos_new, pos_exist)
                neg_pos.append(node_neg_embed[idx2id[idx]][:, 3:])
                neg_chis.append((chis_target[idx], chis_valid[idx, :4]))
            id2pos[idx] = torch.from_numpy(np.stack(neg_pos)).float()
            id2chi[idx] = torch.from_numpy(np.stack(neg_chis)).float()
        # output_info
        type2id = ["buried", "neutral", "surface"]
        pdb_id, chain_id, _ = pickle_file.split("/")[-1].split(".")
        # output_file = "chi_info/{}{}_info.npy".format(pdb_id.upper(), self.alphabet[int(chain_id)])
        # output = []
        # for i, chi in enumerate(select_chis):
        #     chi_row = np.concatenate([
        #         np.array([select_idxs[i]]),
        #         np.array([select_res[i]]),
        #         np.array([type2id.index(select_types[i])]),
        #         chi,
        #     ])
        #     output.append(chi_row)
        # np.save(output_file, np.stack(output))
        p = PDBParser(PERMISSIVE=1)
        rebuilt_pdb = p.get_structure(pdb_id, "data/Rotamer/pdb_backbone/{}_backbone.rebuilt.pdb".format(
                pdb_id.upper()+self.alphabet[int(chain_id)])
                                      )
        md = rebuilt_pdb[0]
        ch = md.child_list[0]
        with open("data/Rotamer/pdb_selects/{}_select.json".format(pdb_id.upper()), "r", encoding="utf-8") as select_f:
            select_pdb = json.load(select_f)[chain_id]
        pos_rebuild = []
        for residue_i, re_res in enumerate(ch.child_list):
            if residue_i in select_pdb:
                res_pos = []
                for atom in re_res.child_list:
                    res_pos.append(atom.coord)
                pos_rebuild.append(torch.from_numpy(np.stack(res_pos)))
        graph = Data(
                x=torch.cat([torch.from_numpy(node_embed[:, :3]), torch.zeros(node_embed.shape[0], 1)], dim=-1).float(),
                # pos=pos_re,
                edge_index=edges,
        )
        try:
            for idx in id2pos:
                if id2pos[idx].shape[0] == 1:
                    pos_rebuild[idx] = id2pos[idx][0]
        except:
            return None, None, None, None, None, None
        return pdb_id.upper() + self.alphabet[int(chain_id)], graph, idx2id, id2pos, id2chi, pos_rebuild


def get_model(model_path):
    model = DimeNetPlusPlusGraph(256, 1, 8, 256, 256, 256, 8, 4, smooth_factor=0.75,
                                 heads=4, bond_channels=8, cutoff=10.0, max_num_neighbors=256).eval()
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def rebatch(data_list: list, batch_size, batch_function):
    split_index = list(range(0, len(data_list), batch_size))
    split_index.append(len(data_list))
    rebatch_data = []
    for i in range(len(split_index) - 1):
        datas = []
        for j in range(split_index[i], split_index[i + 1]):
            datas.append(data_list[j])
        rebatch_data.append(batch_function.from_data_list(datas))
    return rebatch_data


if __name__ == "__main__":
    batch = Batch()
    sample_num = 200
    batch_size = 4
    dataset = test_dataset(neg_samples=sample_num)
    model = get_model("cachedir/DimeNetPlusPlusGraph3Sample20DistSmooth0.75_retrain/model_24000").to(device)
    for name, graph, idx2id, id2pos, id2chi, pos_init in dataset:
        # continue
        if name != "1TUKA":
            print(name)
            continue
        output_file_name = "DimeNet_result/{}.npy".format(name)
        # if os.path.exists(output_file_name):
        #     continue
        old_graph = graph
        energy_min = (torch.ones(len(idx2id)) * torch.Tensor([np.inf]))
        energy_min_i = -torch.ones(len(idx2id)).to(device)
        energy_min_i_new = -torch.ones(len(idx2id)).to(device)
        energy_patience = (torch.ones(len(idx2id)) * torch.Tensor([2]))
        for idx in id2pos:
            if id2pos[idx].shape[0] == 1:
                energy_patience[idx] = 0
        it = 0
        while torch.sum(energy_patience) > 0:
            for idx in tqdm(idx2id, desc="{} iter".format(it)):
                if energy_patience[idx] == 0:
                    continue
                energy_tensor = []
                graph = copy.deepcopy(old_graph)
                graph.x[idx2id[idx], 3] = 1
                pos = torch.cat([id2pos[res_i][int(energy_min_i[res_i])] if energy_min_i[res_i] != -1
                                 else pos_init[res_i]
                                 for res_i in idx2id])
                data_list = []
                for sample_i in range(sample_num):
                    pos[idx2id[idx]] = id2pos[idx][sample_i]
                    min_dist, _ = torch.min(torch.cdist(pos, pos[idx2id[idx]]), dim=-1)
                    nodes = min_dist < 4.5
                    edge_index, _ = subgraph(nodes, graph.edge_index, relabel_nodes=True)
                    data_list.append(Data(
                            x=graph.x[nodes],
                            pos=pos[nodes],
                            edge_index=edge_index
                    ))
                for data in rebatch(data_list, batch_size, batch):
                    energy = model.forward(data.to(device))
                    energy_tensor.append(energy.detach().cpu().reshape(-1))
                    del data
                energies = torch.cat(energy_tensor)
                min_energy_i = torch.argsort(energies)[0]
                min_energy = energies[int(min_energy_i)]
                if min_energy < energy_min[idx]:
                    energy_min[idx] = min_energy
                    energy_min_i_new[idx] = min_energy_i
                    energy_patience[idx] = 2
                else:
                    energy_patience[idx] -= 1
            np.save("DimeNet_result/{}_{}.npy".format(name, it),
                    torch.cat([id2chi[res_i][int(energy_min_i[res_i])] for res_i in idx2id]).numpy())
            energy_min_i = energy_min_i_new
            print("{} iter: {} recovering {} stable".format(
                    it,
                    torch.sum(energy_patience > 0),
                    torch.sum(energy_patience == 0),
            ))
            it += 1
        chi_result = torch.cat([id2chi[res_i][int(energy_min_i[res_i])] for res_i in idx2id])
        np.save("DimeNet_result/{}_{}.npy".format(name, it),
                chi_result.numpy())
