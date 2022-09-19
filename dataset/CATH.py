import glob
import json
import math
import torch
import torch_cluster

import numpy as np
import os.path as osp
import torch.nn.functional as F

from tqdm import tqdm
from torch.multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from torch.utils.data import Dataset
from torch_geometric.data import Data

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from model.GeoMolUtils import get_dihedral_pairs

_amino_acids = lambda x: {
    'ALA': 0,
    'ARG': 1,
    'ASN': 2,
    'ASP': 3,
    'CYS': 4,
    'GLU': 5,
    'GLN': 6,
    'GLY': 7,
    'HIS': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'MET': 12,
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'THR': 16,
    'TRP': 17,
    'TYR': 18,
    'VAL': 19
}.get(x, 20)


def getAngle(data):
    groupIds, vals = torch.unique(data.x[:, 1], return_counts=True)
    groups = torch.split_with_sizes(data.x, tuple(vals))
    for group in groups:
        visited = []


def _orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _sidechains(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def atomFeaturize(filePath):
    try:
        protein = torch.load(filePath)
    except FileNotFoundError:
        return None
    # Atom
    x = torch.stack([protein["nodes"],
                     protein["nodeFeatures"]["groupId"],
                     protein["nodeFeatures"]["type"],
                     protein["nodeFeatures"]["charge"]]).T.long()
    pos = protein["nodeFeatures"]["xyz"]
    edgeIndex = protein["edges"]
    edgeAttr = protein["edgeFeatures"]["type"].T.reshape(-1, 1)
    mask = protein["nodeFeatures"]["residue"].T.reshape(-1, 1)
    return Data(x=x, pos=pos, edge_index=edgeIndex.long(), edge_attr=edgeAttr, mask=mask,
                group=list(map(_amino_acids, protein["nodeFeatures"]["aminoType"])))


def proteinFeaturize(filePath):
    proPath, torPath = filePath
    try:
        protein = torch.load(proPath)
    except FileNotFoundError:
        return None
    torsion = torch.load(torPath.replace("raw", "torsion"))
    CaIndex = protein["nodeFeatures"]["residue"] == 2
    MainIndex = ~(protein["nodeFeatures"]["residue"] == 3)
    groupIndex = torch.unique(protein["nodeFeatures"]["groupId"]).long()
    X_ca = protein["nodeFeatures"]["xyz"][CaIndex].float()
    coords = protein["nodeFeatures"]["xyz"][MainIndex].float()
    dihedralAngle = torsion["matrix"][groupIndex][:, [torsion["dimNames"][1].index(_angle)
                                                      for _angle in ["phi", "psi", "omega"]]]
    dihedralAngle /= 180
    dihedralAngle = torch.nan_to_num(dihedralAngle, 0)
    dihedrals = torch.cat([torch.cos(dihedralAngle), torch.sin(dihedralAngle)], dim=1)
    orientations = _orientations(X_ca)
    sidechains = _sidechains(coords.reshape(-1, 4, 3))
    node_s = dihedrals.float()
    node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
    node_type = torch.Tensor(list(map(_amino_acids, protein["nodeFeatures"]["aminoType"])))
    assert node_s.shape[0] == X_ca.shape[0] and node_v.shape[0] == X_ca.shape[0]
    torsionAngle = torsion["matrix"][groupIndex][:, [torsion["dimNames"][1].index("chi{}".format(chi))
                                                     for chi in range(1, 6)]]
    torsionAngle /= 180
    mask = ~torch.isnan(torsionAngle)
    torsionAngle = torch.nan_to_num(torsionAngle, 0).float()
    return Data(x=X_ca, node_s=node_s, node_v=node_v, node_type=node_type,
                coords=coords, torsionAngle=torsionAngle, mask=mask,
                name=protein["name"], seq=protein["seq"])


class CATHDataset(Dataset):
    def __init__(self, root: str, split_path: str, mode: str, model: str, residueType=None, knn=30,
                 device=torch.device("cpu")):
        super().__init__()
        self.root = root
        self.split = json.load(open(split_path, "r"))[mode]
        self.residueType = residueType
        self.knn = knn
        self.model = model
        self.device = device
        self.samples = []
        self.process()
        if self.model == "GeoMol":
            self.num_node_features = self.samples[0].x.shape[-1]
            self.num_edge_features = self.samples[0].edge_attr.shape[-1]
            self.dihedral_pairs = []

    def process(self):
        if self.model == "GeoMol":
            filePaths = [osp.join(self.root, "{}.pt".format(file)) for file in self.split][:100]
            with ThreadPool(16) as p:
                datas = p.map_async(atomFeaturize, filePaths)
                p.close()
                p.join()
            proteins = list(filter(None, datas.get()))
            for protein in tqdm(proteins, desc="Sampling"):
                groupId = protein.x[:, 1]
                residue = protein.group
                for gId in torch.unique(groupId):
                    if not self.residueType and residue[gId] != self.residueType:
                        continue
                    res = groupId == gId
                print(protein)
        elif self.model == "GVP":
            filePaths = [(osp.join(self.root, "{}.pt".format(file)),
                          osp.join(self.root.replace("raw", "torsion"), "{}.pt".format(file[:4])))
                         for file in self.split]
            with ThreadPool(16) as p:
                datas = p.map_async(proteinFeaturize, filePaths)
                p.close()
                p.join()
            self.samples = list(filter(None, datas.get()))

    def __getitem__(self, item):
        data = self.samples[item]
        if self.model == "GeoMol":
            if item not in self.dihedral_pairs:
                edge_index_dihedral_pairs = get_dihedral_pairs(data.edge_index, data)
                data.edge_index_dihedral_pairs = edge_index_dihedral_pairs
                self.dihedral_pairs.append(item)
        elif self.model == "GVP":
            ...
        return data

    def __len__(self):
        return len(self.samples)


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


class ProteinDataset(Dataset):
    def __init__(self, root: str, split_path: str, mode: str, model: str,
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16, device=torch.device("cpu"),
                 DEBUG=False,):
        super(ProteinDataset, self).__init__()
        self.root = root
        self.split = json.load(open(split_path, "r"))[mode]
        self.mode = mode
        self.model = model
        self.filePaths = [(osp.join(self.root, "{}.pt".format(file)),
                           osp.join(self.root.replace("raw", "torsion"), "{}.pt".format(file[:4])))
                          for file in self.split]
        if DEBUG:
            self.filePaths = self.filePaths[:100]
        self.samples = []
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                              'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                              'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                              'N': 2, 'Y': 18, 'M': 12, '?': 20}
        self.process()

    def process(self):
        for file in tqdm(self.filePaths, desc="Protein {} data".format(self.mode)):
            data = proteinFeaturize(file)
            if not data:
                continue
            with torch.no_grad():
                data.to(self.device)
                X_ca = data.x
                edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
                pos_embeddings = self._positional_embeddings(edge_index)
                E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
                rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

                edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
                edge_v = _normalize(E_vectors).unsqueeze(-2)

                edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

                data.edge_index = edge_index
                data.edge_s = edge_s
                data.edge_v = edge_v

                seq = torch.as_tensor([self.letter_to_num[a] for a in data.seq],
                                      device=self.device, dtype=torch.long)
                data.seq = seq
            self.samples.append(data)

    def __len__(self): return len(self.samples)

    def __getitem__(self, i): return self.samples[i]

    def _positional_embeddings(self, edge_index,
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
                torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
                * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E


if __name__ == "__main__":
    # dataset = ProteinDataset("../data/CATH/raw", "../data/CATH/chain_set_splits.json", "train", "GVP")
    dataset = CATHDataset("../data/CATH/raw", "../data/CATH/chain_set_splits.json", "train", "GeoMol")
    print()
