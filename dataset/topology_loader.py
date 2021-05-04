import torch
from tqdm import tqdm
from meshcnn.models.layers.mesh import Mesh
import numpy as np
from os.path import join as pjoin
import os
from mesh.simple_mesh import SimpleMesh
from models.features import FeatureExtractor


class TopologyLoader:
    """
    This class is used to load and storage topology information (connectivity) from obj file
    """
    def __init__(self, device=None, debug=False):
        self.__upper_bound = 2 if debug else 50
        self.meshes = []
        self.v_masks = []
        self.edges = []
        self.t_poses = []
        self.faces = []
        # self.laplacians = []
        # self.extractors = []
        if device is None:
            device = torch.device('cpu')
        self.device = device

    def __len__(self):
        return len(self.meshes)

    def apply_mesh(self, verts, idx):
        mesh = self.meshes[idx]
        v_mask = self.v_masks[idx]
        if v_mask is not None:
            verts = verts[v_mask]
        new_mesh = mesh.copy_from_mesh(mesh, verts)
        return new_mesh

    def load_from_obj(self, obj_name, v_mask=None, noise_id=None):
        bmesh = SimpleMesh()
        bmesh.load(obj_name)
        mesh = Mesh(file=obj_name, hold_history=True)
        if noise_id is not None:
            offset = np.ones((3, )) * 1e-5
            for id in noise_id:
                bmesh.vs[id] += offset
                mesh.vs[id] += offset
        self.meshes.append(mesh)
        self.t_poses.append(torch.tensor(mesh.vs, dtype=torch.float, device=self.device))
        self.edges.append(torch.tensor(mesh.edges, dtype=torch.int64, device=self.device))
        self.faces.append(torch.tensor(bmesh.faces, dtype=torch.int64, device=self.device))
        # self.laplacians.append(bmesh.get_uniform_laplacian().to(self.device))
        self.v_masks.append(v_mask)
        # self.extractors.append(FeatureExtractor(mesh))

        return len(self.meshes) - 1

    def load_smpl_augment(self, prefix):
        self.load_from_obj(pjoin(prefix, 'T-pose.obj'))
        v_mask = np.load(pjoin(prefix, 'v_mask.npy'))
        self.v_masks[-1] = torch.from_numpy(v_mask).to(self.device)

    def load_smpl_group(self, prefix, is_train=True):
        __upper_bound = self.__upper_bound
        begin = len(self)
        l = sorted([d for d in os.listdir(prefix) if os.path.isdir(pjoin(prefix, d))])
        if len(l) > __upper_bound:
            l = l[:__upper_bound] if is_train else l[__upper_bound: __upper_bound + 10]
        print('Preparing topology augmentation...')
        for idx, name in enumerate(tqdm(l)):
            pre_2 = pjoin(prefix, name)
            self.load_smpl_augment(pre_2)
        return begin, len(l)
