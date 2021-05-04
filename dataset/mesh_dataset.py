import torch
from torch.utils.data.dataset import Dataset
from dataset.obj_io import write_obj
from os.path import join as pjoin
import os
from dataset.topology_loader import TopologyLoader
import numpy as np
from dataset.smpl_layer.smpl_layer import SMPL_Layer
from models.deformation import deform_with_offset
from models.kinematics import ForwardKinematics
from models.transforms import aa2mat


def generate_pose(batch_size, device, uniform=False, factor=1, root_rot=False, n_bone=None, ee=None):
    if n_bone is None: n_bone = 72 // 3
    if ee is not None:
        if root_rot:
            ee.append(0)
        n_bone_ = n_bone
        n_bone = len(ee)
    axis = torch.randn((batch_size, n_bone, 3), device=device)
    axis /= axis.norm(dim=-1, keepdim=True)
    if uniform:
        angle = torch.rand((batch_size, n_bone, 1), device=device) * np.pi
    else:
        angle = torch.randn((batch_size, n_bone, 1), device=device) * np.pi / 6 * factor
        angle.clamp(-np.pi, np.pi)
    poses = axis * angle
    if ee is not None:
        res = torch.zeros((batch_size, n_bone_, 3), device=device)
        for i, id in enumerate(ee):
            res[:, id] = poses[:, i]
        poses = res
    poses = poses.reshape(batch_size, -1)
    if not root_rot:
        poses[..., :3] = 0
    return poses


def generate_shape(batch_size, device):
    bound = 4
    betas = torch.rand((batch_size, 10), device=device)
    betas = (betas - 0.5) * 2
    betas = betas * bound
    return torch.clamp(betas, -bound, bound)


class BaseSkinnedDataset:
    def __init__(self, device):
        self.device = device
        self.parents = None
        self._end_effectors = None
        self.faces = None
        self.bone_num = None

    def end_effectors(self, order=1):
        if self._end_effectors is None:
            self._end_effectors = set()
            if order == 0:
                self._end_effectors.add(7)
                self._end_effectors.add(8)
                self._end_effectors.add(10)
                self._end_effectors.add(11)
            if order < 0:
                order = -order
                self._end_effectors.add(7)
                self._end_effectors.add(8)
            child_cnt = [0] * self.bone_num
            for i, p in enumerate(self.parents):
                if i == 0:
                    continue
                child_cnt[p] += 1
            for i, cnt in enumerate(child_cnt):
                if cnt == 0:
                    p = i
                    for _ in range(order):
                        self._end_effectors.add(p)
                        p = self.parents[i]
            self._end_effectors = list(self._end_effectors)
        return self._end_effectors

    def save_verts_as_obj(self, verts, filename, face=None):
        if face is None:
            face = self.faces
        write_obj(filename, verts, face)


class StaticMeshes(Dataset):
    def __init__(self, filenames, topo_loader: TopologyLoader, weight_gt=None):
        self.t_poses = []
        self.topo_id = []
        self.faces = []
        for filename in filenames:
            self.topo_id.append(topo_loader.load_from_obj(filename))
            self.t_poses.append(topo_loader.t_poses[-1])
            self.faces.append(topo_loader.faces[-1])
        if weight_gt is None:
            weight_gt = torch.tensor([0., ])
        self.weight = weight_gt

    def __getitem__(self, item):
        item %= len(self)
        return self.t_poses[item], self.topo_id[item]

    def __len__(self):
        return len(self.topo_id)

    def save_verts_as_obj(self, verts, filename, face=None, idx=None):
        if idx is None: idx = 0
        if face is None:
            face = self.faces[idx]
        write_obj(filename, verts, face)


class MultiGarmentDataset(BaseSkinnedDataset):
    def __init__(self, prefix, topo_loader: TopologyLoader, device, is_train=True, fk=None):
        super(MultiGarmentDataset, self).__init__(device)
        self.prefix = prefix
        self.smpl_hires = SMPL_Layer(highRes=True).to(device)
        self.smpl = SMPL_Layer().to(device)
        self.parents = self.smpl.kintree_parents
        self.faces_hires = self.smpl_hires.th_faces
        self.faces = self.smpl.th_faces
        self.bone_num = len(self.parents)
        lst = [f for f in os.listdir(prefix) if os.path.isdir(pjoin(prefix, f))]
        lst.sort()

        lst = lst[:80] if is_train else lst[80:]

        self.t_pose_list = []
        self.offset_list = []
        self.weight_hires = self.smpl_hires.th_weights.to(device)
        self.weight = self.smpl.th_weights.to(device)

        self.cloth_all = np.load(pjoin(prefix, 'all_cloths.npy'))
        self.cloth_all = torch.tensor(self.cloth_all, device=device)

        for name in lst:
            prefix2 = pjoin(prefix, name)
            t_pose = np.load(pjoin(prefix2, 't-pose.npy'))
            offset = np.load(pjoin(prefix2, 'offset.npy'))
            t_pose = torch.tensor(t_pose, device=device)
            offset = torch.tensor(offset, device=device)
            self.t_pose_list.append(t_pose.unsqueeze(0))
            self.offset_list.append(offset.unsqueeze(0))

        high2o_mask = np.array([True] * self.smpl.num_verts +
                               [False] * (self.smpl_hires.num_verts - self.smpl.num_verts))
        self.topo_id_hires = topo_loader.load_from_obj(pjoin(prefix, 'high_res.obj'))
        self.topo_id = topo_loader.load_from_obj(pjoin(prefix, 'original.obj'))
        self.t_pose_list = torch.cat(self.t_pose_list, dim=0)
        self.offset_list = torch.cat(self.offset_list, dim=0)

        if fk is None:
            fk = ForwardKinematics(self.parents)
        self.fk = fk

    def forward(self, shape_id, poses=None, hiRes=False, v_offset=0, residual=False):
        if poses is None:
            poses = torch.zeros((1, self.bone_num * 3), device=self.device)
        poses = poses.reshape(poses.shape[0], -1, 3)
        t_poses = self.t_pose_list[shape_id]
        offsets = self.offset_list[shape_id]
        # offsets[:, 0] = 0
        if not hiRes:
            t_poses = t_poses[:, :self.smpl.num_verts]
            weight = self.weight
            smpl = self.smpl
        else:
            weight = self.weight_hires
            smpl = self.smpl_hires

        mat = self.fk.forward(aa2mat(poses), offsets)

        if residual:
            offset = smpl.pose_blendshapes(poses)
        else:
            offset = 0

        res = deform_with_offset(t_poses, weight, mat, offset=v_offset + offset)
        return res, t_poses

    def forward_with_shape(self, shape_id, shapes, poses=None, hiRes=False, residual=False):
        # todo: reduce the number of calling smpl.forward()
        if poses is None:
            poses = torch.zeros((1, self.bone_num * 3), device=self.device)
        if hiRes:
            smpl = self.smpl_hires
            weight = self.weight_hires
        else:
            smpl = self.smpl
            weight = self.weight
        t_poses, _ = smpl.forward(torch.zeros_like(poses), shapes)
        offsets = smpl.get_offset(shapes)
        cloths = self.cloth_all[shape_id, :smpl.num_verts]
        clothed_t_poses = t_poses + cloths

        mat = self.fk.forward(aa2mat(poses.reshape(poses.shape[0], -1, 3)), offsets)

        if residual:
            offset = smpl.pose_blendshapes(poses)
        else:
            offset = 0

        res = deform_with_offset(clothed_t_poses, weight, mat, offset=offset)
        return res, clothed_t_poses

    def __len__(self):
        return self.t_pose_list.shape[0]
