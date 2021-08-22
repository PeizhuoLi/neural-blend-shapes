import random
import torch
from torch.utils.data.dataset import Dataset
from dataset.obj_io import write_obj
from os.path import join as pjoin
import os
from dataset.topology_loader import TopologyLoader
import numpy as np
from dataset.smpl import SMPL_Layer
from models.deformation import deform_with_offset
from models.kinematics import ForwardKinematics
from models.transforms import aa2mat


parent_smpl = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
# SMPL skeleton topology


def generate_pose(batch_size, device, uniform=False, factor=1, root_rot=False, n_bone=None, ee=None):
    if n_bone is None: n_bone = 24
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
                # self._end_effectors.add(7)
                # self._end_effectors.add(8)
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


class SMPLDataset(BaseSkinnedDataset):
    def __init__(self, device, prefix=None):
        super(SMPLDataset, self).__init__(device)
        if prefix is not None:
            self.smpl_layer = SMPL_Layer(model_root=prefix).to(device)
        else:
            self.smpl_layer = SMPL_Layer().to(device)
        self.parents = self.smpl_layer.kintree_parents
        self.bone_num = len(self.parents)

    def forward(self, pose, pose2=None, shape=None, requires_skeleton=False):
        if shape is None:
            shape = generate_shape(pose.shape[0], device=self.device)
        t_pose = self.smpl_layer.forward(torch.zeros_like(pose), shape)[0]
        deformed = self.smpl_layer.forward(pose, shape)[0]
        offsets = self.smpl_layer.get_offset(shape)
        if not requires_skeleton:
            root_loc = offsets[:, 0, :]
        else:
            root_loc = offsets
        if pose2 is None:
            return deformed, t_pose, root_loc
        else:
            deformed2 = self.smpl_layer.forward(pose2, shape)[0]
            return deformed, deformed2, t_pose, root_loc

    def forward_multipose(self, pose, n_shape, requires_skeleton=False, residual=True):
        shape = generate_shape(n_shape, device=self.device)
        t_pose = self.smpl_layer.forward(torch.zeros((shape.shape[0], 72), device=self.device), shape)[0]
        offsets = self.smpl_layer.get_offset(shape)
        if not requires_skeleton:
            root_loc = offsets[:, 0, :]
        else:
            root_loc = offsets
        deformed = []
        for i in range(n_shape):
            deformed.append(self.smpl_layer.forward(pose, shape[[i]].expand(pose.shape[0], -1))[0][None, :])
        deformed = torch.cat(deformed, dim=0)
        return deformed, t_pose, root_loc


class MultiGarmentDataset(BaseSkinnedDataset):
    def __init__(self, prefix, topo_loader: TopologyLoader, device, is_train=True, fk=None):
        super(MultiGarmentDataset, self).__init__(device)
        self.prefix = prefix
        # self.smpl_hires = SMPL_Layer(highRes=True).to(device)
        self.smpl = SMPL_Layer().to(device)
        self.parents = self.smpl.kintree_parents
        # self.faces_hires = self.smpl_hires.th_faces
        self.faces = self.smpl.faces
        self.bone_num = len(self.parents)
        lst = [f for f in os.listdir(prefix) if os.path.isdir(pjoin(prefix, f))]
        lst.sort()

        lst = lst[:80] if is_train else lst[80:]   # division on training and test

        self.t_pose_list = []
        self.offset_list = []
        # self.weight_hires = self.smpl_hires.th_weights.to(device)
        self.weight = self.smpl.weights.to(device)

        for name in lst:
            prefix2 = pjoin(prefix, name)
            t_pose = np.load(pjoin(prefix2, 't-pose.npy'))
            offset = np.load(pjoin(prefix2, 'offset.npy'))
            t_pose = torch.tensor(t_pose, device=device)
            offset = torch.tensor(offset, device=device)
            self.t_pose_list.append(t_pose.unsqueeze(0))
            self.offset_list.append(offset.unsqueeze(0))

        self.t_pose_list = torch.cat(self.t_pose_list, dim=0)
        self.offset_list = torch.cat(self.offset_list, dim=0)

        if fk is None:
            fk = ForwardKinematics(self.parents)
        self.fk = fk

    def forward(self, pose=None, pose2=None, shape_id=None, hiRes=False, residual=False, requires_skeleton=False):
        if hiRes:
            raise Exception("High resolution support is abandoned in MIT license's SMPL implementation")
        if pose is None:
            pose = torch.zeros((1, self.bone_num * 3), device=self.device)
        if shape_id is None:
            shape_id = [random.randint(0, len(self) - 1) for _ in range(pose.shape[0])]
        pose = pose.reshape(pose.shape[0], -1, 3)
        if pose2 is not None:
            pose2 = pose2.reshape(pose.shape[0], -1, 3)
        t_poses = self.t_pose_list[shape_id]
        offsets = self.offset_list[shape_id]
        if not requires_skeleton:
            root_loc = offsets[:, 0, :]
        else:
            root_loc = offsets
        if not hiRes:
            t_poses = t_poses[:, :self.smpl.num_verts]
            weight = self.weight
            smpl = self.smpl
        else:
            weight = self.weight_hires
            smpl = self.smpl_hires

        mat = self.fk.forward(aa2mat(pose), offsets)
        if pose2 is not None:
            mat2 = self.fk.forward(aa2mat(pose2), offsets)

        if residual:
            offset = smpl.pose_blendshapes(pose.reshape(pose.shape[0], -1))
            offset2 = smpl.pose_blendshapes(pose2.reshape(pose2.shape[0], -1)) if pose2 is not None else None
        else:
            offset = 0
            offset2 = 0

        res = deform_with_offset(t_poses, weight, mat, offset=offset)
        if pose2 is not None:
            res2 = deform_with_offset(t_poses, weight, mat2, offset=offset2)
            return res, res2, t_poses, root_loc
        else:
            return res, t_poses, root_loc

    def forward_multipose(self, pose, n_shape, residual=False, requires_skeleton=False):
        shape_id = [random.randint(0, len(self) - 1) for _ in range(n_shape)]
        pose = pose.reshape(pose.shape[0], -1, 3)
        t_poses = self.t_pose_list[shape_id]
        t_poses = t_poses[:, :self.smpl.num_verts]
        offsets = self.offset_list[shape_id]
        if not requires_skeleton:
            root_loc = offsets[:, 0, :]
        else:
            root_loc = offsets

        if residual:
            v_offset = self.smpl.pose_blendshapes(pose.reshape(pose.shape[0], -1))
        else:
            v_offset = 0

        deformed = []
        posemat = aa2mat(pose)
        for i in range(n_shape):
            mat = self.fk.forward(posemat, offsets[[i]])
            deformed.append(deform_with_offset(t_poses[[i]], self.weight, mat, offset=v_offset)[None, :])
        deformed = torch.cat(deformed, dim=0)
        return deformed, t_poses, root_loc

    def __len__(self):
        return self.t_pose_list.shape[0]
