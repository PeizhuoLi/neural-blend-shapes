import os

import numpy as np
import torch
from torch.nn import Module

from dataset.smpl_layer.serialization import ready_arguments
from dataset.smpl_layer.tensutils import (th_posemap_axisang, th_with_zeros, th_pack, make_list, subtract_flat_id)
from dataset.obj_io import write_obj

from models.kinematics import ForwardKinematics
from models.transforms import aa2mat
from models.deformation import deform_with_offset


class SMPL_Layer(Module):
    __constants__ = ['kintree_parents', 'gender', 'center_idx', 'num_joints']

    def __init__(self,
                 center_idx=None,
                 gender='neutral',
                 model_root='./dataset/smpl_model',
                 highRes=False):
        """
        Args:
            center_idx: index of center joint in our computations,
            model_root: path to pkl files for the model
            gender: 'neutral' (default) or 'female' or 'male'
        """
        super().__init__()

        self.center_idx = center_idx
        self.gender = gender

        if gender == 'neutral':
            self.model_path = os.path.join(model_root, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
        elif gender == 'female':
            self.model_path = os.path.join(model_root, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        elif gender == 'male':
            self.model_path = os.path.join(model_root, 'basicModel_m_lbs_10_207_0_v1.0.0.pkl')
        highRes_path = os.path.join(model_root, 'hresMapping.pkl') if highRes else None

        smpl_data = ready_arguments(self.model_path, highRes_path)
        self.smpl_data = smpl_data

        self.register_buffer('th_betas',
                             torch.Tensor(smpl_data['betas']).unsqueeze(0))
        self.register_buffer('th_shapedirs',
                             torch.Tensor(smpl_data['shapedirs'].copy()))
        self.register_buffer('th_posedirs',
                             torch.Tensor(smpl_data['posedirs']))
        self.register_buffer(
            'th_v_template',
            torch.Tensor(smpl_data['v_template']).unsqueeze(0))
        self.register_buffer(
            'th_J_regressor',
            torch.Tensor(np.array(smpl_data['J_regressor'].toarray())))
        self.register_buffer('th_weights',
                             torch.Tensor(smpl_data['weights']))
        self.register_buffer('th_faces',
                             torch.Tensor(smpl_data['f'].astype(np.int32)).long())

        # Kinematic chain params
        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents
        self.num_joints = len(parents)  # 24
        self.num_verts = self.th_v_template.shape[1]

        self.fk = ForwardKinematics(self.kintree_parents)

    def get_offset(self, shapes=torch.zeros((1, 10))):
        batch_size = shapes.shape[0]
        parent_smpl = self.kintree_parents
        t_pose, j_loc = self.forward(torch.zeros((batch_size, 24 * 3), device=shapes.device), shapes)
        for i in list(range(len(parent_smpl)))[::-1]:
            if i == 0:
                break
            p = parent_smpl[i]
            j_loc[:, i] -= j_loc[:, p]
        offset = j_loc
        return offset

    def forward(self,
                th_pose_axisang,
                th_betas=None,
                th_trans=torch.zeros(1),
                requires_transformation=False):
        """
        Args:
        th_pose_axisang (Tensor (batch_size x 72)): pose parameters in axis-angle representation
        th_betas (Tensor (batch_size x 10)): if provided, uses given shape parameters
        th_trans (Tensor (batch_size x 3)): if provided, applies trans to joints and vertices
        """

        batch_size = th_pose_axisang.shape[0]
        # Convert axis-angle representation to rotation matrix rep.
        th_pose_rotmat = th_posemap_axisang(th_pose_axisang)
        # Take out the first rotmat (global rotation)
        root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3)
        # Take out the remaining rotmats (23 joints)
        th_pose_rotmat = th_pose_rotmat[:, 9:]
        th_pose_map = subtract_flat_id(th_pose_rotmat)

        # Below does: v_shaped = v_template + shapedirs * betas
        # If shape parameters are not provided
        if th_betas is None:
            th_v_shaped = self.th_v_template + torch.matmul(
                self.th_shapedirs, self.th_betas.transpose(1, 0)).permute(2, 0, 1)
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(
                batch_size, 1, 1)
        else:
            th_v_shaped = self.th_v_template + torch.matmul(
                self.th_shapedirs, th_betas.transpose(1, 0)).permute(2, 0, 1)
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped)

        # Below does: v_posed = v_shaped + posedirs * pose_map
        th_v_posed = th_v_shaped + torch.matmul(
            self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        # Final T pose with transformation done!

        # Global rigid transformation
        th_results = []

        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

        # Rotate each part
        for i in range(self.num_joints - 1):
            i_val = int(i + 1)
            joint_rot = th_pose_rotmat[:, (i_val - 1) * 9:i_val *
                                   9].contiguous().view(batch_size, 3, 3)
            joint_j = th_j[:, i_val, :].contiguous().view(batch_size, 3, 1)
            parent = make_list(self.kintree_parents)[i_val]
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)
            joint_rel_transform = th_with_zeros(
                torch.cat([joint_rot, joint_j - parent_j], 2))
            th_results.append(
                torch.matmul(th_results[parent], joint_rel_transform))
        th_results_global = th_results

        th_results2 = torch.zeros((batch_size, 4, 4, self.num_joints),
                                  dtype=root_j.dtype,
                                  device=root_j.device)

        for i in range(self.num_joints):
            padd_zero = torch.zeros(1, dtype=th_j.dtype, device=th_j.device)
            joint_j = torch.cat(
                [th_j[:, i],
                 padd_zero.view(1, 1).repeat(batch_size, 1)], 1)
            tmp = torch.bmm(th_results[i], joint_j.unsqueeze(2))
            th_results2[:, :, :, i] = th_results[i] - th_pack(tmp)

        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))

        th_rest_shape_h = torch.cat([
            th_v_posed.transpose(2, 1),
            torch.ones((batch_size, 1, th_v_posed.shape[1]),
                       dtype=th_T.dtype,
                       device=th_T.device),
        ], 1)

        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = torch.stack(th_results_global, dim=1)[:, :, :3, 3]

        # If translation is not provided
        if th_trans is None or bool(torch.norm(th_trans) == 0):
            if self.center_idx is not None:
                center_joint = th_jtr[:, self.center_idx].unsqueeze(1)
                th_jtr = th_jtr - center_joint
                th_verts = th_verts - center_joint
        else:
            th_jtr = th_jtr + th_trans.unsqueeze(1)
            th_verts = th_verts + th_trans.unsqueeze(1)

        # Vertices and joints in meters
        if requires_transformation:
            return th_verts, th_jtr, th_results2
        else:
            return th_verts, th_jtr

    def forward_lbs(self, poses, shapes=None, v_offsets=0):
        if shapes is None:
            shapes = torch.zeros((poses.shape[0], 10), device=poses.device)
        t_pose = self.forward(torch.zeros_like(poses), shapes)[0]
        offsets = self.get_offset(shapes)
        local_mat = aa2mat(poses.reshape(poses.shape[0], -1, 3))
        global_mat = self.fk.forward(local_mat, offsets)
        return deform_with_offset(t_pose, self.th_weights, global_mat, offset=v_offsets)

    def save_obj(self, filename, verts):
        write_obj(filename, verts, self.th_faces)

    def pose_blendshapes(self, th_pose_axisang):
        batch_size = th_pose_axisang.shape[0]
        # Convert axis-angle representation to rotation matrix rep.
        th_pose_rotmat = th_posemap_axisang(th_pose_axisang)
        # Take out the first rotmat (global rotation)
        root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3)
        # Take out the remaining rotmats (23 joints)
        th_pose_rotmat = th_pose_rotmat[:, 9:]
        th_pose_map = subtract_flat_id(th_pose_rotmat)

        return torch.matmul(self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
