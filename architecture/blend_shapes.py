import torch
import torch.nn as nn
from models.transforms import aa2quat, aa2mat
from models.networks import MLP
from os.path import join as pjoin
import os


class BlendShapesModel(nn.Module):
    def __init__(self, n_vert, n_joint, basis_per_joint,
                 weight=None, parent=None, basis_as_model=True, save_freq=500, save_path=None, device=None,
                 threshold=0.05):
        super(BlendShapesModel, self).__init__()
        self.epoch_count = 0

        self.n_vert = n_vert
        self.n_joint = n_joint
        self.basis_per_joint = basis_per_joint
        self.parent = parent
        self.save_path = save_path
        self.save_freq = save_freq
        self.device = device
        self.threshold = threshold

        if save_path is not None:
            os.makedirs(pjoin(save_path, 'model'), exist_ok=True)
            os.makedirs(pjoin(save_path, 'optimizer'), exist_ok=True)

        basis = torch.randn((6890, basis_per_joint, 3)) / 10000
        if basis_as_model:
            self.basis = nn.Parameter(basis)
        else:
            self.basis = basis

        coff_list = [9, 18, 32, basis_per_joint]
        self.coff_branch = nn.ModuleList()
        for i in range(n_joint):
            coff_branch = MLP(coff_list)
            self.coff_branch.append(coff_branch)

        if weight is not None:
            mask = torch.empty((n_vert, n_joint), dtype=torch.bool)
            for i in range(n_joint):
                p = parent[i + 1]
                x = i + 1
                threshold = self.threshold if i not in [19, 20] else 0.02
                mask[:, i] = (weight[:, x] > threshold) + (weight[:, p] > threshold)
            mask = mask.float()
            self.register_buffer('mask', mask)  # shape = (n_vert, n_bone)

    def set_mask(self, weight):
        self.n_vert = weight.shape[0]
        mask = torch.empty((weight.shape[0], weight.shape[1] - 1), dtype=torch.bool, device=weight.device)
        for i in range(weight.shape[1] - 1):
            p = self.parent[i + 1]
            x = i + 1
            threshold = self.threshold if i not in [19, 20] else 0.02
            # Larger control field of wrist joints (joint 19 and 20)
            mask[:, i] = (weight[:, x] > threshold) + (weight[:, p] > threshold)
            # A joint should affect the vertices associated with itself and it parent joint
        mask = mask.float()
        self.mask = mask

    def set_optimizer(self, lr=1e-3, optimizer=torch.optim.Adam):
        params = self.parameters()
        self.optimizer = optimizer(params, lr=lr)

    def get_coff(self, pose):
        """
        @return: (batch_size, n_vert, n_basis_per_bone)
        """
        batch_size = pose.shape[0]
        device = pose.device
        if len(pose.shape) == 2:
            pose_repr = aa2mat(pose.reshape(pose.shape[0], -1, 3))
        elif len(pose.shape) == 4:
            pose_repr = pose.reshape(batch_size, -1, 3, 3)
        else:
            raise Exception('Wrong input format')
        pose_repr = pose_repr[:, 1:]
        pose_repr = pose_repr.reshape(-1, 9)
        identical = torch.eye(3, device=device).reshape(-1)

        pose_repr = pose_repr - identical

        pose_repr = pose_repr.reshape(pose.shape[0], self.n_joint, -1)
        coff = []
        for i in range(pose_repr.shape[1]):
            coff.append(self.coff_branch[i](pose_repr[:, i]).unsqueeze(1))
        coff = torch.cat(coff, dim=1)

        return coff

    def forward(self, pose, basis=None, mem_eff=True, requires_per_joint_off=False):
        """
        Get per-vertex displacement
        @param mem_eff: Use a for loop to increase memory efficiency
        """
        coff = self.get_coff(pose)      # (batch_size, n_bone, n_basis)
        mask_full = self.mask.reshape(self.n_vert, self.n_joint, 1, 1)
        if basis is None:
            basis = self.basis
        basis = basis.reshape(self.n_vert, 1, self.basis_per_joint, 3)
        basis_full = basis * mask_full   # (n_vert, n_bone, n_basis, 3)
        basis_full = basis_full.reshape(1, self.n_vert, -1, 3)
        coff = coff.reshape(coff.shape[0], 1, -1, 1)
        if requires_per_joint_off:
            per_joint_off = coff * basis_full
            per_joint_off = (per_joint_off * per_joint_off).sum(dim=-1).mean(dim=1)
            per_joint_off = per_joint_off.reshape(per_joint_off.shape[0], -1, self.basis_per_joint)
            per_joint_off = per_joint_off.mean(dim=-1)
            per_joint_off = torch.cat((torch.zeros_like(per_joint_off[:, :1]), per_joint_off), dim=1)
            self.per_joint_off = per_joint_off
        if mem_eff:
            res = []
            for i in range(coff.shape[0]):
                res.append((coff[[i]] * basis_full).sum(dim=-2))
            res = torch.cat(res, dim=0)
        else:
            res = (coff * basis_full).sum(dim=-2)
        self.coff = coff
        self.basis_full = basis_full
        return res

    def epoch(self):
        self.epoch_count += 1

    def save_model(self, epoch=None):
        if epoch is None:
            epoch = self.epoch_count

        if epoch % self.save_freq == 0:
            torch.save(self.state_dict(), pjoin(self.save_path, 'model/%05d.pt' % epoch))
            torch.save(self.optimizer.state_dict(), pjoin(self.save_path, 'optimizer/%05d.pt' % epoch))

        torch.save(self.state_dict(), pjoin(self.save_path, 'model/latest.pt'))
        torch.save(self.optimizer.state_dict(), pjoin(self.save_path, 'optimizer/latest.pt'))

    def load_model(self, epoch=None):
        if epoch is None:
            epoch = self.epoch_count

        if isinstance(epoch, str):
            state_dict = torch.load(epoch, map_location=self.device)
            self.load_state_dict(state_dict)

        else:
            filename = ('%05d.pt' % epoch) if epoch != -1 else 'latest.pt'
            state_dict = torch.load(pjoin(self.save_path, f'model/{filename}'), map_location=self.device)
            self.load_state_dict(state_dict, strict=False)
