import torch


class ForwardKinematics:
    def __init__(self, parents, offsets=None):
        self.parents = parents
        if offsets is not None and len(offsets.shape) == 2:
            offsets = offsets.unsqueeze(0)
        self.offsets = offsets

    def forward(self, rots, offsets=None, global_pos=None):
        """
        Forward Kinematics: returns a per-bone transformation
        @param rots: local joint rotations (batch_size, bone_num, 3, 3)
        @param offsets: (batch_size, bone_num, 3) or None
        @param global_pos: global_position: (batch_size, 3) or keep it as in offsets (default)
        @return: (batch_szie, bone_num, 3, 4)
        """
        rots = rots.clone()
        if offsets is None:
            offsets = self.offsets.to(rots.device)
        if global_pos is None:
            global_pos = offsets[:, 0]

        pos = torch.zeros((rots.shape[0], rots.shape[1], 3), device=rots.device)
        rest_pos = torch.zeros_like(pos)
        res = torch.zeros((rots.shape[0], rots.shape[1], 3, 4), device=rots.device)

        pos[:, 0] = global_pos
        rest_pos[:, 0] = offsets[:, 0]

        for i, p in enumerate(self.parents):
            if i != 0:
                rots[:, i] = torch.matmul(rots[:, p], rots[:, i])
                pos[:, i] = torch.matmul(rots[:, p], offsets[:, i].unsqueeze(-1)).squeeze(-1) + pos[:, p]
                rest_pos[:, i] = rest_pos[:, p] + offsets[:, i]

            res[:, i, :3, :3] = rots[:, i]
            res[:, i, :, 3] = torch.matmul(rots[:, i], -rest_pos[:, i].unsqueeze(-1)).squeeze(-1) + pos[:, i]

        return res

    def accumulate(self, local_rots):
        """
        Get global joint rotation from local rotations
        @param local_rots: (batch_size, n_bone, 3, 3)
        @return: global_rotations
        """
        res = torch.empty_like(local_rots)
        for i, p in enumerate(self.parents):
            if i == 0:
                res[:, i] = local_rots[:, i]
            else:
                res[:, i] = torch.matmul(res[:, p], local_rots[:, i])
        return res

    def unaccumulate(self, global_rots):
        """
        Get local joint rotation from global rotations
        @param global_rots: (batch_size, n_bone, 3, 3)
        @return: local_rotations
        """
        res = torch.empty_like(global_rots)
        inv = torch.empty_like(global_rots)

        for i, p in enumerate(self.parents):
            if i == 0:
                inv[:, i] = global_rots[:, i].transpose(-2, -1)
                res[:, i] = global_rots[:, i]
                continue
            res[:, i] = torch.matmul(inv[:, p], global_rots[:, i])
            inv[:, i] = torch.matmul(res[:, i].transpose(-2, -1), inv[:, p])

        return res
