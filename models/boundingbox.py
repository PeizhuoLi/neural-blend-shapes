import torch


class BoundingBox:
    def __init__(self, verts=None):
        """
        Initialization of batched bonding box
        @param verts: (batch, n_vert, 3)
        """
        self.verts = verts

        if self.verts is None:
            return

        left_std = torch.tensor([ 0.8356,  0.2134, -0.0529], device=verts.device)
        right_std = torch.tensor([-0.8349,  0.2105, -0.0566], device=verts.device)

        left_id = torch.argmax(verts[..., 0], dim=-1)
        right_id = torch.argmin(verts[..., 0], dim=-1)
        left = verts[list(range(verts.shape[0])), left_id]
        right = verts[list(range(verts.shape[0])), right_id]

        scale = (left[..., 0] - right[..., 0]) / (left_std[..., 0] - right_std[..., 0])

        delta_y = left_std[..., 1] + right_std[..., 1] - (left[..., 1] + right[..., 1]) / scale
        delta_y /= -2

        self.scale = scale[:, None, None]
        self.center_of_mass = torch.zeros((verts.shape[0], 1, 3), device=verts.device)
        self.center_of_mass[..., 0, 1] = delta_y

    def normalize(self, verts):
        if self.verts is None:
            return verts

        verts = verts - self.center_of_mass
        verts = verts / self.scale
        return verts

    def normalize_offset(self, offsets):
        if self.verts is None:
            return offsets

        offsets[..., :1, :] = offsets[..., :1, :] - self.center_of_mass
        offsets = offsets / self.scale
        return offsets

    def denormalize(self, verts):
        if self.verts is None:
            return verts

        return verts * self.scale + self.center_of_mass

    def denormalize_offset(self, offsets):
        if self.verts is None:
            return offsets

        res = offsets * self.scale
        res[..., :1, :] = res[..., :1, :] + self.center_of_mass
        return res

    def denormalize_basis(self, basis):
        if self.verts is None:
            return basis

        return basis * self.scale
