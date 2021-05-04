import torch
from models.transforms import inv_affine, inv_rigid_affine


def LBS(weight, trans):
    """
    Given per bone deformation and skinning weights, return per-vertex deformation
    :param weight: shape = (*, v, b)
    :param trans: per-bone transformation (*, b, x, y)
    """
    shape2_o = trans.shape[-2:]
    trans = trans.reshape(trans.shape[:-2] + (-1, ))
    res = torch.matmul(weight, trans)
    res = res.reshape(res.shape[:-1] + shape2_o)
    return res


def deform_with_offset(verts, weights, mat, offset=0):
    """
    Deform a given mesh with weights, per-bone mat and offset
    :param verts: (batch_size, n_vert, 3)
    :param weights: (batch_size, n_vert, n_bone)
    :param mat: (batch_size, n_bone, 3, 4)
    :param offset: Per-vertex residual deformation, shape=(batch_size, n_vert, 3) or 0
    :return: (batch_size, n_vert, 3)
    """
    verts = verts + offset
    verts = verts.unsqueeze(-1)
    vert_mat = LBS(weights, mat)
    verts = torch.matmul(vert_mat[..., :3], verts).squeeze(-1) + vert_mat[..., 3]
    return verts


def unpose(verts, weight, mat, inv_lbs=False):
    """
    Given a posed verts and corresponding global affine matrix, put the shape back in t-pose
    @param verts: (batch_size, n_vert, 3)
    @param weight: (batch_size, n_vert, n_bone)
    @param mat: (batch_size, n_bone, 3, 4)
    @param inv_lbs: Use inverted LBS or simple LBS
    @return: Unposed shapes
    """
    if inv_lbs:
        vert_mat = LBS(weight, mat)
        vert_mat_inv = inv_affine(vert_mat)
    else:
        mat_inv = inv_rigid_affine(mat)
        vert_mat_inv = LBS(weight, mat_inv)

    verts = verts.unsqueeze(-1)
    verts = torch.matmul(vert_mat_inv[..., :3], verts).squeeze(-1) + vert_mat_inv[..., 3]
    return verts
