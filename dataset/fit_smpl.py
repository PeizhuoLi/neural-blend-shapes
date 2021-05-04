import torch
from tqdm import tqdm
from models.deformation import LBS


def fit(target, smpl_layer, show_progress=False):
    poses = torch.zeros((target.shape[0], 72), requires_grad=True, device=target.device)
    shapes = torch.zeros((target.shape[0], 10), requires_grad=True, device=target.device)
    global_pos = torch.zeros((target.shape[0], 3), requires_grad=True, device=target.device)

    optimizer = torch.optim.Adam([poses, shapes, global_pos], lr=1e-1)
    criteria = torch.nn.MSELoss()

    n_iter = 200
    loop = tqdm(range(n_iter)) if show_progress else range(n_iter)
    for _ in loop:
        optimizer.zero_grad()
        res_verts, _ = smpl_layer.forward(poses, shapes)
        res_verts = res_verts + global_pos.unsqueeze(1)
        loss = criteria(res_verts, target)
        loss.backward()
        optimizer.step()
        if show_progress:
            loop.set_description('loss = %.10f' % loss.item())

    return poses, shapes, global_pos


def unpose(target, smpl_layer, show_progress=False):
    poses, shapes, global_pose = fit(target, smpl_layer, show_progress)
    ref_verts, _, bone_mat = smpl_layer.forward(poses, shapes, requires_transformation=True)
    bone_mat = bone_mat.permute(0, 3, 1, 2)

    vert_mat = LBS(smpl_layer.th_weights, bone_mat)
    vert_mat_i = torch.inverse(vert_mat)

    un_posed = torch.matmul(vert_mat_i[..., :3, :3], (target - global_pose.unsqueeze(1)).unsqueeze(-1)).squeeze(
        -1) + vert_mat_i[..., :3, 3]
    pb = smpl_layer.pose_blendshapes(poses)

    offsets = smpl_layer.get_offset(shapes)

    return un_posed - pb, offsets, ref_verts, poses, shapes
