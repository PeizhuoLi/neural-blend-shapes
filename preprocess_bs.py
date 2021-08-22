import torch
import os
from tqdm import tqdm
from dataset.mesh_dataset import generate_pose
from option import TrainingOptionParser
from os.path import join as pjoin
from dataset.smpl import SMPL_Layer
import numpy as np
from architecture.blend_shapes import BlendShapesModel


def main():
    n_iter = 1000
    basis_per_bone = 9
    write_back = True

    parser = TrainingOptionParser()
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device != 'cpu':
        torch.cuda.set_device(device)

    save_prefix = pjoin(args.save_path, 'smpl_preprocess')
    os.makedirs(save_prefix, exist_ok=True)

    smpl_layer = SMPL_Layer().to(device)

    model = BlendShapesModel(smpl_layer.num_verts, smpl_layer.num_joints - 1, basis_per_bone,
                             weight=smpl_layer.weights, parent=smpl_layer.kintree_parents).to(device)

    batch_size = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loop = tqdm(range(n_iter))

    for _ in loop:
        optimizer.zero_grad()
        pose = generate_pose(batch_size, device, uniform=True)

        res = model.forward(pose)
        gt = smpl_layer.pose_blendshapes(pose)
        loss = torch.nn.MSELoss()(res, gt)
        loss.backward()
        optimizer.step()
        loop.set_description('loss = %e' % loss.item())

    if write_back:
        torch.save(model.state_dict(), pjoin(save_prefix, 'full_model.pt'))
        np.save(pjoin(save_prefix, 'basis.npy'), model.basis.detach().cpu().numpy())


if __name__ == '__main__':
    main()
