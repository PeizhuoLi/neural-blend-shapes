from dataset.fit_smpl import unpose
import os
from os.path import join as pjoin
from dataset.obj_io import load_obj
from dataset.smpl_layer.smpl_layer import SMPL_Layer
import numpy as np
from tqdm import tqdm
import torch


def batch_fit(source_prefix, dest_prefix):
    lst = [f for f in os.listdir(source_prefix) if os.path.isdir(pjoin(source_prefix, f))]
    lst.sort()

    layer = SMPL_Layer(highRes=True)

    loop = tqdm(enumerate(lst))
    for i, name in loop:
        s_prefix = pjoin(source_prefix, name)
        d_prefix = pjoin(dest_prefix, '%03d' % i)
        obj_name = pjoin(s_prefix, 'smpl_registered.obj')
        verts, _= load_obj(obj_name)
        res = unpose(verts.unsqueeze(0), layer)

        res = [p[0].detach().cpu().numpy() for p in res]
        os.makedirs(d_prefix, exist_ok=True)
        np.save(pjoin(d_prefix, 't-pose.npy'), res[0])
        np.save(pjoin(d_prefix, 'offset.npy'), res[1])
        np.save(pjoin(d_prefix, 'pose.npy'), res[3])
        np.save(pjoin(d_prefix, 'shape.npy'), res[4])


def extract_cloth(prefix, dest_file):
    lst = [f for f in os.listdir(prefix) if os.path.isdir(pjoin(prefix, f))]
    lst.sort()
    layer = SMPL_Layer(highRes=True)
    cloths = []
    for name in lst:
        prefix2 = pjoin(prefix, name)
        shape = np.load(pjoin(prefix2, 'shape.npy'))
        shape = torch.tensor(shape)
        pose = torch.zeros((1, 72))
        t_pose = np.load(pjoin(prefix2, 't-pose.npy'))
        t_pose = torch.tensor(t_pose)

        match_verts, _ = layer.forward(pose, shape.unsqueeze(0))
        cloth = t_pose.unsqueeze(0) - match_verts
        cloths.append(cloth)
    cloths = torch.cat(cloths, dim=0)
    np.save(dest_file, cloths.numpy())


if __name__ == '__main__':
    raise Exception("This module is no longer supported with MIT license's SMPL implementation")
    source = 'path-to-Multi-Garment_dataset/'
    dest = './dataset/Meshes/MultiGarment/'
    batch_fit(source, dest)
    extract_cloth(dest, dest + 'all_cloths.npy')
