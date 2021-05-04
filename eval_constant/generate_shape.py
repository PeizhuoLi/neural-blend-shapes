import random
import numpy as np
import torch
from dataset.mesh_dataset import generate_shape, generate_pose


if __name__ == '__main__':
    random.seed(0)
    torch.random.manual_seed(0)

    device = torch.device('cpu')
    shape = generate_shape(10, device)
    pose = generate_pose(100, device)

    shape = shape.detach().cpu().numpy()

    # np.save('./eval_constant/test_shape.npy', shape)
    np.save('./eval_constant/test_pose.npy', pose)
