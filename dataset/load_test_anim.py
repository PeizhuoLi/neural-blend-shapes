import numpy as np
import torch


def load_test_anim(filename, device):
    anim = np.load(filename)
    anim = torch.tensor(anim, device=device, dtype=torch.float)
    poses = anim[:, :-3]
    loc = anim[:, -3:]
    loc[..., 1] += 1.1174
    loc = loc.unsqueeze(1)

    return poses, loc
