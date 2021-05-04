import numpy as np
import torch


def chamfer_weight(weight, gt):
    """
    Calculate L1 chamfer distance between given skinning weight and ground truth
    """
    res = []
    if isinstance(weight, torch.Tensor):
        weight = weight.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    for j in range(weight.shape[1]):
        delta = weight[:, j][:, None] - gt
        delta = np.abs(delta).mean(axis=0)
        res.append(np.min(delta, axis=0))
    res = np.array(res)
    return res.mean()


def chamfer_weight_L2(weight, gt):
    """
    Calculate L2 chamfer distance between given skinning weight and ground truth
    """
    res = []
    if isinstance(weight, torch.Tensor):
        weight = weight.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    for j in range(weight.shape[1]):
        delta = weight[:, j][:, None] - gt
        delta = (delta**2).mean(axis=0)
        res.append(np.min(delta, axis=0))
    res = np.array(res)
    return res.mean()


def vert_distance(vs, gt):
    """
    Calculate L2 loss between given vertex location and ground truth while ignore displacement
    """
    if isinstance(vs, torch.Tensor):
        vs = vs.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    d = vs.mean(axis=1, keepdims=True) - gt.mean(axis=1, keepdims=True)
    vs = vs - d
    delta = vs - gt
    delta = delta ** 2
    return delta.mean(), delta.max()


def _chamfer_j2j(a, b):
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()

    res = []
    for i in range(a.shape[0]):
        delta = a[[i]] - b
        delta = (delta ** 2).sum(axis=-1)
        res.append(delta.min())
    res = np.array(res)
    return res.mean()


def _chamfer_j2b(a, b, parent):
    joint = offset2joint(a, parent)
    bone = offset2bone(b, parent)
    dist = point2segment(joint, bone)
    dist = dist.min(dim=-1)[0]
    return (dist**2).mean().item()


def _chamfer_b2b(a, b, parent):
    a = offset2bone(a, parent)
    b = offset2bone(b, parent)
    res = np.empty((a.shape[0], b.shape[0]))
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            res[i][j] = segment2segment(a[i][0], a[i][0] + a[i][1], b[i][0], b[i][0] + b[i][1])
    res = res.min(axis=-1)
    return (res**2).mean()


def chamfer_j2j(joint, gt, parent):
    joint = offset2joint(joint, parent)
    gt = offset2joint(gt, parent)
    return (_chamfer_j2j(joint, gt) + _chamfer_j2j(gt, joint)) / 2


def chamfer_j2b(joint, gt, parent):
    return (_chamfer_j2b(joint, gt, parent) + _chamfer_j2b(gt, joint, parent)) / 2


def chamfer_b2b(joint, gt, parent):
    return (_chamfer_b2b(joint, gt, parent) + _chamfer_b2b(gt, joint, parent)) / 2


def offset2joint(offsets, parent):
    res = offsets.clone()
    for i, p in enumerate(parent):
        if i == 0:
            continue
        res[i] += res[p]
    return res


def offset2bone(offset, parent):
    joint = offset2joint(offset, parent)
    res = []
    for i, p in enumerate(parent):
        if i == 0:
            continue
        res.append(torch.stack([joint[p], joint[i] - joint[p]], dim=0))
    res = torch.stack(res, dim=0)
    return res


def segment2segment(a0, a1, b0, b1, clampAll=True, clampA0=False, clampA1=False, clampB0=False, clampB1=False):
    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
        https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
    '''

    a0 = a0.detach().cpu().numpy()
    a1 = a1.detach().cpu().numpy()
    b0 = b0.detach().cpu().numpy()
    b1 = b1.detach().cpu().numpy()

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0 = True
        clampA1 = True
        clampB0 = True
        clampB1 = True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross) ** 2

    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A, (b0 - a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0, b0, np.linalg.norm(a0 - b0)
                    return a0, b1, np.linalg.norm(a0 - b1)


            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1, b0, np.linalg.norm(a1 - b0)
                    return a1, b1, np.linalg.norm(a1 - b1)

        # Segments overlap, return distance between parallel segments
        return None, None, np.linalg.norm(((d0 * _A) + a0) - b0)

    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA / denom
    t1 = detB / denom

    pA = a0 + (_A * t0)  # Projected closest point on segment A
    pB = b0 + (_B * t1)  # Projected closest point on segment B

    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B, (pA - b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A, (pB - a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return np.linalg.norm(pA - pB)


def point2segment(vertices, segments):
    P = vertices.unsqueeze(1)
    A = segments[..., 0, :]
    AP = P - A
    v = segments[..., 1, :]
    v_norm = torch.norm(v, dim=-1)
    B = A + v

    da = torch.norm(AP, dim=-1)
    db = torch.norm(P - B, dim=-1)
    dh0 = torch.sum(AP * v, dim=-1) / v_norm
    ds = (torch.norm(AP, dim=-1)**2 - dh0**2) ** 0.5
    t = dh0 / v_norm
    idx = (t < 0) + (t > 1)
    ds[idx] = da[idx]

    distance = torch.stack([da, db, ds], dim=2)
    distance, _ = torch.min(distance, dim=-1)
    return distance
