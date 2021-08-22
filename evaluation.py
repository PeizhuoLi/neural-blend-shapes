import torch
import numpy as np
from os.path import join as pjoin
from option import TrainingOptionParser
from demo import get_parser, load_model, run_single_mesh
from dataset.mesh_dataset import parent_smpl
from dataset.load_test_anim import load_test_anim
from dataset.smpl import SMPL_Layer
from dataset.topology_loader import TopologyLoader
from models.measurement import chamfer_weight, vert_distance, chamfer_j2j, chamfer_j2b, chamfer_b2b
from tqdm import tqdm


def main():
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device(args.device)

    smpl = SMPL_Layer().to(device)

    train_parser = TrainingOptionParser()
    model_args = train_parser.load(pjoin(args.model_path, 'args.txt'))

    test_pose, test_loc = load_test_anim(args.pose_file, device)
    test_shape = torch.tensor(np.load('./eval_constant/test_shape.npy'), device=device)

    topo_loader = TopologyLoader(device=device, debug=False)
    smpl_topo_begin, len_topo_smpl = topo_loader.load_smpl_group('./dataset/Meshes/SMPL/topology/',
                                                                 is_train=False)

    env_model, res_model = load_model(device, model_args, topo_loader, args.model_path, envelope_only=False)

    res_weight = []
    res_skeleton = []
    res_verts = []
    res_verts_lbs = []

    gt_skeleton = smpl.get_offset(test_shape)
    gt_verts = []

    print('Evaluating model...')
    for i in tqdm(range(test_shape.shape[0])):
        pose_ph = torch.zeros((1, 72), device=device)
        t_pose = smpl.forward(pose_ph, test_shape[[i]])[0][0]
        # t_pose = t_pose[topo_loader.v_masks[i]]
        gt_vs = smpl.forward(test_pose, test_shape[[i]].expand(test_pose.shape[0], -1))[0]
        gt_vs = gt_vs[:, topo_loader.v_masks[i]]
        gt_verts.append(gt_vs)

        weight, skeleton, vs, vs_lbs, _, _ = run_single_mesh(t_pose, smpl_topo_begin + i, test_pose, env_model, res_model, requires_lbs=True)
        res_weight.append(weight)
        res_skeleton.append(skeleton)
        res_verts.append(vs)
        res_verts_lbs.append(vs_lbs)

    err_weight = []
    err_avg_verts = []
    err_max_verts = []
    err_lbs_verts = []
    err_j2j = []
    err_j2b = []
    err_b2b = []

    print('Aggregating error...')
    for i in tqdm(range(test_shape.shape[0])):
        mask = topo_loader.v_masks[i]
        weight_gt = smpl.weights[mask]
        err_weight.append(chamfer_weight(res_weight[i], weight_gt))

        err_vert = vert_distance(res_verts[i], gt_verts[i])
        err_lbs = vert_distance(res_verts_lbs[i], gt_verts[i])
        err_avg_verts.append(err_vert[0])
        err_max_verts.append(err_vert[1])
        err_lbs_verts.append(err_lbs[0])

        err_j2j.append(chamfer_j2j(res_skeleton[i], gt_skeleton[i], parent_smpl))
        err_j2b.append(chamfer_j2b(res_skeleton[i], gt_skeleton[i], parent_smpl))
        err_b2b.append(chamfer_b2b(res_skeleton[i], gt_skeleton[i], parent_smpl))

    err_weight = np.array(err_weight).mean()
    err_avg_verts = np.array(err_avg_verts).mean()
    err_max_verts = np.array(err_max_verts).mean()
    err_lbs_verts = np.array(err_lbs_verts).mean()
    err_j2j = np.array(err_j2j).mean()
    err_j2b = np.array(err_j2b).mean()
    err_b2b = np.array(err_b2b).mean()
    print('Skinning Weight L1 = %.7f' % err_weight)
    print('Vertex Mean Loss L2 = %.7f' % err_avg_verts)
    print('Vertex Max Loss L2 = %.7f' % err_max_verts)
    print('Envelope Mean Loss L2 = %.7f' % err_lbs_verts)
    print('CD-J2J = %.7f' % err_j2j)
    print('CD-J2B = %.7f' % err_j2b)
    print('CD-B2B = %.7f' % err_b2b)


if __name__ == '__main__':
    main()
