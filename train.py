import os
import torch
import numpy as np
from os.path import join as pjoin
from dataset.topology_loader import TopologyLoader
from architecture.generate_model import EnvelopeGenerate, BlendShapesGenerate
from architecture import create_envelope_model, create_residual_model
from models.kinematics import ForwardKinematics
from dataset.mesh_dataset import SMPLDataset, MultiGarmentDataset, generate_pose, parent_smpl
from option import TrainingOptionParser
from tqdm import tqdm
import random


def create_model(device, args, topo_loader):
    fk = ForwardKinematics(parents=parent_smpl)

    geo, att, gen = create_envelope_model(device, args, topo_loader, is_train=args.envelope, parents=parent_smpl)
    envelope_model = EnvelopeGenerate(geo, att, gen, fk=fk, args=args)

    geo2, _, gen2, coff = create_residual_model(device, args, topo_loader, is_train=args.residual, parents=parent_smpl,
                                                requires_att=False)

    residual_model = BlendShapesGenerate(geo2, att, gen2, coff, args=args, fk=fk)

    optimizer = torch.optim.Adam

    if args.envelope:
        for sub_model in envelope_model.models.values():
            if sub_model is None:
                continue
            if sub_model == att:
                sub_model.set_optimizer(lr=args.lr_att, optimizer=optimizer)
            else:
                sub_model.set_optimizer(lr=args.lr, optimizer=optimizer)
    elif args.residual:
        envelope_model.load_model(-1)
        for sub_model in residual_model.models.values():
            if sub_model == coff:
                if not args.fast_train:
                    sub_model.set_optimizer(lr=args.lr_coff, optimizer=optimizer)
            elif sub_model == att:
                continue
            else:
                sub_model.set_optimizer(lr=args.lr, optimizer=optimizer)

    return envelope_model, residual_model


def prepare_dataset(device, args):
    topo_loader = TopologyLoader(device=device, debug=args.debug)

    # Prepare SMPL dataset and MultiGarmentDataset
    dataset_smpl = SMPLDataset(device=device)
    dataset_garment = MultiGarmentDataset('./dataset/Meshes/MultiGarment', topo_loader, device)

    # Prepare topology augmentation
    if args.topo_augment:
        begin_aug_topo, len_topo = topo_loader.load_smpl_group('./dataset/Meshes/SMPL/topology/', is_train=True)
    else:
        begin_aug_topo = topo_loader.load_from_obj('./dataset/eval_constant/meshes/smpl_std.obj')
        len_topo = 1

    return topo_loader, dataset_smpl, dataset_garment, begin_aug_topo, len_topo


def main():
    parser = TrainingOptionParser()
    args = parser.parse_args()

    batch_size = args.batch_size

    device = torch.device(args.device)
    if args.device != 'cpu':
        torch.cuda.set_device(device)

    if args.envelope:
        parser.save(pjoin(args.save_path, 'args.txt'))

    topo_loader, dataset_smpl, dataset_garment, begin_aug_topo, len_topo = prepare_dataset(device, args)
    envelope_model, residual_model = create_model(device, args, topo_loader)

    if args.envelope:
        del residual_model
        model = envelope_model
    elif args.residual:
        model = residual_model
        if args.fast_train:
            basis = np.load(pjoin(args.save_path, 'smpl_preprocess/basis.npy'))
            basis = torch.tensor(basis, device=device)
            basis = basis[None]
            os.makedirs(pjoin(args.save_path, 'coff/model'), exist_ok=True)
            cmd = f"cp {pjoin(args.save_path, 'smpl_preprocess/full_model.pt')} {pjoin(args.save_path, 'coff/model/latest.pt')}"
            os.system(cmd)
        else:
            basis = None
    else:
        raise Exception('Unknown training stage')

    loop = tqdm(range(args.num_epoch))
    it_cnt = 0

    for epoch in loop:
        for _ in range(10):  # We simply take 10 iterations as an epoch
            model.zero_grad()

            dataset = dataset_smpl if it_cnt % 2 == 0 else dataset_garment
            topo_id = begin_aug_topo + random.randint(0, len_topo - 1)

            if args.envelope:
                pose = generate_pose(batch_size, device)
                pose_ee = generate_pose(batch_size, device, uniform=args.ee_uniform, factor=args.ee_factor, ee=dataset.end_effectors(args.ee_order))
                # Examples for capture end_effector deformation
                deformed, deformed_ee, t_pose, root_loc = dataset.forward(pose, pose_ee)

                model.forward(t_pose, pose, topo_id, pose_ee=pose_ee)
                model.backward(deformed, deformed_ee, requires_backward=True, gt_root_loc=root_loc)
            elif args.residual:
                if args.fast_train:
                    pose = generate_pose(batch_size, device, uniform=True)  # placeholder
                    _, t_pose, _ = dataset.forward(pose)
                    model.forward(t_pose, pose, topo_id, None, basis_only=True)
                    model.backward(gt_basis=basis)
                else:
                    pose = generate_pose(args.pose_batch_size, device, uniform=True)
                    deformed, t_pose, skeleton = dataset.forward_multipose(pose, batch_size, residual=True,
                                                                           requires_skeleton=True)
                    model.forward(t_pose, pose, topo_id, skeleton, basis_only=False)
                    model.backward(gt_verts=deformed)

            model.optim_step()
            it_cnt += 1

        if epoch % 50 == 0:
            model.save_model()
        model.epoch()


if __name__ == '__main__':
    main()
