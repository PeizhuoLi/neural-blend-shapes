import os
import torch
import numpy as np
import argparse
from os.path import join as pjoin
from dataset.topology_loader import TopologyLoader
from architecture.generate_model import EnvelopeGenerate, BlendShapesGenerate
from architecture import create_envelope_model, create_residual_model
from models.kinematics import ForwardKinematics
from models.transforms import aa2mat
from models.deformation import deform_with_offset
from dataset.mesh_dataset import StaticMeshes, parent_smpl
from dataset.load_test_anim import load_test_anim
from dataset.bvh_writer import WriterWrapper
from dataset.obj_io import write_obj
from option import TrainingOptionParser
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--pose_file', type=str, default='./eval_constant/sequences/house-dance.npy')
    parser.add_argument('--model_path', type=str, default='./pre_trained')
    parser.add_argument('--obj_path', type=str, default='./eval_constant/meshes/artist-2.obj')
    parser.add_argument('--result_path', type=str, default='./demo')
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--envelope_only', type=int, default=0)
    parser.add_argument('--animated_bvh', type=int, default=0)
    parser.add_argument('--obj_output', type=int, default=1)
    return parser


def eval_envelop(t_pose, topo_id, model: EnvelopeGenerate, pose=None):
    with torch.no_grad():
        t_pose = t_pose.unsqueeze(0)
        model.forward(t_pose, pose, topo_id)
    skinning_weight = model.att_vert[0]
    skeleton = model.skeleton[0]
    return skinning_weight, skeleton


def eval_residual(t_pose, topo_id, pose, model: BlendShapesGenerate):
    with torch.no_grad():
        t_pose = t_pose.unsqueeze(0)
        model.forward(t_pose, pose, topo_id, None)
    return model.offsets[0], model.models['bs'].basis_full, model.models['bs'].coff


def load_model(device, model_args, topo_loader, save_path_base, envelope_only, epoch_num=-1):
    """
    Important: Make sure prepare the dataset before create the model
    """
    fk = ForwardKinematics(parents=parent_smpl)
    geo, att, gen = create_envelope_model(device, model_args, topo_loader, is_train=False, parents=parent_smpl)
    envelop_model = EnvelopeGenerate(geo, att, gen, fk=fk, args=model_args)

    if not envelope_only:
        model_args.fast_train = 0
        geo2, _, gen2, coff = create_residual_model(device, model_args, topo_loader, is_train=False, parents=parent_smpl,
                                                    requires_att=False)
        residual_model = BlendShapesGenerate(geo2, att, gen2, coff, args=model_args, fk=fk)

        sub_models = [geo, att, gen, geo2, gen2, coff]

    else:
        residual_model = None
        sub_models = [geo, att, gen]

    for sub_model in sub_models:
        o_path = sub_model.save_path
        if o_path.endswith('/'): o_path = o_path[:-1]
        o_path = o_path.split('/')[-1]
        sub_model.save_path = pjoin(save_path_base, o_path)
        sub_model.load_model(epoch_num)

    return envelop_model, residual_model


def run_single_mesh(verts, topo_id, pose, env_model, res_model, requires_lbs=False):
    skinning_weight, skeleton = eval_envelop(verts, topo_id, env_model)
    if res_model is not None:
        offset, basis, coff = eval_residual(verts, topo_id, pose, res_model)
    else:
        offset = 0
        basis = None
        coff = None

    local_mat = aa2mat(pose.reshape(pose.shape[0], -1, 3))
    global_mat = env_model.fk.forward(local_mat, skeleton.unsqueeze(0))
    mask = env_model.rec_model.topo_loader.v_masks[topo_id]
    verts = verts[mask]
    vs = deform_with_offset(verts, skinning_weight, global_mat, offset)
    if requires_lbs:
        vs_lbs = deform_with_offset(verts, skinning_weight, global_mat)
        return skinning_weight, skeleton, vs, vs_lbs, basis, coff
    return skinning_weight, skeleton, vs, basis, coff


def prepare_obj(filename, topo_loader):
    mesh = StaticMeshes([filename], topo_loader)
    return mesh


def write_back(prefix, skeleton, skinning_weight, verts, faces, original_path, rot, basis, coff):
    os.makedirs(prefix, exist_ok=True)
    os.makedirs(pjoin(prefix, 'obj'), exist_ok=True)

    bvh_writer = WriterWrapper(parent_smpl)
    skinning_weight = skinning_weight.detach().cpu().numpy()
    np.save(pjoin(prefix, 'weight.npy'), skinning_weight)
    if basis is not None:
        basis = basis.detach().cpu().numpy()
        np.save(pjoin(prefix, 'basis.npy'), basis.squeeze())
    if coff is not None:
        coff = coff.detach().cpu().numpy()
        np.save(pjoin(prefix, 'coff.npy'), coff.squeeze())
    bvh_writer.write(pjoin(prefix, 'skeleton.bvh'), skeleton, rot)

    os.system(f"cp {original_path} {pjoin(prefix, 'T-pose.obj')}")

    if os.path.exists(pjoin(prefix, 'obj')):
        os.system(f"rm -r {pjoin(prefix, 'obj/*')}")
    if verts is not None:
        print('Writing back...')
        for i in tqdm(range(verts.shape[0])):
            write_obj(pjoin(prefix, 'obj/%05d.obj' % i), verts[i], faces)


def main():
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device(args.device)

    train_parser = TrainingOptionParser()
    model_args = train_parser.load(pjoin(args.model_path, 'args.txt'))
    model_args.normalize = args.normalize

    test_pose, test_loc = load_test_anim(args.pose_file, device)

    topo_loader = TopologyLoader(device=device, debug=False)
    mesh = prepare_obj(args.obj_path, topo_loader)

    env_model, res_model = load_model(device, model_args, topo_loader, args.model_path, args.envelope_only)

    t_pose, topo_id = mesh[0]
    skinning_weight, skeleton, vs, basis, coff = run_single_mesh(t_pose, topo_id, test_pose, env_model, res_model)

    faces = topo_loader.faces[topo_id]

    if not args.animated_bvh:
        test_pose = None
    if not args.obj_output:
        vs = None

    write_back(args.result_path, skeleton, skinning_weight, vs, faces, args.obj_path, test_pose, basis, coff)


if __name__ == '__main__':
    main()
