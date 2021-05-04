from models.networks import MeshReprConv, MLP, MLPSkeleton
from architecture.blend_shapes import BlendShapesModel
from os.path import join as pjoin


def create_envelope_model(device, args, topo_loader, parents=None, is_train=True):
    base = args.base
    layers = args.num_layers
    bone_num = 24

    channel_list = [base]

    for i in range(layers - 1):
        channel_list.append(channel_list[-1] * 2)
    geo_list = [3] + channel_list   # This is for vertex position

    gen_list = geo_list[::-1]

    if not args.skeleton_aware:
        gen_list = [c * bone_num for c in gen_list]
    else:
        gen_list = [c * bone_num for c in gen_list]

    channel_list = [args.att_base]
    for i in range(layers - 2):
        channel_list.append(channel_list[-1] * 2)
    att_list = [3] + channel_list + [bone_num]

    save_path = args.save_path

    geometry_branch = MeshReprConv(device, is_train=is_train, save_path=pjoin(save_path, 'geo/'),
                                   channels=geo_list,
                                   topo_loader=topo_loader, requires_recorder=True, is_cont=args.cont,
                                   save_freq=args.save_freq)

    att_branch = MeshReprConv(device, is_train=is_train, save_path=pjoin(save_path, 'att/'),
                              channels=att_list,
                              topo_loader=topo_loader, last_activate=False, requires_recorder=False,
                              pool_ratio=args.pool_ratio, pool_method=args.pool_method,
                              is_cont=args.cont, save_freq=args.att_save_freq)

    if not args.skeleton_aware:
        gen_branch = MLP(layers=gen_list,
                         save_path=pjoin(save_path, 'gen/'),
                         is_train=is_train,
                         device=device).to(device)
    else:
        gen_branch = MLPSkeleton(layers=gen_list, parents=parents,
                                 save_path=pjoin(save_path, 'gen/'),
                                 is_train=is_train, save_freq=args.save_freq,
                                 device=device).to(device)

    return geometry_branch, att_branch, gen_branch


def create_residual_model(device, args, topo_loader, is_train=True, parents=None, requires_att=True):
    base = args.base
    layers = args.num_layers
    bone_num = 24

    channel_list = [base]

    for i in range(layers - 1):
        channel_list.append(channel_list[-1] * 2)
    geo_list = [3] + channel_list  # This is for vertex position

    gen_list = geo_list[::-1]
    gen_list = gen_list[:2] + [args.basis_per_bone * 3]
    gen_list[0] += bone_num

    channel_list = [args.att_base]
    for i in range(layers - 2):
        channel_list.append(channel_list[-1] * 2)
    att_list = [3] + channel_list + [bone_num]

    save_path = args.save_path

    geometry_branch = MeshReprConv(device, is_train=is_train, save_path=pjoin(save_path, 'geo2/'),
                                   channels=geo_list,
                                   topo_loader=topo_loader, requires_recorder=True, is_cont=args.cont,
                                   save_freq=args.save_freq)

    if requires_att:
        att_branch = MeshReprConv(device, is_train=False, save_path=pjoin(args.att_load_path, 'att/'),
                                  channels=att_list,
                                  topo_loader=topo_loader, last_activate=False, requires_recorder=False,
                                  pool_ratio=args.pool_ratio, pool_method=args.pool_method,
                                  is_cont=args.cont, save_freq=args.att_save_freq)
    else:
        att_branch = None

    gen_branch = MeshReprConv(device, is_train=is_train, save_path=pjoin(save_path, 'dec/'),
                              channels=gen_list,
                              topo_loader=topo_loader, last_activate=False, requires_recorder=False,
                              is_cont=args.cont, last_init_div=args.offset_init_div)

    coff_branch = BlendShapesModel(1, bone_num - 1, args.basis_per_bone, parent=parents, basis_as_model=False,
                                   save_freq=args.save_freq, save_path=pjoin(save_path, 'coff/'), device=device).to(device)

    return geometry_branch, att_branch, gen_branch, coff_branch
