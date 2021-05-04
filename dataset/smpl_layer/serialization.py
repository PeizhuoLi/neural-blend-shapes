def ready_arguments(fname_or_dict, highRes):
    import numpy as np
    import pickle
    import chumpy as ch
    from chumpy.ch import MatVecMult
    from dataset.smpl_layer.posemapper import posemap
    import scipy.sparse as sp

    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')
        # dd = pickle.load(open(fname_or_dict, 'rb'))
    else:
        dd = fname_or_dict

    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1] * 3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if (s in dd) and isinstance(dd[s], ch.ch.Ch):
            dd[s] = dd[s].r

    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas']) + dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:, 0])
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:, 1])
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:, 2])
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
    else:
        dd['v_posed'] = dd['v_template'] + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))

    if highRes is not None:
        with open(highRes, 'rb') as f:
            mapping, hf = pickle.load(f, encoding='latin1')
        num_betas = dd['shapedirs'].shape[-1]
        hv = mapping.dot(dd['v_template'].ravel()).reshape(-1, 3)
        J_reg = dd['J_regressor'].asformat('csr')
        dd['f'] = hf
        dd['v_template'] = hv
        dd['weights'] = np.hstack([
                np.expand_dims(
                    np.mean(
                        mapping.dot(np.repeat(np.expand_dims(dd['weights'][:, i], -1), 3)).reshape(-1, 3)
                        , axis=1),
                    axis=-1)
                for i in range(24)
            ])
        dd['posedirs'] = mapping.dot(dd['posedirs'].reshape((-1, 207))).reshape(-1, 3, 207)
        dd['shapedirs'] = mapping.dot(dd['shapedirs'].reshape((-1, num_betas))).reshape(-1, 3, num_betas)
        dd['J_regressor'] = sp.csr_matrix((J_reg.data, J_reg.indices, J_reg.indptr), shape=(24, hv.shape[0]))

    return dd
