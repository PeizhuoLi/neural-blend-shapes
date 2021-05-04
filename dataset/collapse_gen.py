from mesh.simple_mesh import SimpleMesh
from os.path import join as pjoin
import random
from tqdm import tqdm


if __name__ == '__main__':
    prefix = './dataset/Meshes/SMPL/topology'
    n_topo = 100
    collapse_cnt = 2000
    flip_cnt = 2000
    random.seed(0)
    for i in tqdm(range(n_topo)):
        mesh = SimpleMesh()
        mesh.load('./dataset/Meshes/SMPL/obj_quad/T-pose.obj')
        l = list(range(mesh.edge_cnt))
        random.shuffle(l)
        cnt = 0
        for idx in l:
            cnt += mesh.edge_flip(idx)
            if cnt == flip_cnt:
                break
        if cnt != collapse_cnt:
            assert 0

        cnt = 0
        for idx in l:
            cnt += mesh.edge_collapse(idx)
            if cnt == collapse_cnt:
                break
        if cnt != collapse_cnt:
            assert 0

        mesh.save_topology(pjoin(prefix, '%05d' % i))
