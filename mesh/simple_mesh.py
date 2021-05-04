import torch
import numpy as np
import torch.sparse
from os.path import join as pjoin
import os


def get_sparse_identity(n):
    idx = torch.LongTensor([[i, i] for i in range(n)])
    value = torch.FloatTensor([1.] * n)
    return torch.sparse.FloatTensor(idx.t(), value, torch.Size([n, n]))


class SimpleMesh:
    """
    A simple mesh class, contains only faces and vertex position
    Support edge collapse, edge flip
    It only works with 2-manifold meshes
    v_mask: v_mask[i] == 0 means i-th vertex is masked out
    """
    def __init__(self, vs=None, faces=None):
        self.vs = vs    # vertex position
        self.ve = None    # vertex's neighbor edge id
        self.vv = None    # vertex's neighbor vertex id
        self.faces = faces # faces, shape = (n_face, 3)
        self.v_mask = None
        self.f_mask = None
        self.e_mask = None
        self.edge2key = None # Mapping tuple (u, v) to corresponding id
        self.edge_cnt = None # number of edges
        self.ef = None    # edge's neighbor face id
        self.edges = None
        self.vs_mat = None
        if vs is not None and faces is not None:
            self.prepare_mesh()

    def load(self, vs, faces):
        if isinstance(vs, torch.Tensor):
            vs = vs.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        self.vs = vs
        self.faces = faces
        self.prepare_mesh()

    def make_edge(self, face, i):
        if isinstance(face, int):
            face = self.faces[face]
        res = (face[i], face[(i + 1) % 3])
        if res[0] > res[1]: return (res[1], res[0])
        else: return res

    def prepare_mesh(self):
        """
        Set up mesh data-structure. Requires self.vs and self.faces set.
        :return:
        """
        self.edge_cnt = 0
        self.edge2key = {}
        self.ef = []
        self.ve = [[] for _ in range(self.vs.shape[0])]
        self.vv = [[] for _ in range(self.vs.shape[0])]
        self.edges = []
        for face_id, face in enumerate(self.faces):
            for i in range(3):
                edge = self.make_edge(face, i)
                if edge not in self.edge2key:
                    self.edge2key[edge] = self.edge_cnt
                    self.ve[edge[0]].append(self.edge_cnt)
                    self.ve[edge[1]].append(self.edge_cnt)
                    self.vv[edge[0]].append(edge[1])
                    self.vv[edge[1]].append(edge[0])
                    self.edges.append(edge)
                    self.edge_cnt += 1
                    self.ef.append([])
                self.ef[self.edge2key[edge]].append(face_id)

        self.edges = np.asarray(self.edges, dtype=np.int)
        self.v_mask = np.ones((self.vs.shape[0], ), dtype=np.bool)
        self.f_mask = np.ones((self.faces.shape[0], ), dtype=np.bool)
        self.e_mask = np.ones((self.edges.shape[0], ), dtype=np.bool)

    def load(self, file, need_prepare=True):
        """
        This code is adapted from MeshCNN
        """
        self.vs, self.faces = [], []
        with open(file) as f:
            for line in f:
                line = line.strip()
                split_line = line.split()
                if not split_line:
                    continue
                elif split_line[0] == 'v':
                    self.vs.append([float(v) for v in split_line[1:4]])
                elif split_line[0] == 'f':
                    face_vertex_ids = [int(c.split('/')[0]) for c in split_line[1:]]
                    assert len(face_vertex_ids) == 3
                    face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(self.vs) + ind)
                                   for ind in face_vertex_ids]
                    self.faces.append(face_vertex_ids)
        self.vs = np.asarray(self.vs)
        self.faces = np.asarray(self.faces, dtype=np.int)
        assert np.logical_and(self.faces >= 0, self.faces < len(self.vs)).all()
        if need_prepare:
            self.prepare_mesh()

    def save(self, filename):
        vs, faces = self.clear()
        faces = faces + 1

        with open(filename, 'w') as f:
            for vi, v in enumerate(vs):
                f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for i, face in enumerate(faces):
                f.write('f %d %d %d\n' % (face[0], face[1], face[2]))

    def save_topology(self, path_name):
        os.makedirs(path_name, exist_ok=True)
        self.save(pjoin(path_name, 'T-pose.obj'))
        np.save(pjoin(path_name, 'v_mask.npy'), self.v_mask)

    def clear(self):
        vs = self.vs[self.v_mask]
        new_idx = np.zeros(self.v_mask.shape[0], dtype=np.int)
        new_idx[self.v_mask] = np.arange(0, self.v_mask.sum())
        faces = self.faces[self.f_mask]
        faces = new_idx[faces]
        return vs, faces

    def clear_(self):
        self.vs, self.faces = self.clear()
        self.v_mask = np.ones((self.vs.shape[0], ), dtype=np.bool)
        self.f_mask = np.ones((self.faces.shape[0], ), dtype=np.bool)
        self.prepare_mesh()

    def face_v_sanity_check(self):
        faces = self.faces[self.f_mask].reshape(-1)
        res = self.v_mask[faces]
        if not res.all():
            for i in range(self.faces.shape[0]):
                if not self.f_mask[i]: continue
                if not self.v_mask[self.faces[i]].all():
                    print(i)
            self.ef_sanity_check()
            self.save('./results/crash.obj')
            assert 0

    def face_edge_sanity_check(self):
        for fid, face in enumerate(self.faces):
            for i in range(3):
                edge = self.make_edge(fid, i)
                if edge not in self.edge2key:
                    assert 0

    def ve_sanity_check(self):
        ve = []
        for i, v in enumerate(self.ve):
            if not self.v_mask[i]: continue
            ve += v
        ve = np.array(ve, dtype=np.int)
        res = self.e_mask[ve]
        if not res.all():
            for i in range(self.vs.shape[0]):
                if not self.v_mask[i]: continue
                if not self.e_mask[self.ve[i]].all():
                    print(i)
                    assert 0

    def ef_sanity_check(self):
        for id, edge in enumerate(self.edges):
            if not self.e_mask[id]:
                continue
            for face in self.faces[self.ef[id]]:
                if not (edge[0] in face) and (edge[1] in face):
                    print("Dammit")
                    assert 0

    def manifold_check(self):
        edge_set = set()
        for face_id, face in enumerate(self.faces):
            if not self.f_mask[face_id]:
                continue
            for i in range(3):
                edge = (face[i], face[(i + 1) % 3])
                if edge in edge_set:
                    print(face_id)
                    self.save('./results/crash.obj')
                    assert 0
                edge_set.add(edge)

    def edge_collapse(self, target_eid):
        if not self.e_mask[target_eid]:
            return 0

        edge = self.edges[target_eid]

        # Verify the collapse will not cause a non-manifold face
        v_a = set(self.edges[self.ve[edge[0]]].reshape(-1))
        v_b = set(self.edges[self.ve[edge[1]]].reshape(-1))
        shared = (v_a & v_b) - set(edge)
        if len(shared) != 2:
            return 0

        # Change vs[edge[0]] to be the vs of mid point
        self.vs[edge[0]] = (self.vs[edge[0]] + self.vs[edge[1]]) / 2
        self.v_mask[edge[1]] = 0

        # Get all the faces that end point will change
        # IMPORTANT: This must come before remove edges on ve
        to_change_face = set()
        for edge_id in self.ve[edge[1]]:
            to_change_face.add(self.ef[edge_id][0])
            to_change_face.add(self.ef[edge_id][1])

        # Remove deleted edges in ve[edge[1]]
        self.ve[edge[1]].remove(target_eid)
        self.ve[edge[0]].remove(target_eid)
        for face_id in self.ef[target_eid]:
           for i in range(3):
                edge_to_change = self.make_edge(face_id, i)
                if edge[1] in edge_to_change:
                    id = self.edge2key[edge_to_change]
                    if id != target_eid:
                        self.ve[edge_to_change[0]].remove(id)
                        self.ve[edge_to_change[1]].remove(id)

        self.ve[edge[0]].extend(self.ve[edge[1]])

        # Change ef of edge[0]'s neighbor edge
        for face_id in self.ef[target_eid]:
            edge_a, edge_b = None, None
            for i in range(3):
                edge_t = self.make_edge(face_id, i)
                id = self.edge2key[edge_t]
                if id == target_eid:
                    continue
                if edge[0] in edge_t:
                    edge_a = id
                if edge[1] in edge_t:
                    edge_b = id
            ef_a = set(self.ef[edge_a])
            ef_b = set(self.ef[edge_b])
            remained_face = list((ef_a | ef_b) - ef_a)
            assert(len(remained_face) == 1)
            remained_face = remained_face[0]
            self.ef[edge_a].remove(face_id)
            self.ef[edge_a].append(remained_face)
            self.e_mask[edge_b] = 0

        # Change endpoints of edges connected to edge[1] and not deleted
        for edge_id in self.ve[edge[1]]:
            for i in range(2):
                if self.edges[edge_id, i] == edge[1]:
                    self.edges[edge_id, i] = edge[0]
                    new_edge = tuple(sorted([self.edges[edge_id, 0], self.edges[edge_id, 1]]))
                    self.edge2key[new_edge] = edge_id

        # Mask 2 neighbor faces
        self.f_mask[self.ef[target_eid]] = 0

        # Change vertex of faces connected to edge[1] and not deleted
        for face_id in to_change_face:
            if not self.f_mask[face_id]: continue
            for i in range(3):
                if self.faces[face_id, i] == edge[1]:
                    self.faces[face_id, i] = edge[0]

        self.e_mask[target_eid] = 0

        # self.ve_sanity_check()
        # self.face_v_sanity_check()

        return 1

    def edge_flip(self, target_eid):
        if not self.e_mask[target_eid]:
            return 0

        t_edge = self.edges[target_eid]

        v_a = set(self.edges[self.ve[t_edge[0]]].reshape(-1))
        v_b = set(self.edges[self.ve[t_edge[1]]].reshape(-1))
        shared = (v_a & v_b) - set(t_edge)

        if len(shared) != 2:
            return 0

        new_edge = tuple(sorted(list(shared)))

        if new_edge in self.edge2key and \
                (self.edges[self.edge2key[new_edge]] == np.array(new_edge, dtype=np.int64)).all():
            return 0

        # make sure new_edge[0/1] in self.ef[target_eid, 0/1] respectively
        if not (new_edge[0] in self.faces[self.ef[target_eid][0]] and
                new_edge[1] in self.faces[self.ef[target_eid][1]]):
            self.ef[target_eid] = self.ef[target_eid][::-1]

        # modify faces
        for i, fid in enumerate(self.ef[target_eid]):
            for j in range(3):
                if self.faces[fid, j] == t_edge[1 - i]:
                    self.faces[fid, j] = new_edge[1 - i]

        # modify ef
        for i in range(2):
            edge = tuple(sorted([t_edge[1 - i], new_edge[i]]))
            eid = self.edge2key[edge]
            self.ef[eid].remove(self.ef[target_eid][i])
            self.ef[eid].append(self.ef[target_eid][1 - i])

        # modify ve
        for i in range(2):
            self.ve[t_edge[i]].remove(target_eid)
            self.ve[new_edge[i]].append(target_eid)

        # modify edges
        self.edges[target_eid] = new_edge
        self.edge2key[new_edge] = target_eid

        return 1

    def get_uniform_laplacian(self):
        idx = []
        value = []
        n = self.vs.shape[0]
        for i in range(self.vs.shape[0]):
            idx.append((i, i))
            value.append(1.)

            for j in range(len(self.vv[i])):
                idx.append((i, self.vv[i][j]))
                value.append(1. / len(self.vv[i]))

        idx = torch.LongTensor(idx)
        value = torch.FloatTensor(value)
        return torch.sparse.FloatTensor(idx.t(), value, torch.Size([n, n]))
