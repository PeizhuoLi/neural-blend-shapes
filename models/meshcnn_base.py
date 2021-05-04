from meshcnn.models.layers.mesh import Mesh
from os.path import join as pjoin
from loss_recorder import LossRecorder
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.sparse
from models.transforms import batch_mm


class MeshCNNWrapper:
    """
    This class wraps MeshCNN code for easier use
    """
    def __init__(self, device,
                 is_train=True,
                 save_path=None,
                 topo_loader=None,
                 requires_recorder=True,
                 is_cont=False,
                 save_freq=500):
        if save_path is None:
            save_path = './results/ae/'
        """
        Make IDE happy
        """
        self.meshes = None
        self.optimizer = None
        self.model = None
        self.scheduler = None
        self.topo_id = None
        self.v_mask = None
        self.input = None

        self.save_freq = save_freq
        self.epoch_count = 0
        self.is_train = is_train
        self.device = device
        self.topo_loader = topo_loader

        self.edges_set = topo_loader.edges
        self.e2v_mat_set = []
        self.num_verts_set = []
        self.num_edges_set = []
        self.deform_grad_set = []
        for i in range(len(topo_loader)):
            num_verts = topo_loader.t_poses[i].shape[0]
            num_edges = topo_loader.edges[i].shape[0]
            self.num_verts_set.append(num_verts)
            self.num_edges_set.append(num_edges)
            indices = []
            val = []
            for i, neighbors in enumerate(self.topo_loader.meshes[i].ve):
                for e in neighbors:
                    indices.append([i, e])
                val.extend([1 / len(neighbors)] * len(neighbors))
            val = torch.FloatTensor(val)
            indices = torch.LongTensor(indices).t()
            self.e2v_mat_set.append(torch.sparse.FloatTensor(indices, val, torch.Size([num_verts, num_edges])).to(
                device))

        self.criteria = torch.nn.MSELoss()

        # Set up saving dirs
        self.save_path = save_path
        if requires_recorder and is_train:
            loss_path = pjoin(save_path, 'logs/')
            if self.is_train:
                if not is_cont:
                    os.system(f'rm -r {loss_path}/*')
                self.loss_recorder = LossRecorder(SummaryWriter(loss_path), base=is_cont + 1)
        else:
            self.loss_recorder = None
        os.makedirs(pjoin(save_path, 'model'), exist_ok=True)
        os.makedirs(pjoin(save_path, 'optimizer'), exist_ok=True)

    def prepare_topology(self, topo_id):
        self.topo_id = topo_id
        self.num_verts = self.num_verts_set[topo_id]
        self.num_edges = self.num_edges_set[topo_id]
        self.e2v_mat = self.e2v_mat_set[topo_id]
        self.edges = self.edges_set[topo_id]
        self.v_mask = self.topo_loader.v_masks[topo_id]

    def create_mesh_from_data(self, verts=None, size=None):
        meshes = []
        hold_history = False # We don't need meshcnn's pool
        size = verts.shape[0] if verts is not None else size
        for i in range(size):
            mesh = self.topo_loader.meshes[self.topo_id]
            v = verts[i] if verts is not None else None
            new_mesh = Mesh.copy_from_mesh(mesh, v, hold_history=hold_history)
            meshes.append(new_mesh)
        return meshes

    def edge2vert(self, edges):
        return batch_mm(self.e2v_mat, edges)

    def vert2edge(self, verts, edges=None):
        if edges is None:
            edges = self.edges
        return (verts[..., edges[:, 0], :] + verts[..., edges[:, 1], :]) / 2

    def prepare_edge_repr(self, convert2edge=True):
        self.meshes = self.create_mesh_from_data(size=self.input.shape[0])
        if convert2edge:
            self.input = self.vert2edge(self.input)
        self.input = self.input.permute(0, 2, 1)

    def save_model(self, epoch=None):
        if not self.is_train:
            return
        if epoch is None:
            epoch = self.epoch_count

        if epoch % self.save_freq == 0 or epoch == 1000:
            torch.save(self.model.state_dict(), pjoin(self.save_path, 'model/%05d.pt' % epoch))
            torch.save(self.optimizer.state_dict(), pjoin(self.save_path, 'optimizer/%05d.pt' % epoch))

        torch.save(self.model.state_dict(), pjoin(self.save_path, 'model/latest.pt'))
        torch.save(self.optimizer.state_dict(), pjoin(self.save_path, 'optimizer/latest.pt'))

    def load_model(self, epoch=None):
        if epoch is None:
            epoch = self.epoch_count

        if isinstance(epoch, str):
            state_dict = torch.load(epoch, map_location=self.device)
            self.model.load_state_dict(state_dict)

        else:
            filename = ('%05d.pt' % epoch) if epoch != -1 else 'latest.pt'
            state_dict = torch.load(pjoin(self.save_path, f'model/{filename}'), map_location=self.device)
            self.model.load_state_dict(state_dict)

            if self.is_train:
                state_dict = torch.load(pjoin(self.save_path, f'optimizer/{filename}'), map_location=self.device)
                self.optimizer.load_state_dict(state_dict)
                self.epoch_count = epoch + 1

    def epoch(self):
        self.epoch_count += 1
        if self.loss_recorder is not None:
            self.loss_recorder.epoch()
        if self.scheduler is not None:
            self.scheduler.step()
