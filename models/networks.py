import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiplicativeLR
from meshcnn.models.layers.mesh_conv import MeshConv
from models.skeleton import SkeletonLinear, find_neighbor_joint
import torch.sparse
from models.meshcnn_base import MeshCNNWrapper
from os.path import join as pjoin
import os


class PoolPartChannel(nn.Module):
    def __init__(self, ratio=0.5, pooling='mean'):
        super(PoolPartChannel, self).__init__()
        self.ratio = ratio
        self.pooling = pooling

    def forward(self, input):
        n_channel = input.shape[1]
        n_pool = int(n_channel * self.ratio)

        res = input.clone()
        to_pool = input[:, :n_pool, ...]
        if self.pooling == 'mean':
            res_pool = to_pool.mean(dim=2, keepdim=True)
        elif self.pooling == 'max':
            res_pool, _ = to_pool.max(dim=2, keepdim=True)
        else:
            raise Exception('Unknown pooling parameter')
        res[:, :n_pool, ...] = res_pool
        return res


class MeshReprConv(MeshCNNWrapper):
    """
    Build up block upon MeshCNN operators
    """
    def __init__(self, device,
                 is_train=True,
                 save_path=None,
                 channels=None,
                 topo_loader=None,
                 last_activate=False,
                 requires_recorder=True,
                 pool_ratio=2,   # pool only works when pool_ratio is smaller than 1
                 pool_method='mean',
                 is_cont=0,
                 last_init_div=1,
                 save_freq=500):

        if channels is None:
            channels = []
        super(MeshReprConv, self).__init__(device, is_train, save_path,
                                           topo_loader, requires_recorder, is_cont, save_freq)
        self.input = None
        self.gt = None
        self.channels = channels
        self.model = nn.ModuleList()
        self.convs = []
        self.last_activate = last_activate
        self.activate = torch.nn.LeakyReLU(negative_slope=0.2)
        if pool_ratio < 1:
            self.pool = PoolPartChannel(ratio=pool_ratio, pooling=pool_method).to(device)
        else:
            self.pool = None
        for i in range(len(channels) - 1):
            self.model.append(MeshConv(channels[i], channels[i + 1]).to(device))
            self.convs.append(self.model[-1])

        def div_param(param, div):
            param.data /= div

        # When it's used to output very small value (e.g., blend shapes), this initialization boosts training
        div_param(self.convs[-1].conv.weight, last_init_div)
        div_param(self.convs[-1].conv.bias, last_init_div)

    def set_optimizer(self, lr=1e-3, optimizer=torch.optim.Adam):
        params = self.model.parameters()
        self.optimizer = optimizer(params, lr=lr)

    def set_scheduler(self, lmbda):
        self.scheduler = MultiplicativeLR(self.optimizer, lr_lambda=lmbda, verbose=False)

    def apply_topo(self, x):
        if self.topo_id != -1 and self.topo_loader is not None and self.topo_loader.v_masks[self.topo_id] is not None:
            x = x[:, self.topo_loader.v_masks[self.topo_id]]
        return x

    def set_input(self, input, convert2edge=True):
        """
        Set input from vertex/edge features (after applied topogly)
        @param input: (batch_size, n_vert, n_channels) or (batch_size, n_edge, n_channels)
        @param convert2edge: Needs to convert vertex repr to edge repr?
        @return: No return
        """
        self.input = input
        self.prepare_edge_repr(convert2edge)

    def get_mask(self):
        return self.topo_loader.v_masks[self.topo_id]

    def forward(self):
        input = self.input
        for i, conv in enumerate(self.convs):
            input = conv(input, self.meshes)
            if i < len(self.model) - 1 or self.last_activate:
                input = self.activate(input)
                if self.pool is not None:
                    input = self.pool(input)
        input.squeeze_(-1)
        input = input.permute(0, 2, 1)
        self.res = input
        return input

    def backward(self):
        if self.gt is not None:
            self.vs = self.edge2vert(self.res)
            self.loss = self.criteria(self.vs, self.gt)
            self.loss.backward()
            self.loss_recorder.add_scalar('loss', self.loss)


class MLP(nn.Module):
    def __init__(self, layers, activation=None, save_path=None, is_train=True, device=None,
                 save_freq=2000, last_init_div=1):
        super(MLP, self).__init__()
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.save_freq = save_freq

        if save_path is None:
            save_path = './results/tmp/fc/'
        self.save_path = save_path

        self.layers = nn.ModuleList()
        self.step_layers = []

        self.is_train = is_train

        self.epoch_count = 0
        self.optimizer = None

        def div_param(param, div):
            param.data /= div

        for i in range(1, len(layers)):
            model = [nn.Linear(layers[i-1], layers[i])]
            if i == len(layers) - 1:
                div_param(model[0].bias, last_init_div)
                div_param(model[0].weight, last_init_div)
            if i < len(layers) - 1:
                if activation is None:
                    act = nn.LeakyReLU(negative_slope=0.2)
                else:
                    act = activation()
                model.append(act)
            self.step_layers.append(nn.Sequential(*model))
            self.layers.append(self.step_layers[-1])

        self.model = self.step_layers

        os.makedirs(pjoin(save_path, 'model'), exist_ok=True)
        os.makedirs(pjoin(save_path, 'optimizer'), exist_ok=True)

    def forward(self, input):
        for layer in self.step_layers:
            input = layer(input)
        return input

    def epoch(self):
        self.epoch_count += 1

    def set_optimizer(self, lr, optimizer=torch.optim.Adam):
        self.optimizer = optimizer(self.parameters(), lr=lr)

    def save_model(self, epoch=None):
        if epoch is None:
            epoch = self.epoch_count

        if epoch % self.save_freq == 0:
            torch.save(self.layers.state_dict(), pjoin(self.save_path, 'model/%05d.pt' % epoch))
            torch.save(self.optimizer.state_dict(), pjoin(self.save_path, 'optimizer/%05d.pt' % epoch))

        torch.save(self.layers.state_dict(), pjoin(self.save_path, 'model/latest.pt'))
        torch.save(self.optimizer.state_dict(), pjoin(self.save_path, 'optimizer/latest.pt'))

    def load_model(self, epoch=None):
        if epoch is None:
            epoch = self.epoch_count

        if isinstance(epoch, str):
            state_dict = torch.load(epoch, map_location=self.device)
            self.layers.load_state_dict(state_dict)

        else:
            filename = ('%05d.pt' % epoch) if epoch != -1 else 'latest.pt'
            state_dict = torch.load(pjoin(self.save_path, f'model/{filename}'), map_location=self.device)
            self.layers.load_state_dict(state_dict)

            if self.is_train:
                state_dict = torch.load(pjoin(self.save_path, f'optimizer/{filename}'), map_location=self.device)
                self.optimizer.load_state_dict(state_dict)


class MLPSkeleton(nn.Module):
    def __init__(self, layers, parents, activation=None, save_path=None, is_train=True, device=None, save_freq=2000,
                 threshold=2):
        super(MLPSkeleton, self).__init__()
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.save_freq = save_freq

        neighbor_list = find_neighbor_joint(parents, threshold)

        if save_path is None:
            save_path = './results/tmp/fc/'
        self.save_path = save_path

        self.layers = nn.ModuleList()

        self.is_train = is_train

        self.epoch_count = 0
        self.optimizer = None

        for i in range(1, len(layers)):
            model = [SkeletonLinear(neighbor_list, layers[i-1], layers[i])]
            if i < len(layers) - 1:
                if activation is None:
                    act = nn.LeakyReLU(negative_slope=0.2)
                else:
                    act = activation()
                model.append(act)
            self.layers.append(nn.Sequential(*model))

        self.model = self.layers

        os.makedirs(pjoin(save_path, 'model'), exist_ok=True)
        os.makedirs(pjoin(save_path, 'optimizer'), exist_ok=True)

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input

    def epoch(self):
        self.epoch_count += 1

    def set_optimizer(self, lr, optimizer=torch.optim.Adam):
        self.optimizer = optimizer(self.parameters(), lr=lr)

    def save_model(self, epoch=None):
        if not self.is_train:
            return
        if epoch is None:
            epoch = self.epoch_count

        if epoch % self.save_freq == 0:
            torch.save(self.layers.state_dict(), pjoin(self.save_path, 'model/%05d.pt' % epoch))
            torch.save(self.optimizer.state_dict(), pjoin(self.save_path, 'optimizer/%05d.pt' % epoch))

        torch.save(self.layers.state_dict(), pjoin(self.save_path, 'model/latest.pt'))
        torch.save(self.optimizer.state_dict(), pjoin(self.save_path, 'optimizer/latest.pt'))

    def load_model(self, epoch=None):
        if epoch is None:
            epoch = self.epoch_count

        if isinstance(epoch, str):
            state_dict = torch.load(epoch, map_location=self.device)
            self.layers.load_state_dict(state_dict)

        else:
            filename = ('%05d.pt' % epoch) if epoch != -1 else 'latest.pt'
            state_dict = torch.load(pjoin(self.save_path, f'model/{filename}'), map_location=self.device)
            self.layers.load_state_dict(state_dict)

            if self.is_train:
                state_dict = torch.load(pjoin(self.save_path, f'optimizer/{filename}'), map_location=self.device)
                self.optimizer.load_state_dict(state_dict)
