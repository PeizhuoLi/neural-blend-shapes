import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SkeletonConv(nn.Module):
    def __init__(self, neighbour_list, in_channels, out_channels, kernel_size, joint_num, stride=1, padding=0,
                 bias=True, padding_mode='zeros', add_offset=False, in_offset_channel=0):
        self.in_channels_per_joint = in_channels // joint_num
        self.out_channels_per_joint = out_channels // joint_num
        if in_channels % joint_num != 0 or out_channels % joint_num != 0:
            raise Exception('BAD')
        super(SkeletonConv, self).__init__()

        if padding_mode == 'zeros': padding_mode = 'constant'
        if padding_mode == 'reflection': padding_mode = 'reflect'

        self.expanded_neighbour_list = []
        self.expanded_neighbour_list_offset = []
        self.neighbour_list = neighbour_list
        self.add_offset = add_offset
        self.joint_num = joint_num

        self.stride = stride
        self.dilation = 1
        self.groups = 1
        self.padding = padding
        self.padding_mode = padding_mode
        self._padding_repeated_twice = (padding, padding)

        for neighbour in neighbour_list:
            expanded = []
            for k in neighbour:
                for i in range(self.in_channels_per_joint):
                    expanded.append(k * self.in_channels_per_joint + i)
            self.expanded_neighbour_list.append(expanded)

        if self.add_offset:
            self.offset_enc = SkeletonLinear(neighbour_list, in_offset_channel * len(neighbour_list), out_channels)

            for neighbour in neighbour_list:
                expanded = []
                for k in neighbour:
                    for i in range(add_offset):
                        expanded.append(k * in_offset_channel + i)
                self.expanded_neighbour_list_offset.append(expanded)

        self.weight = torch.zeros(out_channels, in_channels, kernel_size)
        if bias:
            self.bias = torch.zeros(out_channels)
        else:
            self.register_parameter('bias', None)

        self.mask = torch.zeros_like(self.weight)
        for i, neighbour in enumerate(self.expanded_neighbour_list):
            self.mask[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1), neighbour, ...] = 1
        self.mask = nn.Parameter(self.mask, requires_grad=False)

        self.description = 'SkeletonConv(in_channels_per_armature={}, out_channels_per_armature={}, kernel_size={}, ' \
                           'joint_num={}, stride={}, padding={}, bias={})'.format(
            in_channels // joint_num, out_channels // joint_num, kernel_size, joint_num, stride, padding, bias
        )

        self.reset_parameters()

    def reset_parameters(self):
        for i, neighbour in enumerate(self.expanded_neighbour_list):
            """ Use temporary variable to avoid assign to copy of slice, which might lead to un expected result """
            tmp = torch.zeros_like(self.weight[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1),
                                   neighbour, ...])
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1),
                        neighbour, ...] = tmp
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weight[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1), neighbour, ...])
                bound = 1 / math.sqrt(fan_in)
                tmp = torch.zeros_like(
                    self.bias[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1)])
                nn.init.uniform_(tmp, -bound, bound)
                self.bias[self.out_channels_per_joint * i: self.out_channels_per_joint * (i + 1)] = tmp

        self.weight = nn.Parameter(self.weight)
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias)

    def set_offset(self, offset):
        if not self.add_offset: raise Exception('Wrong Combination of Parameters')
        self.offset = offset.reshape(offset.shape[0], -1)

    def forward(self, input):
        weight_masked = self.weight * self.mask
        res = F.conv1d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                       weight_masked, self.bias, self.stride,
                       0, self.dilation, self.groups)

        if self.add_offset:
            offset_res = self.offset_enc(self.offset)
            offset_res = offset_res.reshape(offset_res.shape + (1, ))
            res += offset_res / 100
        return res


class SkeletonLinear(nn.Module):
    def __init__(self, neighbour_list, in_channels, out_channels, extra_dim1=False):
        super(SkeletonLinear, self).__init__()
        self.neighbour_list = neighbour_list
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_channels_per_joint = in_channels // len(neighbour_list)
        self.out_channels_per_joint = out_channels // len(neighbour_list)
        self.extra_dim1 = extra_dim1
        self.expanded_neighbour_list = []

        for neighbour in neighbour_list:
            expanded = []
            for k in neighbour:
                for i in range(self.in_channels_per_joint):
                    expanded.append(k * self.in_channels_per_joint + i)
            self.expanded_neighbour_list.append(expanded)

        self.weight = torch.zeros(out_channels, in_channels)
        self.mask = torch.zeros(out_channels, in_channels)
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for i, neighbour in enumerate(self.expanded_neighbour_list):
            tmp = torch.zeros_like(
                self.weight[i*self.out_channels_per_joint: (i + 1)*self.out_channels_per_joint, neighbour]
            )
            self.mask[i*self.out_channels_per_joint: (i + 1)*self.out_channels_per_joint, neighbour] = 1
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[i*self.out_channels_per_joint: (i + 1)*self.out_channels_per_joint, neighbour] = tmp

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.weight = nn.Parameter(self.weight)
        self.mask = nn.Parameter(self.mask, requires_grad=False)

    def forward(self, input):
        input = input.reshape(input.shape[0], -1)
        weight_masked = self.weight * self.mask
        res = F.linear(input, weight_masked, self.bias)
        if self.extra_dim1: res = res.reshape(res.shape + (1,))
        return res


class SkeletonPoolJoint(nn.Module):
    def __init__(self, topology, pooling_mode, channels_per_joint, last_pool=False):
        super(SkeletonPoolJoint, self).__init__()

        if pooling_mode != 'mean':
            raise Exception('Unimplemented pooling mode in matrix_implementation')

        self.joint_num = len(topology)
        self.parent = topology
        self.pooling_list = []
        self.pooling_mode = pooling_mode

        self.pooling_map = [-1 for _ in range(len(self.parent))]
        self.child = [-1 for _ in range(len(self.parent))]
        children_cnt = [0 for _ in range(len(self.parent))]
        for x, pa in enumerate(self.parent):
            if pa < 0: continue
            children_cnt[pa] += 1
            self.child[pa] = x
        self.pooling_map[0] = 0
        for x in range(len(self.parent)):
            if children_cnt[x] == 0 or (children_cnt[x] == 1 and children_cnt[self.child[x]] > 1):
                while children_cnt[x] <= 1:
                    pa = self.parent[x]
                    if last_pool:
                        seq = [x]
                        while pa != -1 and children_cnt[pa] == 1:
                            seq = [pa] + seq
                            x = pa
                            pa = self.parent[x]
                        self.pooling_list.append(seq)
                        break
                    else:
                        if pa != -1 and children_cnt[pa] == 1:
                            self.pooling_list.append([pa, x])
                            x = self.parent[pa]
                        else:
                            self.pooling_list.append([x, ])
                            break
            elif children_cnt[x] > 1:
                self.pooling_list.append([x, ])

        self.description = 'SkeletonPool(in_joint_num={}, out_joint_num={})'.format(
            len(topology), len(self.pooling_list),
        )

        self.pooling_list.sort(key=lambda x:x[0])
        for i, a in enumerate(self.pooling_list):
            for j in a:
                self.pooling_map[j] = i

        self.output_joint_num = len(self.pooling_list)
        self.new_topology = [-1 for _ in range(len(self.pooling_list))]
        for i, x in enumerate(self.pooling_list):
            if i < 1: continue
            self.new_topology[i] = self.pooling_map[self.parent[x[0]]]

        self.weight = torch.zeros(len(self.pooling_list) * channels_per_joint, self.joint_num * channels_per_joint)

        for i, pair in enumerate(self.pooling_list):
            for j in pair:
                for c in range(channels_per_joint):
                    self.weight[i * channels_per_joint + c, j * channels_per_joint + c] = 1.0 / len(pair)

        self.weight = nn.Parameter(self.weight, requires_grad=False)

    def forward(self, input: torch.Tensor):
        return torch.matmul(self.weight, input.unsqueeze(-1)).squeeze(-1)



class SkeletonPool(nn.Module):
    def __init__(self, edges, pooling_mode, channels_per_edge, last_pool=False):
        super(SkeletonPool, self).__init__()

        if pooling_mode != 'mean':
            raise Exception('Unimplemented pooling mode in matrix_implementation')

        self.channels_per_edge = channels_per_edge
        self.pooling_mode = pooling_mode
        self.edge_num = len(edges) + 1
        self.seq_list = []
        self.pooling_list = []
        self.new_edges = []
        degree = [0] * 100

        for edge in edges:
            degree[edge[0]] += 1
            degree[edge[1]] += 1

        def find_seq(j, seq):
            nonlocal self, degree, edges

            if degree[j] > 2 and j != 0:
                self.seq_list.append(seq)
                seq = []

            if degree[j] == 1:
                self.seq_list.append(seq)
                return

            for idx, edge in enumerate(edges):
                if edge[0] == j:
                    find_seq(edge[1], seq + [idx])

        find_seq(0, [])
        for seq in self.seq_list:
            if last_pool:
                self.pooling_list.append(seq)
                continue
            if len(seq) % 2 == 1:
                self.pooling_list.append([seq[0]])
                self.new_edges.append(edges[seq[0]])
                seq = seq[1:]
            for i in range(0, len(seq), 2):
                self.pooling_list.append([seq[i], seq[i + 1]])
                self.new_edges.append([edges[seq[i]][0], edges[seq[i + 1]][1]])

        # add global position
        self.pooling_list.append([self.edge_num - 1])

        self.description = 'SkeletonPool(in_edge_num={}, out_edge_num={})'.format(
            len(edges), len(self.pooling_list)
        )

        self.weight = torch.zeros(len(self.pooling_list) * channels_per_edge, self.edge_num * channels_per_edge)

        for i, pair in enumerate(self.pooling_list):
            for j in pair:
                for c in range(channels_per_edge):
                    self.weight[i * channels_per_edge + c, j * channels_per_edge + c] = 1.0 / len(pair)

        self.weight = nn.Parameter(self.weight, requires_grad=False)

    def forward(self, input: torch.Tensor):
        return torch.matmul(self.weight, input)


class SkeletonUnpool(nn.Module):
    def __init__(self, pooling_list, channels_per_edge):
        super(SkeletonUnpool, self).__init__()
        self.pooling_list = pooling_list
        self.input_joint_num = len(pooling_list)
        self.output_joint_num = 0
        self.channels_per_edge = channels_per_edge
        for t in self.pooling_list:
            self.output_joint_num += len(t)

        self.description = 'SkeletonUnpool(in_joint_num={}, out_joint_num={})'.format(
            self.input_joint_num, self.output_joint_num,
        )

        self.weight = torch.zeros(self.output_joint_num * channels_per_edge, self.input_joint_num * channels_per_edge)

        for i, pair in enumerate(self.pooling_list):
            for j in pair:
                for c in range(channels_per_edge):
                    self.weight[j * channels_per_edge + c, i * channels_per_edge + c] = 1

        self.weight = nn.Parameter(self.weight)
        self.weight.requires_grad_(False)

    def forward(self, input: torch.Tensor):
        return torch.matmul(self.weight, input.unsqueeze(-1)).squeeze(-1)


def find_neighbor_joint(parents, threshold):
    n_joint = len(parents)
    dist_mat = np.empty((n_joint, n_joint), dtype=np.int)
    dist_mat[:, :] = 100000
    for i, p in enumerate(parents):
        dist_mat[i, i] = 0
        if i != 0:
            dist_mat[i, p] = dist_mat[p, i] = 1

    """
    Floyd's algorithm
    """
    for k in range(n_joint):
        for i in range(n_joint):
            for j in range(n_joint):
                dist_mat[i, j] = min(dist_mat[i, j], dist_mat[i, k] + dist_mat[k, j])

    neighbor_list = []
    for i in range(n_joint):
        neighbor = []
        for j in range(n_joint):
            if dist_mat[i, j] <= threshold:
                neighbor.append(j)
        neighbor_list.append(neighbor)

    return neighbor_list
