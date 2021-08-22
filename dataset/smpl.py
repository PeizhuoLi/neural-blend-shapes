import numpy as np
import pickle
import torch
from torch.nn import Module
import os


class SMPL_Layer(Module):
    def __init__(self, model_root='./dataset/smpl_model', gender='neutral'):
        super(SMPL_Layer, self).__init__()

        if gender == 'neutral':
            self.model_path = os.path.join(model_root, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
        elif gender == 'female':
            self.model_path = os.path.join(model_root, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        elif gender == 'male':
            self.model_path = os.path.join(model_root, 'basicModel_m_lbs_10_207_0_v1.0.0.pkl')

        with open(self.model_path, 'rb') as f:
            params = pickle.load(f, encoding='latin1')
        # self.J_regressor = torch.from_numpy(
        #     np.array(params['J_regressor'].todense())
        # ).type(torch.float)
        self.register_buffer('J_regressor', torch.from_numpy(
            np.array(params['J_regressor'].todense())
        ).type(torch.float))
        if 'joint_regressor' in params.keys():
            # self.joint_regressor = torch.from_numpy(
            #     np.array(params['joint_regressor'].T.todense())
            # ).type(torch.float)
            self.register_buffer('joint_regressor', torch.from_numpy(
                np.array(params['joint_regressor'].T.todense())
            ).type(torch.float))
        else:
            # self.joint_regressor = torch.from_numpy(
            #     np.array(params['J_regressor'].todense())
            # ).type(torch.float)
            self.register_buffer('joint_regressor', torch.from_numpy(
                np.array(params['J_regressor'].todense())
            ).type(torch.float))
        # self.weights = torch.from_numpy(params['weights']).type(torch.float)
        self.register_buffer('weights', torch.from_numpy(params['weights']).type(torch.float))
        # self.posedirs = torch.from_numpy(params['posedirs']).type(torch.float)
        self.register_buffer('posedirs', torch.from_numpy(params['posedirs']).type(torch.float))
        # self.v_template = torch.from_numpy(params['v_template']).type(torch.float)
        self.register_buffer('v_template', torch.from_numpy(params['v_template']).type(torch.float))
        # self.shapedirs = torch.from_numpy(params['shapedirs']).type(torch.float)
        self.register_buffer('shapedirs', torch.tensor(params['shapedirs'].r).type(torch.float))
        self.kintree_table = params['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents
        self.faces = params['f']

        self.num_joints = len(parents)  # 24
        self.num_verts = self.v_template.shape[0]

    @staticmethod
    def rodrigues(r):
        """
        Rodrigues' rotation formula that turns axis-angle tensor into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size * angle_num, 3, 3].

        """
        eps = 1e-8
        theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
        theta_dim = theta.shape[0]
        r_hat = r / theta
        cos = torch.cos(theta)
        z_stick = torch.zeros(theta_dim, dtype=torch.float).to(r.device)
        m = torch.stack(
            (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
             -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
        m = torch.reshape(m, (-1, 3, 3))
        i_cube = (torch.eye(3, dtype=torch.float).unsqueeze(dim=0)
                  + torch.zeros((theta_dim, 3, 3), dtype=torch.float)).to(r.device)
        A = r_hat.permute(0, 2, 1)
        dot = torch.matmul(A, r_hat)
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R

    @staticmethod
    def with_zeros(x):
        """
        Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

        Parameter:
        ---------
        x: Tensor to be appended.

        Return:
        ------
        Tensor after appending of shape [4,4]

        """
        ones = torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float
        ).expand(x.shape[0], -1, -1).to(x.device)
        ret = torch.cat((x, ones), dim=1)
        return ret

    @staticmethod
    def pack(x):
        """
        Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

        Parameter:
        ----------
        x: A tensor of shape [batch_size, 4, 1]

        Return:
        ------
        A tensor of shape [batch_size, 4, 4] after appending.

        """
        zeros43 = torch.zeros(
            (x.shape[0], x.shape[1], 4, 3), dtype=torch.float).to(x.device)
        ret = torch.cat((zeros43, x), dim=3)
        return ret

    def save_obj(self, filename, verts):
        with open(filename, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def forward(self, pose, betas, trans=None, simplify=False):

        """
              Construct a compute graph that takes in parameters and outputs a tensor as
              model vertices. Face indices are also returned as a numpy ndarray.

              20190128: Add batch support.

              Parameters:
              ---------
              pose: Also known as 'theta', an [N, 24, 3] tensor indicating child joint rotation
              relative to parent joint. For root joint it's global orientation.
              Represented in a axis-angle format.

              betas: Parameter for model shape. A tensor of shape [N, 10] as coefficients of
              PCA components. Only 10 components were released by SMPL author.

              trans: Global translation tensor of shape [N, 3].

              Return:
              ------
              A 3-D tensor of [N * 6890 * 3] for vertices,
              and the corresponding [N * 19 * 3] joint positions.

        """
        batch_num = betas.shape[0]
        id_to_col = {self.kintree_table[1, i]: i
                     for i in range(self.kintree_table.shape[1])}
        parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }
        v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [2])) + self.v_template
        J = torch.matmul(self.J_regressor, v_shaped)
        R_cube_big = self.rodrigues(pose.reshape(-1, 1, 3)).reshape(batch_num, -1, 3, 3)

        device = self.weights.device

        if simplify:
            v_posed = v_shaped
        else:
            R_cube = R_cube_big[:, 1:, :, :]
            I_cube = (torch.eye(3, dtype=torch.float).unsqueeze(dim=0) + \
                      torch.zeros((batch_num, R_cube.shape[1], 3, 3), dtype=torch.float)).to(device)
            lrotmin = (R_cube - I_cube).reshape(batch_num, -1, 1).squeeze(dim=2)
            v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=([1], [2]))

        results = []
        results.append(
            self.with_zeros(torch.cat((R_cube_big[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2))
        )
        for i in range(1, self.kintree_table.shape[1]):
            results.append(
                torch.matmul(
                    results[parent[i]],
                    self.with_zeros(
                        torch.cat(
                            (R_cube_big[:, i], torch.reshape(J[:, i, :] - J[:, parent[i], :], (-1, 3, 1))),
                            dim=2
                        )
                    )
                )
            )

        stacked = torch.stack(results, dim=1)
        results = stacked - \
                  self.pack(
                      torch.matmul(
                          stacked,
                          torch.reshape(
                              torch.cat((J, torch.zeros((batch_num, 24, 1), dtype=torch.float).to(device)),
                                        dim=2),
                              (batch_num, 24, 4, 1)
                          )
                      )
                  )

        T = torch.tensordot(results, self.weights, dims=([1], [1])).permute(0, 3, 1, 2)

        rest_shape_h = torch.cat(
            (v_posed, torch.ones((batch_num, v_posed.shape[1], 1), dtype=torch.float).to(device)), dim=2
        )
        v = torch.matmul(T, torch.reshape(rest_shape_h, (batch_num, -1, 4, 1)))
        v = torch.reshape(v, (batch_num, -1, 4))[:, :, :3]

        result = v if trans is None else v + torch.reshape(trans, (batch_num, 1, 3))

        joints = torch.tensordot(result, self.joint_regressor, dims=([1], [1])).transpose(1, 2)
        return result.detach(), joints

    def forward_lbs(self, poses, shapes=None, v_offsets=0):
        if shapes is None:
            shapes = torch.zeros((poses.shape[0], 10), device=poses.device)
        return self.forward(poses, shapes, simplify=True)[0] + v_offsets

    def pose_blendshapes(self, pose):
        device = pose.device
        batch_num = pose.shape[0]
        R_cube_big = self.rodrigues(pose.view(-1, 1, 3)).reshape(batch_num, -1, 3, 3)

        R_cube = R_cube_big[:, 1:, :, :]
        I_cube = (torch.eye(3, dtype=torch.float).unsqueeze(dim=0) +
                  torch.zeros((batch_num, R_cube.shape[1], 3, 3), dtype=torch.float)).to(device)
        lrotmin = (R_cube - I_cube).reshape(batch_num, -1, 1).squeeze(dim=2)
        return torch.tensordot(lrotmin, self.posedirs, dims=([1], [2]))

    def get_offset(self, shapes=torch.zeros((1, 10))):
        batch_size = shapes.shape[0]
        parent_smpl = self.kintree_parents
        t_pose, j_loc = self.forward(torch.zeros((batch_size, 24 * 3), device=shapes.device), shapes)
        for i in list(range(len(parent_smpl)))[::-1]:
            if i == 0:
                break
            p = parent_smpl[i]
            j_loc[:, i] -= j_loc[:, p]
        offset = j_loc
        return offset
