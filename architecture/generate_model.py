import torch
from models.transforms import aa2mat
from models.deformation import deform_with_offset
from models.kinematics import ForwardKinematics
from models.boundingbox import BoundingBox
from models.networks import MeshReprConv, MLPSkeleton


class GenerateModelBase:
    def __init__(self, use_collapse=True):
        self.models = {}

        self.criteria = torch.nn.MSELoss()
        self.criteria_basic = torch.nn.MSELoss()
        self.loss = None
        self.rec_model = None

        self.epoch_cnt = 0
        self.use_collapse = use_collapse

    def zero_grad(self):
        for model in self.models.values():
            if model is not None and model.optimizer is not None:
                model.optimizer.zero_grad()

    def optim_step(self):
        for model in self.models.values():
            if model is not None and model.optimizer is not None:
                model.optimizer.step()

    def set_optimizer(self, lr=1e-3):
        for model in self.models.values():
            model.set_optimizer(lr)

    def set_topology(self, topo_id):
        for model in self.models.values():
            if hasattr(model, 'prepare_topology') and callable(model.prepare_topology):
                model.prepare_topology(topo_id)
        if hasattr(self.criteria, 'set_mask') and callable(self.criteria.set_mask):
            self.criteria.set_mask(self.rec_model.v_mask)

    def save_model(self):
        for model in self.models.values():
            model.save_model()

    def load_model(self, epoch):
        for model in self.models.values():
            model.load_model(epoch=epoch)

    def epoch(self):
        for model in self.models.values():
            model.epoch()

        self.epoch_cnt += 1


def attention_pooling(tensor, att):
    """
    Corresponding to the Eq(1) in the paper.
    """
    latent = torch.matmul(att.transpose(-2, -1), tensor)
    coff = att.sum(dim=1).unsqueeze(-1)
    coff[coff < 1e-10] = 1
    latent = latent / coff
    return latent


class EnvelopeGenerate(GenerateModelBase):
    def __init__(self, geo: MeshReprConv, att: MeshReprConv, gen: MLPSkeleton, fk: ForwardKinematics, args):
        super(EnvelopeGenerate, self).__init__()
        self.models = {'geo': geo, 'att': att, 'gen': gen}

        self.args = args
        self.rec_model = geo # Use this model as log recorder
        self.fk = fk

    def forward_att(self, t_pose_mapped):
        self.models['att'].set_input(t_pose_mapped)
        att_logit = self.models['att'].forward()
        att = torch.softmax(att_logit, dim=-1)
        att_vert = self.models['att'].edge2vert(att)
        return att, att_vert

    def forward_geometry(self, t_pose_mapped):
        self.models['geo'].set_input(t_pose_mapped)
        deep_v = self.models['geo'].forward()
        return deep_v

    def forward(self, t_pose, pose, topo_id, pose_ee=None):
        self.set_topology(topo_id)
        self.t_pose_mapped = self.rec_model.apply_topo(t_pose)

        # Normalize
        if self.args.normalize:
            self.bb = BoundingBox(self.t_pose_mapped)
        else:
            self.bb = BoundingBox()

        self.t_pose_mapped = self.bb.normalize(self.t_pose_mapped)

        # Generate skinning weight (attention)
        self.att, self.att_vert = self.forward_att(self.t_pose_mapped)

        self.deep_v = self.forward_geometry(self.t_pose_mapped)

        # Deal with skeleton generation
        deep_s = attention_pooling(self.deep_v, self.att)
        deep_s = deep_s.reshape(deep_s.shape[0], -1)
        skeleton = self.models['gen'](deep_s)

        skeleton = skeleton.reshape(deep_s.shape[0], -1, 3)

        self.skeleton = skeleton = self.bb.denormalize_offset(skeleton)
        self.t_pose_mapped = self.bb.denormalize(self.t_pose_mapped)

        if pose is None:
            return

        # Generate deformed shape
        if pose_ee is not None:
            pose = torch.cat((pose, pose_ee), dim=0)
        pose = pose.reshape(pose.shape[0], -1, 3)

        self.local_mat = aa2mat(pose)

        if pose_ee is not None:
            self.local_mat_ee = self.local_mat[-pose_ee.shape[0]:]
            self.local_mat = self.local_mat[:-pose_ee.shape[0]]

        trans_b = self.fk.forward(self.local_mat, skeleton)
        self.res_verts = deform_with_offset(self.t_pose_mapped, self.att_vert, trans_b)

        if pose_ee is not None:
            trans_b_ee = self.fk.forward(self.local_mat_ee, skeleton)
            self.res_verts_ee = deform_with_offset(self.t_pose_mapped, self.att_vert, trans_b_ee)

        return self.res_verts

    def backward(self, gt_verts, gt_verts_ee=None, gt_root_loc=None, requires_backward=True):
        gt_verts = self.rec_model.apply_topo(gt_verts)
        self.loss = self.loss_vert = self.criteria(self.res_verts, gt_verts)

        if gt_verts_ee is not None:
            gt_verts_ee = self.rec_model.apply_topo(gt_verts_ee)
            self.loss_ee = self.criteria(self.res_verts_ee, gt_verts_ee)
            self.loss = self.loss + self.loss_ee * self.args.lambda_ee
        else:
            self.loss_ee = None

        if gt_root_loc is not None:
            self.loss_root = self.criteria(self.skeleton[:, 0], gt_root_loc)
            self.loss = self.loss + self.loss_root
        else:
            self.loss_root = None

        if requires_backward:
            self.rec_model.loss_recorder.add_scalar('loss', self.loss.item())
            self.rec_model.loss_recorder.add_scalar('loss_vert', self.loss_vert.item())
            if self.loss_ee is not None:
                self.rec_model.loss_recorder.add_scalar('loss_ee', self.loss_ee.item())
            if self.loss_root is not None:
                self.rec_model.loss_recorder.add_scalar('loss_root', self.loss_root.item())
            self.loss.backward()


class BlendShapesGenerate(GenerateModelBase):
    def __init__(self, geo: MeshReprConv, att: MeshReprConv, gen: MeshReprConv, bs, args,
                 fk: ForwardKinematics):
        super(BlendShapesGenerate, self).__init__()

        self.models = {'geo': geo, 'att': att, 'gen': gen}
        if bs is not None:
            self.models['bs'] = bs

        self.args = args
        self.rec_model = geo
        self.fk = fk

    def forward_att(self, t_pose_mapped):
        self.models['att'].set_input(t_pose_mapped)
        att_logit = self.models['att'].forward()
        att = torch.softmax(att_logit, dim=-1)
        att_vert = self.models['att'].edge2vert(att)
        return att, att_vert

    def forward_geometry(self, t_pose_mapped):
        self.models['geo'].set_input(t_pose_mapped)

        deep_v = self.models['geo'].forward()
        return deep_v

    def forward_gen(self, latent):
        self.models['gen'].set_input(latent, convert2edge=False)
        basis_edge = self.models['gen'].forward()
        basis_vert = self.models['gen'].edge2vert(basis_edge)
        return basis_vert

    def forward_residual(self, mat, basis, weight):
        import time
        self.models['bs'].set_mask(weight)
        a = time.time()
        disp = self.models['bs'](mat, basis)
        b = time.time()
        # print(f'Evaluate {mat.shape[0]} frames on {mat.device} takes {b - a}s')
        return disp

    def forward(self, t_pose, pose, topo_id, skeletons=None, basis_only=False):
        self.loss = 0
        self.set_topology(topo_id)
        self.t_pose_mapped = self.rec_model.apply_topo(t_pose)

        # Normalize
        if self.args.normalize:
            self.bb = BoundingBox(self.t_pose_mapped)
        else:
            self.bb = BoundingBox()

        self.t_pose_mapped = self.bb.normalize(self.t_pose_mapped)

        # Get skinning weight
        with torch.no_grad():
            self.att, self.att_vert = self.forward_att(self.t_pose_mapped)
        self.att = self.att.detach()
        self.att_vert = self.att_vert.detach()

        # Generate blend shapes
        self.deep_v = self.forward_geometry(self.t_pose_mapped)
        offset_input = torch.cat((self.deep_v, self.att), dim=-1)
        basis = self.forward_gen(offset_input)
        basis = basis.reshape(basis.shape[:-1] + (-1, 3))  # (batch_size, n_vert, n_basis, 3)
        self.basis = basis
        self.res_verts = basis

        self.t_pose_mapped = self.bb.denormalize(self.t_pose_mapped)
        self.basis = self.bb.denormalize_basis(self.basis)

        if basis_only:
            return self.basis

        # Generate deformed shape
        self.res_verts = []
        self.offsets = []
        local_mat = aa2mat(pose.reshape(pose.shape[0], -1, 3))

        for i in range(t_pose.shape[0]):
            offset = self.forward_residual(local_mat, self.basis[i], self.att_vert[i])
            self.offsets.append(offset)
            if skeletons is None:
                continue
            trans_b = self.fk.forward(local_mat, skeletons[[i]])
            res_vert = deform_with_offset(self.t_pose_mapped[[i]].expand(pose.shape[0], -1, -1), self.att_vert[[i]],
                                          trans_b, offset)
            self.res_verts.append(res_vert[None, ...])
        if skeletons is not None:
            self.res_verts = torch.cat(self.res_verts, dim=0)
        return self.res_verts

    def backward(self, gt_basis=None, gt_verts=None, requires_backward=True):
        if gt_basis is not None:
            gt_basis = self.rec_model.apply_topo(gt_basis)
            self.loss = self.criteria(self.basis, gt_basis)
        elif gt_verts is not None:
            shape0 = gt_verts.shape[0]
            gt_verts = gt_verts.reshape((-1,) + gt_verts.shape[2:])
            gt_verts = self.rec_model.apply_topo(gt_verts)
            gt_verts = gt_verts.reshape((shape0, -1) + gt_verts.shape[1:])
            self.loss = self.criteria(self.res_verts, gt_verts)

        if requires_backward:
            self.rec_model.loss_recorder.add_scalar('loss', self.loss.item())
            self.loss.backward()
