import argparse
import pickle
import sys
import os


class BaseOptionParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--save_path', type=str, default='./results/tmp/')
        self.parser.add_argument('--device', type=str, default='cpu')
        self.parser.add_argument('--use_weight_as_att', type=int, default=0)
        self.parser.add_argument('--batch_size', type=int, default=5)
        self.parser.add_argument('--num_layers', type=int, default=4)
        self.parser.add_argument('--base', type=int, default=64)
        self.parser.add_argument('--uniform_angle', type=int, default=0)
        self.parser.add_argument('--debug', type=int, default=0)
        self.parser.add_argument('--att_base', type=int, default=32)
        self.parser.add_argument('--att_save_freq', type=int, default=2000)
        self.parser.add_argument('--pool_ratio', type=float, default=2)
        self.parser.add_argument('--pool_method', type=str, default='mean')
        self.parser.add_argument('--cont', type=int, default=0)
        self.parser.add_argument('--skeleton_aware', type=int, default=1)
        self.parser.add_argument('--topo_augment', type=int, default=1)
        self.parser.add_argument('--offset_init_div', type=float, default=1000)
        self.parser.add_argument('--save_freq', type=int, default=2000)
        self.parser.add_argument('--normalize', type=int, default=0)
        self.parser.add_argument('--basis_per_bone', type=int, default=3)
        self.parser.add_argument('--basis_only', type=int, default=0)
        self.parser.add_argument('--sanity_check', type=int, default=0)
        self.parser.add_argument('--pose_batch_size', type=int, default=200)
        self.parser.add_argument('--coff_init_div', type=float, default=1)

    def parse_args(self, args_str=None):
        return self.parser.parse_args(args_str)

    def get_parser(self):
        return self.parser

    def save(self, filename, args_str=None):
        if args_str is None:
            args_str = ' '.join(sys.argv[1:])
        path = '/'.join(filename.split('/')[:-1])
        os.makedirs(path, exist_ok=True)
        with open(filename, 'w') as file:
            file.write(args_str)

    def load(self, filename):
        with open(filename, 'r') as file:
            args_str = file.readline()
        return self.parse_args(args_str.split())


class TrainingOptionParser(BaseOptionParser):
    def __init__(self):
        super(TrainingOptionParser, self).__init__()
        self.parser.add_argument('--lr', type=float, default=1e-5)
        self.parser.add_argument('--lr_coff', type=float, default=1e-3)
        self.parser.add_argument('--lr_att', type=float, default=5e-4)


class TestOptionParser(BaseOptionParser):
    def __init__(self):
        super(TestOptionParser, self).__init__()
        self.save_path = None
        self.parser.add_argument('--eval_smpl', type=int, default=0)
        self.parser.add_argument('--eval_mixamo', type=int, default=0)
        self.parser.add_argument('--eval_garment', type=int, default=0)
        self.parser.add_argument('--eval_epoch', type=int, default=-1)
        self.parser.add_argument('--eval_outside', type=int, default=0)

    def parse_args(self, args_str=None):
        res, _ = self.parser.parse_known_args(args_str)
        if self.save_path is None:
            self.save_path = res.save_path
        else:
            res.save_path = self.save_path
        return res
