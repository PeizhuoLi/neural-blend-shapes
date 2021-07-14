import argparse
import pickle
import sys
import os


class BaseOptionParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--save_path', type=str, default='./results/tmp/')
        self.parser.add_argument('--device', type=str, default='cpu')
        self.parser.add_argument('--num_layers', type=int, default=4)
        self.parser.add_argument('--base', type=int, default=64)
        self.parser.add_argument('--att_base', type=int, default=64)
        self.parser.add_argument('--pool_ratio', type=float, default=0.2)
        self.parser.add_argument('--pool_method', type=str, default='max')
        self.parser.add_argument('--skeleton_aware', type=int, default=1)
        self.parser.add_argument('--offset_init_div', type=float, default=1000)
        self.parser.add_argument('--normalize', type=int, default=0)
        self.parser.add_argument('--basis_per_bone', type=int, default=9)
        self.parser.add_argument('--pose_batch_size', type=int, default=200)
        self.parser.add_argument('--debug', type=int, default=0)

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
        self.parser.add_argument('--lr_att', type=float, default=2e-4)
        self.parser.add_argument('--batch_size', type=int, default=5)
        self.parser.add_argument('--envelope', type=int, default=0)
        self.parser.add_argument('--residual', type=int, default=0)
        self.parser.add_argument('--num_epoch', type=int, default=16000)
        self.parser.add_argument('--topo_augment', type=int, default=1)
        self.parser.add_argument('--fast_train', type=int, default=1)
        self.parser.add_argument('--ee_order', type=int, default=1)
        self.parser.add_argument('--att_save_freq', type=int, default=2000)
        self.parser.add_argument('--lambda_ee', type=float, default=5.)
        self.parser.add_argument('--save_freq', type=int, default=5000)
        self.parser.add_argument('--cont', type=int, default=0)
        self.parser.add_argument('--ee_factor', type=float, default=1.)
        self.parser.add_argument('--ee_uniform', type=int, default=0)
