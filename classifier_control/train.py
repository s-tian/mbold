import matplotlib;
import torch

matplotlib.use('Agg');
import argparse
import os
import time
from shutil import copy
import datetime
import imp
from tensorflow.contrib.training import HParams
from tensorboardX import SummaryWriter
import numpy as np
from torch import autograd
from torch.optim import Adam, SGD
from functools import partial

from classifier_control.classifier.utils.general_utils import AverageMeter, RecursiveAverageMeter, map_dict
from classifier_control.classifier.utils.checkpointer import CheckpointHandler, save_cmd, save_git, get_config_path
from classifier_control.classifier.utils.general_utils import AttrDict

from classifier_control.classifier.datasets.data_loader import FixLenVideoDataset

from classifier_control.classifier.utils.trainer_base import BaseTrainer


def save_checkpoint(state, folder, filename='checkpoint.pth'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))


def get_exp_dir():
    return os.environ['VMPC_EXP'] + '/classifier_control/'


def datetime_str():
    return datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")


def make_path(exp_dir, conf_path, prefix, make_new_dir):
    # extract the subfolder structure from config path
    path = conf_path.split('experiments/', 1)[1]
    if make_new_dir:
        prefix += datetime_str()
    base_path = os.path.join(exp_dir,  '/'.join(str.split(path, '/')[:-1]))
    return os.path.join(base_path, prefix) if prefix else base_path


def set_seeds(seed):
    """Sets all seeds and disables non-determinism in cuDNN backend."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


class ModelTrainer(BaseTrainer):
    def __init__(self):
        self.batch_idx = 0
        
        ## Set up params
        args, conf_module, conf, model_conf, data_conf, exp_dir, conf_path = self.get_configs()
        
        self._hp = self._default_hparams()
        self.override_defaults(conf)  # override defaults with config file

        self._hp.set_hparam('exp_path', make_path(exp_dir, args.path, args.prefix, args.new_dir))
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'events')
        print('using log dir: ', log_dir)
        
        self.run_testmetrics = args.metric
        if args.deterministic: set_seeds(self._hp.seed)
        
        ## Log
        print('Writing to the experiment directory: {}'.format(self._hp.exp_path))
        if not os.path.exists(self._hp.exp_path):
            os.makedirs(self._hp.exp_path)

        save_cmd(self._hp.exp_path)
        save_git(self._hp.exp_path)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')

        ## Buld dataset, model. logger, etc.
        writer = SummaryWriter(log_dir)

        model_conf['batch_size'] = self._hp.batch_size
        model_conf['device'] = self.device.type
        model_conf['data_conf'] = data_conf
        
        def build_phase(logger, ModelClass, phase):
            logger = logger(log_dir, summary_writer=writer)
            model = ModelClass(model_conf, logger)
            model.to(self.device)
            model.device = self.device
            if phase is not 'test':
                loader = FixLenVideoDataset(self._hp.data_dir, model._hp, data_conf, phase, shuffle=True).get_data_loader(self._hp.batch_size)
                return model, loader
            else:
                return model
        self.model, self.train_loader = build_phase(self._hp.logger, self._hp.model, 'train')
        self.model_val, self.val_loader = build_phase(self._hp.logger, self._hp.model, 'val')
        if self._hp.model_test is not None:
            self.model_test = build_phase(self._hp.logger, self._hp.model_test, 'test')

        if 'add_negative_sample' in self.model._hp and self.model._hp.add_negative_sample and \
            'negative_sample_type' in self.model._hp and self.model._hp.negative_sample_type == 'nn_idx':
            from classifier_control.classifier.models.utils.nn_idx import NNIndex
            self.model.nn_idx = self.model_val.nn_idx = NNIndex(self.train_loader, self.args.gpu, self.model._hp.nn_dim)

        self.model.init_optimizers(self._hp)
        self._hp.mpar = self.model._hp

        # TODO clean up resuming
        self.global_step = 0
        start_epoch = 0
        if args.resume:
            start_epoch = self.resume(args.resume)
        
        if args.val_sweep:
            epochs = CheckpointHandler.get_epochs(os.path.join(self._hp.exp_path, 'weights'))
            for epoch in list(sorted(epochs))[::4]:
                self.resume(epoch)
                self.val()
            return

        if self.args.online:
            self.start_epoch = start_epoch
            return

        ## Train
        if args.train:
            self.train(start_epoch)
        else:
            self.val()

    def resume(self, ckpt):
        weights_file = CheckpointHandler.get_resume_ckpt_file(ckpt, os.path.join(self._hp.exp_path, 'weights'))
        self.global_step, start_epoch, _ = \
            CheckpointHandler.load_weights(weights_file, self.model,
                                           load_step_and_opt=True, optimizer=self.model.optimizer,
                                           dataset_length=len(self.train_loader) * self._hp.batch_size,
                                           strict=self.args.strict_weight_loading)
        self.model.to(self.model.device)
        return start_epoch
    
    def get_configs(self):
        self.args = args = self.get_trainer_args()
        exp_dir = get_exp_dir()
        # conf_path = get_config_path(args.path)
        # print('loading from the config file {}'.format(conf_path))

        conf_path = os.path.abspath(args.path)
        conf_module = imp.load_source('conf', args.path)
        conf = conf_module.configuration
        model_conf = conf_module.model_config
        
        try:
            data_conf = conf_module.data_config
        except AttributeError:
            data_conf_file = imp.load_source('dataset_spec',os.path.join(AttrDict(conf).data_dir, 'dataset_spec.py'))
            data_conf = AttrDict()
            data_conf.dataset_spec = AttrDict(data_conf_file.dataset_spec)

        if args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

        return args, conf_module, conf, model_conf, data_conf, exp_dir, conf_path
    
    def get_trainer_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("path", help="path to the config file directory")
        
        # Folder settings
        parser.add_argument("--prefix", help="experiment prefix, if given creates subfolder in experiment directory")
        parser.add_argument('--new_dir', default=False, type=int, help='If True, concat datetime string to exp_dir.')

        # Running protocol
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--train', default=True, type=int,
                            help='if False, will run one validation epoch')
        parser.add_argument('--online', default=False, type=int,
                            help='online or not')
        parser.add_argument('--test_prediction', default=True, type=int,
                            help="if False, prediction isn't run at validation time")
        parser.add_argument('--metric', default=False, type=int,
                            help='if True, run test metrics')
        parser.add_argument('--val_sweep', default=False, type=int,
                            help='if True, runs validation on all existing model checkpoints')
        
        # Misc
        parser.add_argument('--gpu', default=-1, type=int,
                            help='gpu index to use. NOTE THAT THIS SETS CUDA_VISIBLE_DEVICES!')
        parser.add_argument('--strict_weight_loading', default=True, type=int,
                            help='if True, uses strict weight loading function')
        parser.add_argument('--deterministic', default=False, type=int,
                            help='if True, sets fixed seeds for torch and numpy')
        parser.add_argument('--imepoch', default=4, type=int,
                            help='number of image loggings per epoch')
        parser.add_argument('--val_data_size', default=-1, type=int,
                            help='number of sequences in the validation set. If -1, the full dataset is used')

        return parser.parse_args()
    
    def _default_hparams(self):
        # put new parameters in here:
        default_dict = {
            'model': None,
            'model_test': None,
            'logger': None,
            'logger_test': None,
            'data_dir': None, # directory where dataset is in
            'batch_size': 64,
            'mpar': None,   # model parameters
            'data_conf': None,   # model parameters
            'exp_path': None,  # Path to the folder with experiments
            'num_epochs': 200,
            'log_every': 5,
            'epoch_cycles_train': 1,
            'mujoco_xml': None,
            'optimizer': 'adam',    # supported: 'adam', 'rmsprop', 'sgd'
            'lr': 1e-3,
            'momentum': 0,      # momentum in RMSProp / SGD optimizer
            'adam_beta': 0.9,       # beta1 param in Adam
            'seed': 0,
        }
        # add new params to parent params
        parent_params = HParams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params
    
    def train(self, start_epoch):
        for epoch in range(start_epoch, self._hp.num_epochs):
            if epoch > start_epoch:
                self.val()
            if epoch % self._hp.log_every == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.model.optimizer.state_dict(),
                },  os.path.join(self._hp.exp_path, 'weights'), CheckpointHandler.get_ckpt_name(epoch))
                self.model.dump_params(self._hp.exp_path)
            self.train_epoch(epoch)

    @property
    def log_images_now(self):
        return self.global_step % self.log_images_interval == 0
    
    @property
    def log_outputs_now(self):
        return self.global_step % self.log_outputs_interval == 0 or self.global_step % self.log_images_interval == 0

    def train_epoch(self, epoch):
        self.model.train()
        self.model.to(torch.device('cuda'))
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        self.log_outputs_interval = 10
        self.log_images_interval = int(epoch_len / self.args.imepoch)

        print('starting epoch ', epoch)

        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))

            output = self.model(inputs)
            losses = self.model.optim_step(output)

            upto_log_time.update(time.time() - end)
            if self.log_outputs_now:
                self.model.log_outputs(output, inputs, losses, self.global_step,
                                       log_images=self.log_images_now, phase='train')
            batch_time.update(time.time() - end)
            end = time.time()
            
            if self.log_outputs_now:
                print('GPU {}: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"] if self.use_cuda else 'none', self._hp.exp_path))
                print(('itr: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.global_step, epoch, self.batch_idx, len(self.train_loader),
                    100. * self.batch_idx / len(self.train_loader), losses.total_loss.item())))
                
                print('avg time for loading: {:.2f}s, logs: {:.2f}s, compute: {:.2f}s, total: {:.2f}s'
                      .format(data_load_time.avg,
                              batch_time.avg - upto_log_time.avg,
                              upto_log_time.avg - data_load_time.avg,
                              batch_time.avg))
                togo_train_time = batch_time.avg * (self._hp.num_epochs - epoch) * epoch_len / 3600.
                print('ETA: {:.2f}h'.format(togo_train_time))
            
            del output, losses
            self.global_step = self.global_step + 1
        self.model.to(torch.device('cpu'))

    def val(self):
        print('Running Testing')
        if self.args.test_prediction:
            start = time.time()
            self.model_val.to(torch.device('cuda'))
            self.model_val.load_state_dict(self.model.state_dict())
            if self._hp.model_test is not None:
                self.model_test.load_state_dict(self.model.state_dict())
            losses_meter = RecursiveAverageMeter()
            with autograd.no_grad():
                for batch_idx, sample_batched in enumerate(self.val_loader):
                    inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
                    output = self.model_val(inputs)
                    losses = self.model_val.loss(output)
                    losses_meter.update(losses)
                    del losses
                self.model_val.log_outputs(
                    output, inputs, losses_meter.avg, self.global_step, log_images=True, phase='val')
                print(('\nTest set: Average loss: {:.4f} in {:.2f}s\n'
                       .format(losses_meter.avg.total_loss.item(), time.time() - start)))
            del output
        self.model_val.to(torch.device('cpu'))

    def get_optimizer_class(self):
        if self._hp.optimizer == 'adam':
            optim = partial(Adam, betas=(self._hp.adam_beta, 0.999))
        else:
            raise ValueError("Optimizer '{}' not supported!".format(self._hp.optimizer))
        return optim


def save_config(conf_path, exp_conf_path):
    copy(conf_path, exp_conf_path)


if __name__ == '__main__':
    ModelTrainer()
