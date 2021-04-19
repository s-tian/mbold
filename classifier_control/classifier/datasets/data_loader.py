import torch.utils.data as data
import numpy as np
import glob
import h5py
import pickle as pkl
import random
import pdb
import matplotlib.pyplot as plt
from torchvision.transforms import Resize
import imp
from torch.utils.data import DataLoader
import os

from classifier_control.classifier.utils.general_utils import AttrDict, map_dict
from classifier_control.classifier.utils.general_utils import resize_video


class BaseVideoDataset(data.Dataset):
    def __init__(self, data_dir, mpar, data_conf, phase, shuffle=True):
        """

        :param data_dir:
        :param mpar:
        :param data_conf:
        :param phase:
        :param shuffle: whether to shuffle within batch, set to False for computing metrics
        :param dataset_size:
        """

        self.phase = phase
        self.data_dir = data_dir
        self.data_conf = data_conf

        self.shuffle = shuffle and phase == 'train'
        self.img_sz = mpar.img_sz

        if shuffle:
            self.n_worker = 8
        else:
            self.n_worker = 1

    def get_data_loader(self, batch_size):
        print('len {} dataset {}'.format(self.phase, len(self)))
        return DataLoader(self, batch_size=batch_size, shuffle=self.shuffle, num_workers=self.n_worker,
                                  drop_last=True)


class FixLenVideoDataset(BaseVideoDataset):
    """
    Variable length video dataset
    """

    def __init__(self, data_dir, mpar, data_conf, phase='train', shuffle=True):
        """

        :param data_dir:
        :param data_conf:
        :param data_conf:  Attrdict with keys
        :param phase:
        :param shuffle: whether to shuffle within batch, set to False for computing metrics
        :param dataset_size:
        """
        super().__init__(data_dir, mpar, data_conf, phase, shuffle)

        self.filenames = self._maybe_post_split(self._get_filenames())
        random.seed(1)
        random.shuffle(self.filenames)

        self._data_conf = data_conf
        self.traj_per_file = self.get_traj_per_file(self.filenames[0])

        if hasattr(data_conf, 'T'):
            self.T = data_conf.T
        else: self.T = self.get_total_seqlen(self.filenames[0])

        self.transform = Resize([data_conf.img_sz[0], data_conf.img_sz[1]])
        self.flatten_im = False
        self.filter_repeated_tail = False

        print(phase)
        print(len(self.filenames))

    def _get_filenames(self):
        assert 'hdf5' not in self.data_dir, "hdf5 most not be containted in the data dir!"
        filenames = sorted(glob.glob(os.path.join(self.data_dir, os.path.join('hdf5', self.phase) + '/*')))
        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(self.data_dir))
        return filenames

    def get_traj_per_file(self, path):
        with h5py.File(path, 'r') as F:
            return F['traj_per_file'].value

    def get_total_seqlen(self, path):
        with h5py.File(path, 'r') as F:
            return F['traj0']['images'].value.shape[0]

    def _get_num_from_str(self, s):
        return int(''.join(filter(str.isdigit, s)))

    def get_extra_obs(self, traj_ind):
        main_dir = self.data_dir
        raw_dir = os.path.join(main_dir, 'raw')
        group_dir = os.path.join(raw_dir, f'traj_group{traj_ind//1000}')
        obs_path = os.path.join(os.path.join(group_dir, f'traj{traj_ind}'), 'obs_dict.pkl')
        with open(obs_path, 'rb') as f:
            obs_data = pkl.load(f)
        return obs_data

    def __getitem__(self, index):
        file_index = index // self.traj_per_file
        path = self.filenames[file_index]
        start_ind_str, _ = path.split('/')[-1][:-3].split('to')
        start_ind = self._get_num_from_str(start_ind_str)

        with h5py.File(path, 'r') as F:
            ex_index = index % self.traj_per_file  # get the index
            key = 'traj{}'.format(ex_index)

            traj_ind = start_ind + ex_index

            data_dict = AttrDict(images=(F[key + '/images'].value))
            # Fetch data into a dict
            for name in F[key].keys():
                if name in ['states', 'actions', 'pad_mask']:
                    data_dict[name] = F[key + '/' + name].value.astype(np.float32)

        data_dict = self.process_data_dict(data_dict)
        if self._data_conf.sel_len != -1:
            data_dict = self.sample_rand_shifts(data_dict)

        data_dict['index'] = index

        return data_dict

    def process_data_dict(self, data_dict):
        data_dict.demo_seq_images = self.preprocess_images(data_dict['images'])
        return data_dict

    def sample_rand_shifts(self, data_dict):
        """ This function processes data tensors so as to have length equal to max_seq_len
        by sampling / padding if necessary """
        offset = np.random.randint(0, self.T - self._data_conf.sel_len, 1)

        data_dict = map_dict(lambda tensor: self._croplen(tensor, offset, self._data_conf.sel_len), data_dict)
        if 'actions' in data_dict:
            data_dict.actions = data_dict.actions[:-1]

        return data_dict

    def preprocess_images(self, images):
        # Resize video
        if len(images.shape) == 5:
            images = images[:, 0]  # Number of cameras, used in RL environments
        assert images.dtype == np.uint8, 'image need to be uint8!'
        images = resize_video(images, (self.img_sz[0], self.img_sz[1]))
        images = np.transpose(images, [0, 3, 1, 2])  # convert to channel-first
        images = images.astype(np.float32) / 255 * 2 - 1
        assert images.dtype == np.float32, 'image need to be float32!'
        if self.flatten_im:
            images = np.reshape(images, [images.shape[0], -1])
        return images

    def _maybe_post_split(self, filenames):
        """Splits dataset percentage-wise if respective field defined."""
        try:
            return self._split_with_percentage(self.data_conf.train_val_split['post_split'], filenames)
        except (KeyError, AttributeError):
            return filenames

    def _split_with_percentage(self, frac, filenames):
        assert sum(frac.values()) <= 1.0  # fractions cannot sum up to more than 1
        assert self.phase in frac
        if self.phase == 'train':
            start, end = 0, frac['train']
        elif self.phase == 'val':
            start, end = frac['train'], frac['train'] + frac['val']
        else:
            start, end = frac['train'] + frac['val'], frac['train'] + frac['val'] + frac['test']
        start, end = int(len(filenames) * start), int(len(filenames) * end)
        return filenames[start:end]

    @staticmethod
    def _repeat_tail(data_dict, end_ind):
        data_dict.images[end_ind:] = data_dict.images[end_ind][None]
        if 'states' in data_dict:
            data_dict.states[end_ind:] = data_dict.states[end_ind][None]
        data_dict.pad_mask = np.ones_like(data_dict.pad_mask)
        end_ind = data_dict.pad_mask.shape[0] - 1
        return data_dict, end_ind

    def __len__(self):
        return len(self.filenames) * self.traj_per_file

    @staticmethod
    def _croplen(val, offset, target_length):
        """Pads / crops sequence to desired length."""

        val = val[int(offset):]
        len = val.shape[0]
        if len > target_length:
            return val[:target_length]
        elif len < target_length:
            raise ValueError("not enough length")
        else:
            return val

    @staticmethod
    def get_dataset_spec(data_dir):
        return imp.load_source('dataset_spec', os.path.join(data_dir, 'dataset_spec.py')).dataset_spec


if __name__ == '__main__':
    data_dir = os.environ['VMPC_DATA'] + '/classifier_control/data_collection/sim/1_obj_cartgripper_xz_rejsamp'
    hp = AttrDict(img_sz=(48, 64),
                  sel_len=-1,
                  T=31)

    loader = FixLenVideoDataset(data_dir, hp).get_data_loader(32)

    for i_batch, sample_batched in enumerate(loader):
        images = np.asarray(sample_batched['demo_seq_images'])

        pdb.set_trace()
        images = np.transpose((images + 1) / 2, [0, 1, 3, 4, 2])  # convert to channel-first
        actions = np.asarray(sample_batched['actions'])
        print('actions', actions)

        plt.imshow(np.asarray(images[0, 0]))
        plt.show()

