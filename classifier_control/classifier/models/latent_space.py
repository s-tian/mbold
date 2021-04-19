from contextlib import contextmanager
import numpy as np
import pdb
import torch
from classifier_control.classifier.utils.general_utils import AttrDict
import torch.nn as nn
import torch.nn.functional as F

import cv2
from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.utils.vae import VAE

class LatentSpace(BaseModel):
    def __init__(self, overrideparams, logger=None):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self.overrideparams = overrideparams
        self.override_defaults(overrideparams)  # override defaults with config file
        self.postprocess_params()

        assert self._hp.batch_size != -1   # make sure that batch size was overridden

        self.tdist_classifiers = []
        self.build_network()
        self._use_pred_length = False
        

    def _default_hparams(self):
        default_dict = AttrDict({
            'use_skips':False, #todo try resnet architecture!
            'ngf': 8,
            'action_size': 2,
            'nz_enc': 64,
#             'input_nc':3,
            'classifier_restore_path':None,  # not really needed here.,
            'hidden_size':256,
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def build_network(self, build_encoder=True):
        self.vae = VAE(self._hp)

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        tlen = inputs.demo_seq_images.shape[1]
#         print(inputs.demo_seq_images.min(), inputs.demo_seq_images.max(), "*****")
        self.images = inputs.demo_seq_images
        self.mu, self.logvar, _, self.rec = self.vae(self.images.view(-1, 3, 64, 64))
        self.rec = self.rec.view(-1,tlen, 3, 64, 64)
      
        pos_pairs, neg_pairs, pos_act, neg_act = self.sample_image_triplet_actions(inputs.demo_seq_images, inputs.actions, tlen, 1, inputs.states[:, :,  :2])
        
        ims = torch.cat([pos_pairs, neg_pairs], dim=0) 
        _, _, curr_z, _ = self.vae(ims[:, :3, :, :])
        _, _, goal_z, _ = self.vae(ims[:, 6:, :, :])
        
        dist = ((curr_z - goal_z)**2).mean(1)
        return dist
        
    
    def sample_image_triplet_actions(self, images, actions, tlen, tdist, states):
        
        # get positives:
        t0 = np.random.randint(0, tlen - tdist - 1, self._hp.batch_size)
        t1 = t0 + 1
        tg = t0 + 1 + np.random.randint(0, tdist, self._hp.batch_size)
        t0, t1, tg = torch.from_numpy(t0), torch.from_numpy(t1), torch.from_numpy(tg)

        im_t0 = select_indices(images, t0)
        im_t1 = select_indices(images, t1)
        im_tg = select_indices(images, tg)
        s_t0 = select_indices(states, t0)
        s_t1 = select_indices(states, t1)
        s_tg = select_indices(states, tg)
        pos_act = select_indices(actions, t0)

        self.pos_pair = torch.stack([im_t0, im_tg], dim=1)
        self.pos_pair_cat = torch.cat([im_t0, im_t1, im_tg], dim=1)

        # get negatives:
        t0 = np.random.randint(0, tlen - tdist - 1, self._hp.batch_size)
        t1 = t0 + 1
        tg = [np.random.randint(t0[b] + tdist + 1, tlen, 1) for b in range(self._hp.batch_size)]
        tg = np.array(tg).squeeze()
        t0, t1, tg = torch.from_numpy(t0), torch.from_numpy(t1), torch.from_numpy(tg)

        im_t0 = select_indices(images, t0)
        im_t1 = select_indices(images, t1)
        im_tg = select_indices(images, tg)
        s_t0 = select_indices(states, t0)
        s_t1 = select_indices(states, t1)
        s_tg = select_indices(states, tg)
        neg_act = select_indices(actions, t0)
        self.neg_pair = torch.stack([im_t0, im_tg], dim=1)
        self.neg_pair_cat = torch.cat([im_t0, im_t1, im_tg], dim=1)

        # one means within range of tdist range,  zero means outside of tdist range
        self.labels = torch.cat([torch.ones(self._hp.batch_size), torch.zeros(self._hp.batch_size)])

        return self.pos_pair_cat, self.neg_pair_cat, pos_act, neg_act


    def loss(self, model_output):
#         BCE = F.binary_cross_entropy(self.rec.view(-1, 3, 64, 64), self.images.view(-1, 3, 64, 64), size_average=False)
#         BCE = F.mse_loss(self.rec.view(-1, 3, 64, 64), ((self.images.view(-1, 3, 64, 64) + 1 ) / 2.0), size_average=False)
        BCE = ((self.rec - ((self.images + 1 ) / 2.0))**2).mean()
        for i in range(10):
            rec = self.rec[i, 0].permute(1,2,0).cpu().detach().numpy() * 255.0
            im = ((self.images + 1 ) / 2.0)[i, 0].permute(1,2,0).cpu().detach().numpy() * 255.0
            ex = np.concatenate([rec,im], 0)
            cv2.imwrite("ex"+str(i)+".png", ex)
            
#         print(BCE)
        KLD = -0.5 * torch.mean(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
#         print(KLD)
        losses = AttrDict()
        losses.total_loss = BCE + 0.00001 * KLD
        return losses
    
    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        if log_images:
            self._logger.log_single_tdist_classifier_image(self.pos_pair, self.neg_pair, model_output.squeeze(),
                                                          'tdist{}'.format("Q"), step, phase)
#             self._logger.log_heatmap_image(self.pos_pair, qvals, model_output.squeeze(),
#                                                           'tdist{}'.format("Q"), step, phase)

    def get_device(self):
        return self._hp.device
    
    


    
def select_indices(tensor, indices):
    new_images = []
    for b in range(tensor.shape[0]):
        new_images.append(tensor[b, indices[b]])
    tensor = torch.stack(new_images, dim=0)
    return tensor

class LatentSpaceTestTime(LatentSpace):
    def __init__(self, overrideparams, logger=None):
        super(LatentSpaceTestTime, self).__init__(overrideparams, logger)
        checkpoint = torch.load(self._hp.classifier_restore_path, map_location=self._hp.device)
        self.load_state_dict(checkpoint['state_dict'])

    def _default_hparams(self):
        parent_params = super()._default_hparams()
        parent_params.add_hparam('classifier_restore_path', None)
        return parent_params
      
      
    def visualize_test_time(self, content_dict, visualize_indices, verbose_folder):
        pass
      
    def forward(self, inputs):
      _, _, curr_z, _ = self.vae(inputs['current_img'])
      _, _, goal_z, _ = self.vae(inputs['goal_img'])
      dist = ((curr_z - goal_z)**2).mean(1)
      return dist.detach().cpu().numpy()
