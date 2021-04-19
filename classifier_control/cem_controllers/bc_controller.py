from visual_mpc.policy.policy import Policy
import numpy as np
import torch
import cv2

from classifier_control.classifier.models.gc_bc import GCBCTestTime
from classifier_control.classifier.utils.DistFuncEvaluation import DistFuncEvaluation


def resample_imgs(images, img_size):
    if images.shape[-1] > img_size[-1]:
        interp_type = cv2.INTER_AREA
    else:
        interp_type = cv2.INTER_CUBIC
    if len(images.shape) == 5:
        resized_images = np.zeros([images.shape[0], 1, img_size[0], img_size[1], 3], dtype=np.uint8)
        for t in range(images.shape[0]):
            resized_images[t] = \
            cv2.resize(images[t].squeeze(), (img_size[1], img_size[0]), interpolation=interp_type)[None]
        return resized_images
    elif len(images.shape) == 3:
        return cv2.resize(images, (img_size[1], img_size[0]), interpolation=interp_type)

class BCController(Policy):
    """
    Use the goal-conditioned behavior cloning baseline model to perform control.
    """

    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        """

        :param ag_params: agent parameters
        :param policyparams: policy parameters
        :param gpu_id: starting gpu id
        :param ngpu: number of gpus
        """
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

        self.agentparams = ag_params
        self.img_sz = (64, 64)
        learned_cost_testparams = {}
        learned_cost_testparams['batch_size'] = self._hp.num_samples
        learned_cost_testparams['data_conf'] = {'img_sz': self.img_sz}  #todo currently uses 64x64!!
        learned_cost_testparams['classifier_restore_path'] = self._hp.learned_cost_model_path
        learned_cost_testparams['classifier_restore_paths'] = ['']
        self.learned_cost = DistFuncEvaluation(GCBCTestTime, learned_cost_testparams)
        self.device = self.learned_cost.model.get_device()

        self._img_height, self._img_width = [ag_params['image_height'], ag_params['image_width']]

        self._adim = self.agentparams['adim']
        self._sdim = self.agentparams['sdim']

        self._n_cam = 1

        self._desig_pix = None
        self._goal_pix = None
        self._images = None

        self._goal_image = None
        self._start_image = None
        self._verbose_worker = None

    def reset(self):
        self._expert_score = None
        self._images = None
        self._expert_images = None
        self._goal_image = None
        self._start_image = None
        self._verbose_worker = None
        return super(BCController, self).reset()

    def _default_hparams(self):
        default_dict = {
            'action_sample_batches': 1,
            'num_samples': 200,
            'learned_cost_model_path': None,
            'verbose_every_iter': False,
        }
        parent_params = super(BCController, self)._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def get_best_action(self, t=None):
        resampled_imgs = resample_imgs(self._images, self.img_sz) / 255.
        input_images = ten2pytrch(resampled_imgs, self.device)[-1]
        input_images = input_images[None].repeat(self._hp.num_samples, 1, 1, 1)
        input_states = torch.from_numpy(self._states)[None].float().to(self.device).repeat(self._hp.num_samples, 1)
        goal_img = uint2pytorch(resample_imgs(self._goal_image, self.img_sz), self._hp.num_samples, self.device)

        inp_dict = {'current_img': input_images,
                    'current_state': input_states,
                    'goal_img': goal_img,}
        act = self.learned_cost.predict(inp_dict).action[0].cpu().detach().numpy()
        return act

    def act(self, t=None, i_tr=None, images=None, goal_image=None, verbose_worker=None, state=None):
        self._images = images
        self._states = state
        self._verbose_worker = verbose_worker

        ### Support for getting goal images from environment
        if goal_image.shape[0] == 1:
          self._goal_image = goal_image[0]
        else:
          self._goal_image = goal_image[-1, 0]  # pick the last time step as the goal image

        return {'actions': self.get_best_action(t)}


def ten2pytrch(img, device):
    """Converts images to the [-1...1] range of the hierarchical planner."""
    img = img[:, 0]
    img = np.transpose(img, [0, 3, 1, 2])
    return torch.from_numpy(img * 2 - 1.0).float().to(device)


def uint2pytorch(img, num_samples, device):
    img = np.tile(img[None], [num_samples, 1, 1, 1])
    img = np.transpose(img, [0, 3, 1, 2])
    return torch.from_numpy(img * 2 - 1.0).float().to(device)


