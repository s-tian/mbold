import numpy as np
import torch

from visual_mpc.policy.policy import Policy
from classifier_control.classifier.utils.DistFuncEvaluation import DistFuncEvaluation
from classifier_control.classifier.models.q_function import QFunctionTestTime
from classifier_control.cem_controllers.pytorch_classifier_controller import ten2pytrch, uint2pytorch, resample_imgs


class QFunctionController(Policy):
    """
    Cross Entropy Method Stochastic Optimizer
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

        learned_cost_testparams = self.setup_model_testparams(self._hp.learned_cost_model_path)

        self.learned_cost = DistFuncEvaluation(QFunctionTestTime, learned_cost_testparams)
        self.device = self.learned_cost.model.get_device()

        self._img_height, self._img_width = [ag_params['image_height'], ag_params['image_width']]

        self._adim = self.agentparams['adim']
        self._sdim = self.agentparams['sdim']

        self._n_cam = 1 #self.predictor.n_cam

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
        return super(QFunctionController, self).reset()

    def setup_model_testparams(self, model_dir):
        learned_cost_testparams = {
            'batch_size': self._hp.num_samples,
            'data_conf': {
                'img_sz': self.img_sz
            },
            'classifier_restore_path': model_dir,
            'classifier_restore_paths': ['']
        }
        return learned_cost_testparams

    def _default_hparams(self):
        default_dict = {
            'action_sample_batches': 1,
            'num_samples': 200,
            'learned_cost_model_path': None,
            'verbose_every_iter': False,
        }
        parent_params = super(QFunctionController, self)._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def get_best_action(self, t=None):
        resampled_imgs = resample_imgs(self._images, self.img_sz) / 255.
        input_images = ten2pytrch(resampled_imgs, self.device)[-1]
        input_images = input_images[None].repeat(self._hp.num_samples, 1, 1, 1)
        input_states = torch.from_numpy(self._states)[None].float().to(self.device).repeat(self._hp.num_samples, 1)
        goal_img = uint2pytorch(resample_imgs(self._goal_image, self.img_sz), self._hp.num_samples, self.device)

        try_actions = np.random.uniform(-1, 1, size=(self._hp.num_samples, self._adim))
        try_actions = np.clip(try_actions, -1, 1)
        try_actions_tensor = torch.FloatTensor(try_actions).cuda()
        inp_dict = {
                 'current_img': input_images,
                 'goal_img': goal_img,
                 'actions': try_actions_tensor
              }
        qvalues = self.learned_cost.predict(inp_dict)
        best_action_ind = np.argmin(qvalues, axis=0)
        act = try_actions[best_action_ind]

        return act

    def act(self, t=None, i_tr=None, images=None, goal_image=None, verbose_worker=None, state=None):
        self._images = images
        self._states = state[-1][:2]
        print(f'state {t}: {self._states}')
        self._verbose_worker = verbose_worker

        ### Support for getting goal images from environment
        if goal_image.shape[0] == 1:
          self._goal_image = goal_image[0]
        else:
          self._goal_image = goal_image[-1, 0]  # pick the last time step as the goal image

        return {'actions': self.get_best_action(t)}

