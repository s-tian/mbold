from visual_mpc.policy.cem_controllers import CEMBaseController
import imp
import numpy as np
from visual_mpc.policy.cem_controllers.visualizer.construct_html import save_gifs, save_html, save_img, fill_template, img_entry_html, save_imgs, save_gifs_direct, save_imgs_direct, save_img_direct, save_img, save_html_direct
from visual_mpc.video_prediction.pred_util import get_context, rollout_predictions
from collections import OrderedDict
from classifier_control.classifier.models.base_tempdistclassifier import BaseTempDistClassifierTestTime

from robonet.video_prediction.testing import VPredEvaluation
import torch

LOG_SHIFT = 1e-5

import cv2

def resample_imgs(images, img_size):
    if len(images.shape) == 5:
        resized_images = np.zeros([images.shape[0], 1, img_size[0], img_size[1], 3], dtype=np.uint8)
        for t in range(images.shape[0]):
            resized_images[t] = \
            cv2.resize(images[t].squeeze(), (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)[None]
        return resized_images
    elif len(images.shape) == 3:
        return cv2.resize(images, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)

class GroundTruthDistController(CEMBaseController):
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
        CEMBaseController.__init__(self, ag_params, policyparams)

        predictor_hparams = {}
        predictor_hparams['run_batch_size'] = min(self._hp.vpred_batch_size, self._hp.num_samples)
        if self._hp.use_gt_model:
            self.predictor = MujocoPredictor(self.agentparams['env'])
        else:
            self.predictor = VPredEvaluation(self._hp.vidpred_model_path, predictor_hparams, n_gpus=ngpu, first_gpu=gpu_id)
            self.predictor.restore(gpu_mem_limit=True)

        self._net_context = self.predictor.n_context
        if self._hp.start_planning < self._net_context - 1:
            self._hp.start_planning = self._net_context - 1
        self.img_sz = self.predictor._input_hparams['img_size']

        learned_cost_testparams = {}
        learned_cost_testparams['batch_size'] = self._hp.num_samples
        learned_cost_testparams['data_conf'] = {'img_sz': self.img_sz}  #todo currently uses 64x64!!
        learned_cost_testparams['classifier_restore_path'] = self._hp.learned_cost_model_path
        #self.learned_cost = DistFuncEvaluation(self._hp.learned_cost, learned_cost_testparams)
        #self.device = torch.cuda()

        self._net_context = self.predictor.n_context
        if self._hp.start_planning < self._net_context - 1:
            self._hp.start_planning = self._net_context - 1

        self._img_height, self._img_width = [ag_params['image_height'], ag_params['image_width']]

        self._n_cam = 1 #self.predictor.n_cam

        self._desig_pix = None
        self._goal_pix = None
        self._images = None

        self._goal_image = None
        self._goal_obj_pos = None
        self._start_image = None
        self._verbose_worker = None

    def reset(self):
        self._expert_score = None
        self._images = None
        self._expert_images = None
        self._goal_image = None
        self._goal_obj_pos = None
        self._start_image = None
        self._verbose_worker = None
        return super(GroundTruthDistController, self).reset()

    def _default_hparams(self):
        default_dict = {
            'finalweight': 10,
            'state_append': None,
            'compare_to_expert': False,
            'verbose_img_height': 128,
            'verbose_frac_display': 0.,
            'vidpred_model_path': '',
            'learned_cost_model_path': '',
            'vpred_batch_size': 200,
            'learned_cost': BaseTempDistClassifierTestTime,
            'use_gt_model': False,
        }
        parent_params = super(GroundTruthDistController, self)._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    @staticmethod
    def gt_cost(inp_dict):
        goal = inp_dict['goal_state']
        states = inp_dict['current_state'][:, 9:15]
        if inp_dict['goal_state'].shape[-1] > 6:
            goal = inp_dict['goal_state'][..., 9:15]
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        if isinstance(goal, torch.Tensor):
            goal = goal.cpu().numpy()
        return np.sqrt(np.sum((states - goal)**2, axis=1))

    def evaluate_rollouts(self, actions, cem_itr):
        previous_actions = np.concatenate([x[None] for x in self._sampler.chosen_actions[-self._net_context:]], axis=0)
        previous_actions = np.tile(previous_actions, [actions.shape[0], 1, 1])
        # input_actions = np.concatenate((previous_actions, actions), axis=1)[:, :self.predictor.sequence_length]

        resampled_imgs = resample_imgs(self._images, self.img_sz)
        last_frames, last_states = get_context(self._net_context, self._t,
                                               self._state, resampled_imgs, self._hp)
        context = {
            "context_frames": last_frames[0],  #only take first batch example
            "context_actions": previous_actions[0],
            "context_states": last_states[0]
        }
        prediction_dict = self.predictor(context, {'actions': actions})
        gen_images = prediction_dict['predicted_frames']
        gen_states = prediction_dict['predicted_states']
        scores = []

        for tpred in range(gen_images.shape[1]):
            #input_images = ten2pytrch(gen_images[:, tpred], self.device)
            #inp_dict = {'current_img': input_images,
            #            'goal_img': uint2pytorch(resample_imgs(self._goal_image, self.img_sz), self._hp.num_samples, self.device)}
            inp_dict = {
                'current_state': gen_states[:, tpred],
                'goal_state': self._goal_obj_pos
            }

            print('peform prediction for ', tpred)
            scores.append(self.gt_cost(inp_dict))

        # weight final time step by some number and average over time.
        scores = np.stack(scores, 1)
        scores = self._weight_scores(scores)

        if self._verbose_condition(cem_itr):
            verbose_folder = self.traj_log_dir + "/planning_{}_itr_{}".format(self._t, cem_itr)

            content_dict = OrderedDict()
            visualize_indices = scores.argsort()[:10]

            # start images
            for c in range(self._n_cam):
                name = 'cam_{}_start'.format(c)
                save_path = save_img_direct(verbose_folder, name, self._images[-1, c])
                content_dict[name] = [save_path for _ in visualize_indices]

            name = 'goal_img'
            save_path = save_img_direct(verbose_folder, name, (self._goal_image*255).astype(np.uint8))
            content_dict[name] = [save_path for _ in visualize_indices]

            # render predicted images
            for c in range(self._n_cam):
                verbose_images = [(gen_images[g_i, :, c]*255).astype(np.uint8) for g_i in visualize_indices]
                row_name = 'cam_{}_pred_images'.format(c)
                content_dict[row_name] = save_gifs_direct(verbose_folder,
                                                       row_name, verbose_images)

            #self.learned_cost.model.visualize_test_time(content_dict, visualize_indices, verbose_folder)

            # save scores
            content_dict['scores'] = scores[visualize_indices]

            html_page = fill_template(cem_itr, self._t, content_dict, img_height=self._hp.verbose_img_height)
            save_html_direct("{}/plan.html".format(verbose_folder), html_page)

            #todo make logger instead of verbose worker !!

        return scores

    def _weight_scores(self, raw_scores):
        scores = raw_scores.copy()
        if self.last_plan:
            last_step = ((self.agentparams['T'] - self._hp.start_planning) % self.predictor.horizon) - 1
            for i in range(last_step, scores.shape[1]):
                scores[:, i] = scores[:, last_step]

        if self._hp.finalweight >= 0:
            scores[:, -1] *= self._hp.finalweight
            scores = np.sum(scores, axis=1) / sum([1. for _ in range(self.predictor.horizon - 1)] + [self._hp.finalweight])
        else:
            scores = scores[:, -1].copy()
        return scores


    def act(self, t=None, i_tr=None, images=None, goal_image=None, goal_obj_pose=None, verbose_worker=None, state=None):
        self._images = images
        self._verbose_worker = verbose_worker
        self._goal_obj_pos = goal_obj_pose
        if self.agentparams['T'] - t < self.predictor.horizon:
            self.last_plan = True
        else:
            self.last_plan = False
        ### Support for getting goal images from environment
        if goal_image.shape[0] == 1:
          self._goal_image = goal_image[0]
        else:
          self._goal_image = goal_image[-1, 0]  # pick the last time step as the goal image

        return super(GroundTruthDistController, self).act(t, i_tr, state)


def ten2pytrch(img, device):
    """Converts images to the [-1...1] range of the hierarchical planner."""
    img = img[:, 0]
    img = np.transpose(img, [0, 3, 1, 2])
    return torch.from_numpy(img * 2 - 1.0).float().to(device)

def uint2pytorch(img, num_samples, device):
    img = np.tile(img[None], [num_samples, 1, 1, 1])
    img = np.transpose(img, [0, 3, 1, 2])
    return torch.from_numpy(img * 2 - 1.0).float().to(device)


