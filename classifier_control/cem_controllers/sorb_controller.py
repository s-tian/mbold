import os
import numpy as np
import torch
import cv2
import networkx as nx
import tqdm
import pickle as pkl

from visual_mpc.policy.policy import Policy
from classifier_control.classifier.utils.DistFuncEvaluation import DistFuncEvaluation
from classifier_control.classifier.models.dist_q_function import DistQFunctionTestTime
from classifier_control.classifier.models.gc_bc import GCBCTestTime
from classifier_control.classifier.datasets.data_loader import FixLenVideoDataset
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.cem_controllers.pytorch_classifier_controller import ten2pytrch, uint2pytorch, resample_imgs


class SORBController(Policy):
    """
    Run Search on the Replay Buffer.
    Code largely based on the author's implementation in https://colab.research.google.com/github/google-research/google-research/blob/master/sorb/SoRB.ipynb.
    However, a key difference is that we use a goal-conditioned behavior cloning inverse model rather than an actor
    learned alongside the distance function, which we find performs much better empirically.
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

        self.learned_cost = DistFuncEvaluation(DistQFunctionTestTime, learned_cost_testparams)
        self.device = self.learned_cost.model.get_device()
        learned_cost_dir = os.path.dirname(learned_cost_testparams['classifier_restore_path'])
        graph_dir = learned_cost_dir + '/graph.pkl'
        if not os.path.isfile(graph_dir):
            self.preconstruct_graph(graph_dir)

        self.graph, self.graph_states = self.construct_graph(graph_dir)

        inv_model_testparams = self.setup_model_testparams(self._hp.inv_model_path)
        self.inverse_model = DistFuncEvaluation(GCBCTestTime, inv_model_testparams)

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

    def compute_pairwise_dist(self, v1, v2=None):
        if v2 is None:
            v2 = v1
        dists = []
        if not torch.is_tensor(v2):
            v2 = torch.FloatTensor(v2)
        if not torch.is_tensor(v1):
            v1 = torch.FloatTensor(v1)

        if v2.shape[0] == 1:
            curr = 0
            while curr < v1.shape[0]:
                batch = v1[curr:curr+self._hp.dloader_bs]
                inp_dict = {'current_img': batch.cuda(),
                            'goal_img': v2.repeat(batch.shape[0], 1, 1, 1).cuda(), }
                score = self.learned_cost.predict(inp_dict)
                if not hasattr(score, '__len__'):
                    score = np.array([score])
                dists.append(score)
                curr += self._hp.dloader_bs
            dists = np.concatenate(dists)[None]
        else:
            for i, image in tqdm.tqdm(enumerate(v1)):
                inp_dict = {'current_img': image[None].repeat(v2.shape[0], 1, 1, 1).cuda(),
                            'goal_img': v2.cuda(),}
                score = self.learned_cost.predict(inp_dict)
                dists.append(score)
        dists = np.stack(dists)
        return dists

    def preconstruct_graph(self, cache_fname):
        images = self.get_random_observations()
        dist = self.compute_pairwise_dist(images)
        graph = {'images': images.cpu().numpy(), 'dists': dist}
        with open(cache_fname, 'wb') as f:
            pkl.dump(graph, f)

    def construct_graph(self, cache_fname):
        # Load cache
        with open(cache_fname, 'rb') as f:
            data = pkl.load(f)
            images, dists = data['images'], data['dists']
        g = nx.DiGraph()
        for i, s_i in enumerate(images):
            for j, s_j in enumerate(images):
                length = dists[i, j]
                if self.dist_check(length):
                    g.add_edge(i, j, weight=length)
        return g, images

    def get_random_observations(self):
        hp = AttrDict(img_sz=(64, 64),
                      sel_len=-1,
                      T=31)
        dataset = FixLenVideoDataset(self._hp.graph_dataset, self.learned_cost.model._hp, hp).get_data_loader(self._hp.dloader_bs)
        total_images = []
        dl = iter(dataset)
        for i in range(self._hp.graph_size // self._hp.dloader_bs):
            try:
                batch = next(dl)
            except StopIteration:
                dl = iter(dataset)
                batch = next(dl)
            images = batch['demo_seq_images']
            selected_images = images[torch.arange(len(images)), torch.randint(0, images.shape[1], (len(images),))]
            total_images.append(selected_images)
        total_images = torch.cat(total_images)
        return total_images

    def reset(self):
        self._expert_score = None
        self._images = None
        self._expert_images = None
        self._goal_image = None
        self._start_image = None
        self._verbose_worker = None
        return super(SORBController, self).reset()

    def _default_hparams(self):
        default_dict = {
            'learned_cost_model_path': None,
            'inv_model_path': None,
            'verbose_every_iter': False,
            'dist_q': True,
            'graph_dataset': None,
            'graph_size': 5000,
            'dloader_bs': 500,
            'num_samples': 200,
            'max_dist': 15.0,
            'min_dist': 0.0,
        }
        parent_params = super(SORBController, self)._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def dist_check(self, dist):
        return self._hp.min_dist < dist < self._hp.max_dist

    def get_waypoint(self, input_images, goal_img):
        g2 = self.graph.copy()
        start_to_rb = self.compute_pairwise_dist(input_images[None], self.graph_states).flatten()
        rb_to_goal = self.compute_pairwise_dist(self.graph_states, goal_img).flatten()
        start_to_goal = self.compute_pairwise_dist(input_images[None], goal_img).flatten().squeeze()
        for i, (dist_from_start, dist_to_goal) in enumerate(zip(start_to_rb, rb_to_goal)):
            if self.dist_check(dist_from_start):
                g2.add_edge('start', i, weight=dist_from_start)
            if self.dist_check(dist_to_goal):
                g2.add_edge(i, 'goal', weight=dist_to_goal)
        try:
            path = nx.shortest_path(g2, 'start', 'goal', weight='weight')
            edge_lengths = []
            for (i, j) in zip(path[:-1], path[1:]):
                edge_lengths.append(g2[i][j]['weight'])
        except:
            path = ['start', 'goal']
            edge_lengths = [start_to_goal]

        wypt_to_goal_dist = np.cumsum(edge_lengths[::-1])[::-1]  # Reverse CumSum
        waypoint_vec = list(path)[1:-1]
        verbose_folder = self.traj_log_dir
        plan_imgs = [self.graph_states[i] for i in waypoint_vec]
        plan_imgs_cat = np.concatenate([input_images.cpu().numpy()] + plan_imgs + [goal_img[0].cpu().numpy()], axis=1)
        plan_imgs_cat = np.transpose((plan_imgs_cat + 1) / 2 * 255, [1, 2, 0])
        cv2.imwrite(verbose_folder + '/plan_{}.png'.format(self._t), plan_imgs_cat[:, :, ::-1])

        return waypoint_vec, wypt_to_goal_dist[1:], edge_lengths[0], start_to_goal

    def get_best_action(self, t=None):
        resampled_imgs = resample_imgs(self._images, self.img_sz) / 255.
        input_images = ten2pytrch(resampled_imgs, self.device)[-1]
        goal_img = uint2pytorch(resample_imgs(self._goal_image, self.img_sz), self._hp.num_samples, self.device)

        waypoints, graph_dists, first_wp_dist, start_to_goal = self.get_waypoint(input_images, goal_img[0][None])
        if len(waypoints) > 0 and (first_wp_dist < start_to_goal or start_to_goal > self._hp.max_dist):
            wpt_goal = torch.FloatTensor(self.graph_states[waypoints[0]])[None].to(self.device)
        else:
            wpt_goal = goal_img[0][None]

        inp_dict = {
            'current_img': input_images[None],
            'goal_img': wpt_goal,
        }

        act = self.inverse_model.predict(inp_dict).action[0].cpu().detach().numpy()
        return act

    def act(self, t=None, i_tr=None, images=None, goal_image=None, verbose_worker=None, state=None):
        self._images = images
        self._states = state
        self._verbose_worker = verbose_worker
        self._t = t

        ### Support for getting goal images from environment
        if goal_image.shape[0] == 1:
          self._goal_image = goal_image[0]
        else:
          self._goal_image = goal_image[-1, 0]  # pick the last time step as the goal image

        action = {'actions': self.get_best_action(t)}
        print(action)
        return action

