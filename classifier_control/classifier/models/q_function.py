import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.utils.actor_network import ActorNetwork
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.classifier.utils.q_network import QNetwork


class QFunction(BaseModel):
    INFINITELY_FAR = 1000

    def __init__(self, overrideparams, logger=None):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self.overrideparams = overrideparams
        self.override_defaults(overrideparams)  # override defaults with config file
        self.postprocess_params()

        assert self._hp.batch_size != -1   # make sure that batch size was overridden

        self.build_network()
        self._use_pred_length = False
        self.target_network_counter = 0
        self.update_pi_counter = 0
        self.hm_counter = 0 # counts when to compute and log a heatmap of the Q-function

    def init_optimizers(self, hp):
        self.critic_optimizer = Adam(self.qnetworks.parameters(), lr=hp.lr)
        self.optimizer = self.critic_optimizer
        if self.actor_critic:
            self.actor_optimizer = Adam(self.pi_net.parameters(), lr=hp.lr)
        if self._hp.min_q_lagrange:
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.get_device())
            self.alpha_optimizer = Adam(
                [self.log_alpha, ],
                lr=1e-3,
            )

    def optim_step(self, output):
        losses = self.loss(output)
        if self._hp.min_q_lagrange:
            self.alpha_optimizer.zero_grad()
            lagrange_loss = self.min_q_lagrange_loss
            lagrange_loss.backward(retain_graph=True)
            self.alpha_optimizer.step()
        self.critic_optimizer.zero_grad()
        losses.total_loss.backward(retain_graph=self.actor_critic)
        self.critic_optimizer.step()

        if self.actor_critic:
            for p in self.qnetworks.parameters():
                p.requires_grad = False
            actor_loss = self.actor_loss()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            for p in self.qnetworks.parameters():
                p.requires_grad = True
            losses.actor_loss = actor_loss

        # Target network updates
        """ Target network lagging update """
        self.target_network_counter = self.target_network_counter + 1
        if self.target_network_counter % self._hp.update_target_rate == 0:
            self.target_network_counter = 0
            if self._hp.target_network_update == 'replace':
                self.target_qnetworks.load_state_dict(self.qnetworks.state_dict())
                if self._hp.eval_target_nets:
                    self.target_qnetworks.eval()
                if self.actor_critic:
                    self.target_pi_net.load_state_dict(self.pi_net.state_dict())
                    if self._hp.eval_target_nets:
                        self.target_pi_net.eval()
            elif self._hp.target_network_update == 'polyak':
                with torch.no_grad():
                    for q_net, t_q_net in zip(self.qnetworks, self.target_qnetworks):
                        for p, p_targ in zip(q_net.parameters(), t_q_net.parameters()):
                            p_targ.data.mul_(self._hp.polyak)
                            p_targ.data.add_((1 - self._hp.polyak) * p.data)
                        # Copy over batchnorm statistics
                        for p, p_targ in zip(q_net.buffers(), t_q_net.buffers()):
                            p_targ.data.mul_(0)
                            p_targ.data.add_(p.data)
                    if self.actor_critic:
                        for p, p_targ in zip(self.pi_net.parameters(), self.target_pi_net.parameters()):
                            p_targ.data.mul_(self._hp.polyak)
                            p_targ.data.add_((1 - self._hp.polyak) * p.data)
                            # Copy over batchnorm statistics
                        for p, p_targ in zip(self.pi_net.buffers(), self.target_pi_net.buffers()):
                            p_targ.data.mul_(0)
                            p_targ.data.add_(p.data)
        return losses

    def actor_loss(self):
        image_pairs = self.get_sg_pair(self.images)
        self.target_actions_taken = self.pi_net(image_pairs)
        return -self.network_out_2_qval(self.qnetworks[0](image_pairs, self.target_actions_taken)).mean()

    @property
    def num_network_outputs(self):
        return 1

    def _default_hparams(self):
        default_dict = AttrDict({
            'use_skips':False,
            'ngf': 8,
            'action_size': 2,
            'state_size': 30,
            'nz_enc': 64,
            'linear_layer_size': 128,
            'classifier_restore_path':None,  # not really needed here.,
            'low_dim':False,
            'gamma':0.0,
            'terminal': True,
            'update_target_rate': 1,
            'action_range': [-1.0, 1.0],
            'action_stds': [0.6, 0.6, 0.3, 0.3],
            'est_max_samples': 100,
            'binary_reward': [0, 1],
            'n_step': 1,
            'min_q': False,
            'min_q_weight': 1.0,
            'min_q_lagrange': False,
            'min_q_eps': 0.1,
            'sigmoid': False,
            'optimize_actions': 'random_shooting',
            'target_network_update': 'replace',
            'polyak': 0.995,
            'sg_sample': 'half_unif_half_first',
            'geom_sample_p': 0.5,
            'bellman_weight': 1.0,
            'td_loss': 'mse',
            'add_negative_sample': False,
            'negative_sample_type': 'copy_arm', # also rand_arm, batch_goal
            'gaussian_blur': False,
            'twin_critics': False,
            'add_action_noise': False,
            'action_scaling': 1.0,
            'eval_target_nets': True,
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def build_network(self, build_encoder=True):
        q_network_type = QNetwork
        num_q_fns = 2 if self._hp.twin_critics else 1
        self.qnetworks = torch.nn.ModuleList([q_network_type(self._hp, self.num_network_outputs) for _ in range(num_q_fns)])
        with torch.no_grad():
            self.target_qnetworks = torch.nn.ModuleList([q_network_type(self._hp, self.num_network_outputs) for _ in range(num_q_fns)])
            for i, t_qn in enumerate(self.target_qnetworks):
                t_qn.load_state_dict(self.qnetworks[i].state_dict())
                if self._hp.eval_target_nets:
                    t_qn.eval()

        if self.actor_critic:
            self.pi_net = ActorNetwork(self._hp)
            with torch.no_grad():
                self.target_pi_net = ActorNetwork(self._hp)
                self.target_pi_net.load_state_dict(self.pi_net.state_dict())
                if self._hp.eval_target_nets:
                    self.target_pi_net.eval()

    def train(self, mode=True):
        super(QFunction, self).train(mode)
        if self._hp.eval_target_nets:
            if self.actor_critic:
                self.target_pi_net.eval()
            for t_qn in self.target_qnetworks:
                t_qn.eval()
        return self

    def network_out_2_qval(self, network_outputs):
        """
        :param softmax_values: Tensor of softmax values of dimension [..., num_bins]
        :return: Tensor of dimension [...] containing scalar q values.
        """
        if self._hp.sigmoid:
            network_outputs = F.sigmoid(network_outputs)
        return network_outputs.squeeze()

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        #### Train vs Test
        if "demo_seq_images" in inputs.keys():
            tlen = inputs.demo_seq_images.shape[1]
            self.inputs = inputs
            inputs.actions = inputs.actions * self._hp.action_scaling
            image_pairs, acts = self.sample_image_triplet_actions(inputs.demo_seq_images,
                                                                  inputs.actions, tlen, 1,
                                                                  inputs.states)
            self.images = image_pairs

            if self._hp.low_dim:
                image_0 = self.images[:, :self._hp.state_size]
                image_g = self.images[:, 2*self._hp.state_size:]
            else:
                image_0 = self.images[:, :3]
                image_g = self.images[:, 6:]

            image_pairs = torch.cat([image_0, image_g], dim=1)
            self.acts = acts
            network_out = [qnet(image_pairs, acts) for qnet in self.qnetworks] 
            self.network_out = network_out
            qval = [self.network_out_2_qval(n_out) for n_out in network_out]

        else:
            if self._hp.low_dim:
                image_pairs = torch.cat([inputs["current_state"], inputs['goal_state']], dim=1)
            else:
                image_pairs = torch.cat([inputs["current_img"], inputs["goal_img"]], dim=1)

            if 'actions' in inputs:
                network_out = [qnet(image_pairs, inputs['actions']) for qnet in
                               self.target_qnetworks]  # Just use the first one if we have two critics

                qval = [self.network_out_2_qval(n_out) for n_out in network_out]
                qs, _ = self.get_worst_of_qs(torch.stack(qval))
                return qs.detach().cpu().numpy()

            qval = self.get_max_q(image_pairs)
            qval = torch.squeeze(qval).detach().cpu().numpy()
        return qval

    def get_best_of_qs(self, qvals):
        return torch.max(qvals, dim=0)

    def get_worst_of_qs(self, qvals):
        return torch.min(qvals, dim=0)

    def get_max_q(self, image_pairs, return_raw=False):
        """
        :param image_pairs: image pairs (s)
        :param return_raw: whether or not to return pre-network_out_2_q_val outputs
        :return: max_a Q(s, a)
        """
        if self._hp.optimize_actions == 'random_shooting':
            ensem = []
            for target_qnet in self.target_qnetworks:
                qs = self.compute_action_samples(image_pairs, target_qnet, parallel=True, detach_grad=True, use_hp=False)
                max_qs, inds = self.get_best_of_qs(self.network_out_2_qval(qs))
                ensem.append(max_qs.detach())
            max_qs, inds = self.get_worst_of_qs(torch.stack(ensem))
            if return_raw:
                max_q_raw_outs = max_qs[inds, torch.arange(len(inds))]
        elif self.actor_critic:
            with torch.no_grad():
                best_actions = self.target_pi_net(image_pairs)
                if self._hp.add_action_noise and self.training:
                    best_actions += torch.clamp(torch.normal(mean=0, std=0.1, size=best_actions.shape).cuda(), min=-0.2, max=0.2)
                max_q_raw_outs = [q_net(image_pairs, best_actions) for q_net in self.target_qnetworks]
                max_qs = [self.network_out_2_qval(raw_outs) for raw_outs in max_q_raw_outs]
                max_qs, inds = self.get_worst_of_qs(torch.stack(max_qs))
                max_q_raw_outs = torch.stack(max_q_raw_outs)[inds, torch.arange(len(image_pairs))]
        if return_raw:
            return max_qs, max_q_raw_outs
        return max_qs

    @property
    def actor_critic(self):
        return self._hp.optimize_actions == 'actor_critic'

    def get_arm_state(self, states):
        if states.shape[-1] > 18:
            return torch.cat((states[..., :9], states[..., self._hp.state_size//2:self._hp.state_size//2+9]), axis=-1)
        elif states.shape[-1] == 6:
            return states
        else:
            raise NotImplementedError('state shape does not fit expectations')

    def mse_reward(self, image_pairs):
        split_ind = image_pairs.shape[1]//2
        start_im, goal_im = image_pairs[:, :split_ind], image_pairs[:, split_ind:]
        return -torch.mean((start_im - goal_im) ** 2)

    def sample_sg_indices(self, tlen, bs, sampling_strat):
        if sampling_strat == 'uniform_pair':
            """
            Sample two sets of indices, and then compute the min to be the start index and max to be the goal.
            This gives uniform probability over all possible selected _pairs_
            """
            i0 = torch.randint(0, tlen, (bs,), device=self.get_device(), dtype=torch.long)
            i1 = torch.randint(0, tlen-1, (bs,), device=self.get_device(), dtype=torch.long)
            i1[i1 == i0] = tlen-1
            return torch.min(i0, i1), torch.max(i0, i1)
        elif sampling_strat == 'uniform_distance':
            """
            Sample the distances between the pairs uniformly at random
            """
            distance = torch.randint(1, tlen, (bs,), device=self.get_device(), dtype=torch.long)
            i0 = torch.LongTensor([torch.randint(0, tlen-distance[b], (1,)) for b in range(bs)]).to(self.get_device())
            i1 = i0 + distance
            return i0, i1
        elif sampling_strat == 'uniform_negatives':
            """
            Sample the starting state first, then the goal from remaining possibilities. Note this is different from "uniform_pair".
            """
            i0 = torch.randint(0, tlen - 2, (bs,), device=self.get_device(), dtype=torch.long)
            i1 = torch.LongTensor([torch.randint(i0[b] + 2, tlen, (1,)) for b in range(bs)]).to(self.get_device())
            return torch.min(i0, i1), torch.max(i0, i1)
        elif sampling_strat == 'geometric':
            """
            Sample the first index, and then sample the second according to a geometric distribution with parameter p 
            """
            i0 = torch.randint(0, tlen-1, (bs,), device=self.get_device(), dtype=torch.long)
            dist = torch.distributions.geometric.Geometric(self._hp.geom_sample_p).sample((bs,)).to(self.get_device()).long()
            return i0, torch.clamp(i0+dist, max=tlen-1)
        elif sampling_strat == 'half_unif_half_first':
            """
            Sample half of the batch uniformly, and the other half so that ig = i0 + 1
            """
            i0_first = torch.randint(0, tlen-1, (bs//2,), device=self.get_device(), dtype=torch.long)
            ig_first = i0_first + 1
            i0, ig = self.sample_sg_indices(tlen, bs//2, 'uniform_negatives')
            return torch.cat([i0_first, i0]), torch.cat([ig_first, ig])
        else:
            assert NotImplementedError(f'Sampling method {sampling_strat} not implemented!')

    def add_negative_sample(self, s_t0, s_t1, s_tg, query_goal, acts, t0, t1, tg):
        curr_bs = self._hp.batch_size
        state_shape = [1] * len(s_t0.shape)
        state_shape[0] = 2
        s_t0 = s_t0.repeat(*state_shape) # These two remain unchanged, but tiled in dim 1
        s_t1 = s_t1.repeat(*state_shape)

        if self._hp.negative_sample_type == 'nn_idx':
            arm_pos_query = self.get_arm_state(query_goal)[..., :self._hp.nn_dim]
            close_inds = self.nn_idx.find_knn(arm_pos_query, k=2)[:, 1]
            negative_sampled_goals = self.nn_idx.lookup(close_inds)
        elif self._hp.negative_sample_type == 'batch_goal':
            negative_sampled_goals = torch.roll(s_tg, 1, dims=0)
        else:
            raise NotImplementedError(f'Negative sample type {self._hp.negative_sample_type} not implemented!')

        s_tg = torch.cat((s_tg, negative_sampled_goals))
        acts = acts.repeat(2, 1)
        t0 = t0.repeat(2)
        t1 = t1.repeat(2)
        if self._hp.negative_sample_type in ['copy_arm', 'batch_goal', 'nn_idx']:
            tg_prime = torch.ones(curr_bs).to(self.get_device()).long() * self.INFINITELY_FAR
        elif self._hp.negative_sample_type == 'rand_arm':
            tg_prime = tg
        tg = torch.cat((tg, tg_prime))
        return s_t0, s_t1, s_tg, acts, t0, t1, tg

    def sample_image_triplet_actions(self, images, actions, tlen, tdist, states):

        if self._hp.state_size == 18:
            states = self.get_arm_state(states)
        else:
            states = states[:, :, :self._hp.state_size]

        t0, tg = self.sample_sg_indices(tlen, self._hp.batch_size, self._hp.sg_sample)
        t1 = t0 + self._hp.n_step

        im_t0 = select_indices(images, t0)
        im_t1 = select_indices(images, t1)
        im_tg = select_indices(images, tg)
        if self._hp.low_dim:
            s_t0 = select_indices(states, t0)
            s_t1 = select_indices(states, t1)
        else:
            s_t0, s_t1 = None, None
        s_tg = select_indices(states, tg)
        acts = select_indices(actions, t0)

        self.t0, self.t1, self.tg = t0, t1, tg

        if self._hp.add_negative_sample:
            if self._hp.low_dim:
                s_t0, s_t1, s_tg, acts, t0, t1, tg = self.add_negative_sample(s_t0, s_t1, s_tg, s_tg, acts, t0, t1, tg)
            else:
                im_t0, im_t1, im_tg, acts, t0, t1, tg = self.add_negative_sample(im_t0, im_t1, im_tg, s_tg, acts, t0, t1, tg)
            self.t0, self.t1, self.tg = t0, t1, tg

        self.image_pairs = torch.stack([im_t0, im_tg], dim=1)
        if self._hp.low_dim:
            self.image_pairs_cat = torch.cat([s_t0, s_t1, s_tg], dim=1)
        else:
            self.image_pairs_cat = torch.cat([im_t0, im_t1, im_tg], dim=1)

        return self.image_pairs_cat, acts

    def select_close_arm_goals(self, states, s_t0):
        # States is [B, T, N]
        # get arm states
        arm_states = states[:, :, :9]
        tsteps = states.shape[1]
        s_t0 = s_t0[:, :9]
        out = []
        for i, state in enumerate(s_t0):
            distances = torch.norm(arm_states - state, dim=2) #distances is [B, T]
            distances = torch.cat((distances[:i], distances[i+1:]), dim=0) #remove batch that traj is from
            distances = distances.view(-1) # flatten into [(B-1)*T]
            idx = torch.argmin(distances)
            if idx >= tsteps * i:
                idx += tsteps
            out.append(states[idx//tsteps, idx % tsteps])
        return torch.stack(out, dim=0)

    def compute_action_samples(self, image_pairs, network, parallel=True, return_actions=False, detach_grad=True, use_hp=True):
        if not parallel:
            qs = []
            actions_l = []
            with torch.no_grad():
                for _ in range(self._hp.est_max_samples):
                    actions = torch.FloatTensor(image_pairs.size(0), self._hp.action_size).uniform_(
                        *self._hp.action_range).to(self._hp.device)
                    targetq = network(image_pairs, actions)
                    if detach_grad:
                        targetq = targetq.detach()
                    qs.append(targetq)
                    if return_actions:
                        actions_l.append(actions)
            qs = torch.stack(qs)
            if actions_l:
                actions = torch.stack(actions_l)
        else:
            num_samps = self._hp.est_max_samples if use_hp else 100
            image_pairs_rep = image_pairs[None]  # Add one dimension
            repeat_shape = [num_samps] + [1] * len(image_pairs.shape)
            image_pairs_rep = image_pairs_rep.repeat(*repeat_shape)  # [num_samp, B, s_dim]
            image_pairs_rep = image_pairs_rep.view(
                *[num_samps * image_pairs.shape[0]] + list(image_pairs_rep.shape[2:]))

            actions = torch.FloatTensor(image_pairs_rep.size(0), self._hp.action_size).uniform_(
                *self._hp.action_range).to(self.get_device())
            targetq = network(image_pairs_rep, actions)
            if detach_grad:
                targetq = targetq.detach()
            qs = targetq.view(num_samps, image_pairs.size(0), -1)

        if return_actions:
            actions = actions.view(num_samps, image_pairs.size(0), -1)
            return qs, actions
        else:
            return qs

    def get_td_error(self, image_pairs, model_output):
        max_qs = self.get_max_q(image_pairs)
        give_reward = (self.t1 >= self.tg).float()
        rew = give_reward * self._hp.binary_reward[1] + (1-give_reward) * self._hp.binary_reward[0]

        terminal_flag = (self.t1 >= self.tg).type(torch.ByteTensor).to(self._hp.device)

        self.train_batch_rews = rew

        discount = self._hp.gamma**(1.0*self._hp.n_step)

        if self._hp.terminal:
            target = (rew + discount * max_qs * (1 - terminal_flag))  # terminal value
        else:
            target = rew + discount * max_qs

        self.train_target_q_vals = target

        if self._hp.td_loss == 'mse':
            return sum([F.mse_loss(target, out) for out in model_output])
        elif self._hp.td_loss == 'huber':
            return sum([F.smooth_l1_loss(target, out) for out in model_output])

    def get_sg_pair(self, t):
        if self._hp.low_dim:
            x = self._hp.state_size
        else:
            x = 3
        return torch.cat((t[:, :x], t[:, 2 * x:]), axis=1)

    @property
    def cql_sign(self):
        return 1

    def loss(self, model_output):
        if self._hp.low_dim:
            image_pairs = self.images[:, self._hp.state_size:]
        else:
            image_pairs = self.images[:, 3:]

        losses = AttrDict()

        if self._hp.min_q:
            # Implement minq loss
            total_min_q_loss = []
            self.min_q_lse = 0
            for i, q_fn in enumerate(self.qnetworks):
                random_q_values = self.network_out_2_qval(self.compute_action_samples(self.get_sg_pair(self.images), q_fn, parallel=True, detach_grad=False))
                random_density = np.log(0.5 ** self._hp.action_size) # log uniform density
                random_q_values -= random_density
                min_q_loss = torch.logsumexp(random_q_values, dim=0) - np.log(self._hp.est_max_samples)
                min_q_loss = min_q_loss.mean()
                self.min_q_lse += min_q_loss
                total_min_q_loss.append(min_q_loss - model_output[i].mean())
            total_min_q_loss = self.cql_sign * torch.stack(total_min_q_loss).mean()
            if self._hp.min_q_lagrange and hasattr(self, 'log_alpha'):
                min_q_weight = self.log_alpha.exp().squeeze()
                total_min_q_loss -= self._hp.min_q_eps
            else:
                min_q_weight = self._hp.min_q_weight
            losses.min_q_loss = min_q_weight * total_min_q_loss
            self.min_q_lagrange_loss = -1 * losses.min_q_loss

        losses.bellman_loss = self._hp.bellman_weight * self.get_td_error(image_pairs, model_output)

        losses.total_loss = torch.stack(list(losses.values())).sum()

        if 'min_q_loss' in losses:
            losses.min_q_loss /= min_q_weight # Divide this back out so we can compare log likelihoods
        return losses

    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase, prefix=''):
        model_output = model_output[0] # take first Q fn if multiple
        if phase == 'train':
            if self._hp.optimize_actions == 'actor_critic':
                self.log_batch_statistics(f'{prefix}policy_update_actions', torch.abs(self.target_actions_taken), step, phase)
            self.log_batch_statistics(f'{prefix}target_q_values', self.train_target_q_vals, step, phase)
            self.log_batch_statistics(f'{prefix}q_values', model_output, step, phase)
            self.log_batch_statistics(f'{prefix}rewards', self.train_batch_rews, step, phase)
            if hasattr(self, 'min_q_lse'):
                self.log_batch_statistics(f'{prefix}min_q_lse', self.min_q_lse, step, phase)

        if hasattr(self, 'log_alpha'):
            self._logger.log_scalar(self.log_alpha.exp().item(), f'{prefix}alpha', step, phase)

        if log_images:
            image_pairs = self.image_pairs
            self._logger.log_single_tdist_classifier_image(image_pairs[:self._hp.batch_size//2], image_pairs[self._hp.batch_size//2:], model_output[:self._hp.batch_size],
                                                      '{}tdist{}'.format(prefix, "Q"), step, phase)
            if self._hp.add_negative_sample:
                self._logger.log_single_tdist_classifier_image(image_pairs[self._hp.batch_size:3*self._hp.batch_size // 2],
                                                               image_pairs[3*self._hp.batch_size // 2:],
                                                               model_output[self._hp.batch_size:],
                                                               '{}_negative_sample_tdist{}'.format(prefix, "Q"), step, phase)


    def log_batch_statistics(self, name, values, step, phase):
        self._logger.log_scalar(torch.mean(values).item(), f'{name}_mean', step, phase)
        self._logger.log_scalar(torch.median(values).item(), f'{name}_median', step, phase)
        if values.ndim != 0: #Don't log std for scalars
            self._logger.log_scalar(torch.std(values).item(), f'{name}_std', step, phase)

    def get_heatmap(self, data, x_range, y_range, side_len=101, frac=0.5):

        def linspace_to_slice(min, max, num):
            lsp = np.linspace(min, max, num)
            delta = lsp[1] - lsp[0]
            return slice(min, max + delta, delta)

        x, y = np.mgrid[
            linspace_to_slice(x_range[0], x_range[1], side_len),
            linspace_to_slice(y_range[0], y_range[1], side_len)
        ]

        import matplotlib.pyplot as plt
        data = np.reshape(data, (side_len, side_len)).copy()

        fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
        plt.plot(np.linspace(y_range[0], y_range[1], num=side_len), data[:, int(data.shape[0]*frac)])
        ax.set(xlabel='obj y pos', ylabel='expected distance')
        fig.canvas.draw()
        slice_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        slice_image = slice_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        slice_image = np.transpose(slice_image, (2, 0, 1))  # [C, H, W]
        plt.clf()

        data = data[:-1, :-1]
        cmap = plt.get_cmap('hot')
        fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
        im = ax.pcolormesh(x, y, data, cmap=cmap)
        fig.colorbar(im, ax=ax)
        # plt.subplots_adjust(left=0.3, right=0, bottom=0.3, top=0)
        ax.set(xlabel='obj x pos', ylabel='object y pos')
        plt.tight_layout()
        fig.canvas.draw()
        hmap = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        hmap_image = hmap.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        hmap_image = np.transpose(hmap_image, (2, 0, 1))  # [C, H, W]
        plt.clf()

        return np.concatenate((hmap_image, slice_image), axis=1)  # Concat heightwise

    def get_device(self):
        return self._hp.device

    def qval_to_timestep(self, qvals):
        return np.log(np.clip(qvals, 1e-5, 2)) / np.log(self._hp.gamma) + 1


def select_indices(tensor, indices, batch_offset=0):
    batch_idx = torch.arange(indices.shape[0]).cuda()
    if batch_offset != 0:
        batch_idx = torch.roll(batch_idx, batch_offset)
    if not isinstance(indices, torch.Tensor):
        indices = torch.LongTensor(indices).cuda()
    out = tensor[batch_idx, indices]
    return out


class QFunctionTestTime(QFunction):
    def __init__(self, overrideparams, logger=None):
        if 'gaussian_blur' in overrideparams:
            overrideparams.pop('gaussian_blur')
        super(QFunctionTestTime, self).__init__(overrideparams, logger)
        if self._hp.classifier_restore_path is not None:
            checkpoint = torch.load(self._hp.classifier_restore_path, map_location=torch.device('cpu'))
            self.load_state_dict(checkpoint['state_dict'])
            self.target_qnetworks.load_state_dict(self.qnetworks.state_dict())
            self.target_qnetworks.eval()
            if self.actor_critic:
                self.target_pi_net.load_state_dict(self.pi_net.state_dict())
                self.target_pi_net.eval()
        else:
            print('#########################')
            print("Warning Q function weights not restored during init!!")
            print('#########################')

    def _default_hparams(self):
        parent_params = super()._default_hparams()
        parent_params.add_hparam('classifier_restore_path', None)
        return parent_params
      
    def visualize_test_time(self, content_dict, visualize_indices, verbose_folder):
        pass
      
    def forward(self, inputs):
        qvals = super().forward(inputs)
        # Compute the log to get the units to be in timesteps
        return -qvals
