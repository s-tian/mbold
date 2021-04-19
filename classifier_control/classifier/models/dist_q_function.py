from contextlib import contextmanager
import numpy as np
import pdb
import torch
from classifier_control.classifier.utils.general_utils import AttrDict
import torch.nn as nn
import torch.nn.functional as F

from classifier_control.classifier.models.q_function import QFunction


class DistQFunction(QFunction):

    def __init__(self, overrideparams, logger=None):
        super().__init__(overrideparams, logger)

    @property
    def num_network_outputs(self):
        return self._hp.num_bins

    def _default_hparams(self):
        default_dict = AttrDict({
            'num_bins': 10,
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def network_out_2_qval(self, softmax_values):
        """
        :param softmax_values: Tensor of softmax values of dimension [..., num_bins]
        :return: Tensor of dimension [...] containing scalar q values. In this case, the expectation.
        """
        qval = torch.sum((1 + torch.arange(softmax_values.shape[-1])[None]).float().to(self._hp.device) * softmax_values, -1)
        return qval

    def get_best_of_qs(self, qvals):
        """
        The reason why this function exists is because this operation is a _min_
        for the distributional case, and a max for the standard case
        :param qvals: Tensor of qvalues of dimension [N, Qs]
        :return: tuple of maximum qvalue and max Q index
        """
        return torch.min(qvals, dim=0)

    def get_worst_of_qs(self, qvals):
        return torch.max(qvals, dim=0)

    @property
    def cql_sign(self):
        return -1

    def actor_loss(self):
        image_pairs = self.get_sg_pair(self.images)
        self.target_actions_taken = self.pi_net(image_pairs)
        return self.network_out_2_qval(self.qnetworks[0](image_pairs, self.target_actions_taken)).mean()

    def get_td_error(self, image_pairs, model_output):
        qvals, target_q_distribution = self.get_max_q(image_pairs, return_raw=True)

        ## Shift Q*(s_t+1) to get Q*(s_t)
        shifted = torch.zeros(target_q_distribution.size()).to(self._hp.device)
        shifted[:, 1:] = target_q_distribution[:, :-1]
        shifted[:, -1] += target_q_distribution[:, -1]
        terminal_flag = (self.t1 >= self.tg).type(torch.ByteTensor).to(self._hp.device)
        self.train_batch_rews = terminal_flag.type(torch.FloatTensor)
        terminal_flag = terminal_flag.unsqueeze(-1).expand(-1, self._hp.num_bins)
        isg = torch.zeros((terminal_flag.shape[0], self._hp.num_bins)).to(self._hp.device)
        isg[:, 0] = 1
        ## If next state is goal then target should be 0, else should be shifted
        target = (terminal_flag * isg) + ((1 - terminal_flag) * shifted)

        self.train_target_q_vals = self.network_out_2_qval(target)

        ## KL between target and output
        loss = 0
        for network_out in self.network_out:
            log_q = network_out.clamp(1e-5, 1 - 1e-5).log()
            log_t = target.clamp(1e-5, 1 - 1e-5).log()
            kl_d = (target * (log_t - log_q)).sum(1).mean()
            loss += kl_d
        return loss

class DistQFunctionTestTime(DistQFunction):
    def __init__(self, overrideparams, logger=None):
        super(DistQFunctionTestTime, self).__init__(overrideparams, logger)
        if 'gaussian_blur' in overrideparams:
            overrideparams.pop('gaussian_blur')
        super(DistQFunctionTestTime, self).__init__(overrideparams, logger)
        if self._hp.classifier_restore_path is not None:
            checkpoint = torch.load(self._hp.classifier_restore_path, map_location=self._hp.device)
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
      ## Qvals represent distance now (lower is better), so directly return cost (no negative needed)
      return qvals
