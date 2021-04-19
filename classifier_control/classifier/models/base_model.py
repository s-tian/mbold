import os
import yaml
import torch
import torch.nn as nn
from tensorflow.contrib.training import HParams
from torch.optim import Adam, SGD

from classifier_control.classifier.utils.layers import LayerBuilderParams
from classifier_control.classifier.utils.general_utils import AttrDict


class BaseModel(nn.Module):
    def __init__(self, logger):
        super().__init__()
        self._hp = self._default_hparams()
        print('hp', self._hp)
        self._logger = logger

    def override_defaults(self, params):
        for name, value in params.items():
            print('overriding param {} to value {}'.format(name, value))
            if name == 'ignore_same_as_default': continue
            if value == getattr(self._hp, name) and 'ignore_same_as_default' not in params:
                raise ValueError("attribute is {} is identical to default value!!".format(name))
            self._hp.set_hparam(name, value)

    def _default_hparams(self):
        # Data Dimensions
        default_dict = AttrDict({
            'batch_size': -1,
            'max_seq_len': -1,
            'n_actions': -1,
            'state_dim': -1,
            'input_nc': 3,  # number of input feature maps
            'device':None,
            'data_conf':None,
            'img_sz': None,
            'goal_cond':True,
        })
        
        # Network params
        default_dict.update({
            'use_convs': True,
            'use_batchnorm': True,  # TODO deprecate
            'normalization': 'batch',
            'spatial_softmax': True,
            'use_spectral_norm': False,
        })

        # add new params to parent params
        parent_params = HParams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])

        return parent_params

    def postprocess_params(self):
        self._hp.add_hparam('builder', LayerBuilderParams(
            self._hp.use_convs, self._hp.use_batchnorm, self._hp.normalization, self._hp.use_spectral_norm))
        self._hp.img_sz = self._hp.data_conf['img_sz']

    def init_optimizers(self, hp):
        self.optimizer = Adam(self.parameters(), lr=hp.lr)

    def optim_step(self, output):
        losses = self.loss(output)
        self.optimizer.zero_grad()
        losses.total_loss.backward()
        self.optimizer.step()
        return losses

    def build_network(self):
        raise NotImplementedError("Need to implement this function in the subclass!")

    def forward(self, inputs):
        raise NotImplementedError("Need to implement this function in the subclass!")

    def loss(self, model_output):
        raise NotImplementedError("Need to implement this function in the subclass!")

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        # Log generally useful outputs
        self._log_losses(losses, step, phase)

        if phase == 'train':
            self.log_gradients(step, phase)
            
        for module in self.modules():
            if hasattr(module, '_log_outputs'):
                module._log_outputs(model_output, inputs, losses, step, log_images, phase)

    def _log_losses(self, losses, step, phase):
        for name, loss in losses.items():
            self._logger.log_scalar(loss, name, step, phase)

    def _load_weights(self, weight_loading_info):
        """
        Loads weights of submodels from defined checkpoints + scopes.
        :param weight_loading_info: list of tuples: [(model_handle, scope, checkpoint_path)]
        """

        def get_filtered_weight_dict(checkpoint_path, scope):
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self._hp.device)
                filtered_state_dict = {}
                remove_key_length = len(scope) + 1      # need to remove scope from checkpoint key
                for key, item in checkpoint['state_dict'].items():
                    if key.startswith(scope):
                        filtered_state_dict[key[remove_key_length:]] = item
                if not filtered_state_dict:
                    raise ValueError("No variable with scope '{}' found in checkpoint '{}'!".format(scope, checkpoint_path))
                return filtered_state_dict
            else:
                raise ValueError("Cannot find checkpoint file '{}' for loading '{}'.".format(checkpoint_path, scope))

        print("")
        for loading_op in weight_loading_info:
            print(("=> loading '{}' from checkpoint '{}'".format(loading_op[1], loading_op[2])))
            filtered_weight_dict = get_filtered_weight_dict(checkpoint_path=loading_op[2],
                                                            scope=loading_op[1])
            loading_op[0].load_state_dict(filtered_weight_dict)
            print(("=> loaded '{}' from checkpoint '{}'".format(loading_op[1], loading_op[2])))
        print("")

    def log_gradients(self, step, phase):
        grad_norms = list([torch.norm(p.grad.data) for p in self.parameters() if p.grad is not None])
        if len(grad_norms) == 0:
            return
        grad_norms = torch.stack(grad_norms)

        self._logger.log_scalar(grad_norms.mean(), 'gradients/mean_norm', step, phase)
        self._logger.log_scalar(grad_norms.max(), 'gradients/max_norm', step, phase)

    def dump_params(self, path):
        with open(os.path.join(path, 'params.yaml'), 'w') as f:
            config = self._hp.values()
            config['data_conf'] = dict(config['data_conf'])
            yaml.dump(config, f)

