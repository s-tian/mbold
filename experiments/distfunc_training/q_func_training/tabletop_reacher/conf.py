import os
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.classifier.utils.logger import TdistClassifierLogger
current_dir = os.path.dirname(os.path.realpath(__file__))
from classifier_control.classifier.models.q_function import QFunction, QFunctionTestTime

import imp

configuration = {
    'model': QFunction,
    'model_test': QFunctionTestTime,
    'logger': TdistClassifierLogger,
    'data_dir': os.environ['VMPC_DATA'] + '/classifier_control/data_collection/sim/tabletop-reacher',
    'batch_size': 32,
    'num_epochs': 305,
    'lr': 3e-4,
    'seed': 6,
}

configuration = AttrDict(configuration)

data_config = AttrDict(
                img_sz=(64, 64),
                sel_len=-1,
                T=31)

model_config = {
    'gamma':0.8,
    'action_size': 4,
    'optimize_actions': 'actor_critic',
    'target_network_update': 'polyak',
    'sg_sample': 'geometric',
    'geom_sample_p': 0.3,
    'binary_reward': [1, 10],
    'twin_critics': True,
    'log_control_proxy': False,
    'add_action_noise': True,
    'negative_sample_type': 'nn_idx',
    'min_q': True,
    'min_q_lagrange': True,
    'min_q_eps': 3.0,
    'est_max_samples': 10,
}

