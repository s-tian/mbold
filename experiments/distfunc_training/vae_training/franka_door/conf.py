import os
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.classifier.utils.logger import TdistClassifierLogger
current_dir = os.path.dirname(os.path.realpath(__file__))
# from classifier_control.classifier.models.base_tempdistclassifier import BaseTempDistClassifier
from classifier_control.classifier.models.latent_space import LatentSpace
# from experiments.control.sim.multiroom2d import env_benchmark_conf

import imp

configuration = {
    'model': LatentSpace,
    'logger': TdistClassifierLogger,
    'data_dir': os.environ['VMPC_DATA'] + '/classifier_control/data_collection/sim/franka_desk',       # 'directory containing data.' ,
    'batch_size' : 32,
    'num_epochs': 1000,
    'seed': 1,
}

configuration = AttrDict(configuration)

data_config = AttrDict(
                img_sz=(64, 64),
                sel_len=-1,
                T=31)

model_config = {
#     'hidden_size':128,
#     'input_nc': 3,
    'goal_cond': False,
}
