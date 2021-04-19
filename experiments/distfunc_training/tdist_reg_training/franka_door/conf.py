import os
from classifier_control.classifier.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from classifier_control.classifier.models.tempdist_regressor import TempdistRegressor, TempdistRegressorTestTime
from classifier_control.classifier.utils.logger import TdistRegressorLogger


configuration = {
    'model': TempdistRegressor,
    'model_test': TempdistRegressorTestTime,
    'logger': TdistRegressorLogger,
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

}
