""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from classifier_control.environments.sim.cartgripper.cartgripper_xz import CartgripperXZ
from visual_mpc.policy.cem_controllers.samplers.correlated_noise import CorrelatedNoiseSampler

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from classifier_control.cem_controllers.pytorch_classifier_controller import LearnedCostController
from classifier_control.environments.sim.franka_desk.franka_desk import FrankaDesk
from classifier_control.baseline_costs.image_mse_cost import ImageMseCost

env_params = {
    # resolution sufficient for 16x anti-aliasing
    'viewer_image_height': 192,
    'viewer_image_width': 256,
    'textured': True,
}


agent = {
    'type': BenchmarkAgent,
    'env': (FrankaDesk, env_params),
    'T': 30,
    'gen_xml': (True, 20),  # whether to generate xml, and how often
    # 'make_final_gif_freq':1,
    'start_goal_confs': os.environ['VMPC_DATA'] + '/classifier_control/data_collection/sim/franka_door_startgoal/raw',
    'num_load_steps':31,
    'image_height': 64,
}


policy = {
    'type': LearnedCostController,
    'replan_interval': 6,
    'nactions': 13,
    # 'num_samples': 200,
    'selection_frac': 0.05,
    'sampler': CorrelatedNoiseSampler,
    'initial_std':  [0.3, 0.3, 0.3, 0.15],
    'learned_cost': ImageMseCost,
    "vidpred_model_path": os.environ['VMPC_EXP'] + '/vpred_models/franka_desk/checkpoint_185000',
    'verbose_every_iter': True,
    'finalweight': -10000,
}

config = {
    'traj_per_file':1,  #28,
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 100,
    'agent': agent,
    'policy': policy,
    'save_data': False,
    'save_format': ['raw'],
}

