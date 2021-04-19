""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.policy.random.gaussian import GaussianPolicy
from visual_mpc.agent.general_agent import GeneralAgent
# from classifier_control.environments.sim.cartgripper.cartgripper_xz import CartgripperXZ
from classifier_control.environments.sim.tabletop.tabletop_reacher import TabletopReacher
import numpy as np

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from visual_mpc.policy.random.sampler_policy import SamplerPolicy


env_params = {
    # resolution sufficient for 16x anti-aliasing
    'viewer_image_height': 48,
    'viewer_image_width': 64,
    'textured': True,
    #     'difficulty': 'm',
}


agent = {
    'type': GeneralAgent,
    'env': (TabletopReacher, env_params),
    'T': 30,
    'make_final_gif_freq':100,
    'rejection_sample': True,
}

policy = {
    'type' : SamplerPolicy,
    'nactions': 100,
    'initial_std':  [0.6, 0.6, 0.3, 0.3],
}

config = {
    'traj_per_file':1,  #28,
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 100,
    'agent': agent,
    'policy': policy,
    #'save_format': ['hdf5', 'raw', 'tfrec'],
    #'save_format': ['raw'],
}
