from visual_mpc.envs.mujoco_env.base_mujoco_env import BaseMujocoEnv
import numpy as np
import visual_mpc.envs as envs
from visual_mpc.envs.mujoco_env.util.create_xml import create_object_xml, create_root_xml, clean_xml
import copy
from pyquaternion import Quaternion
import os
from visual_mpc.utils.im_utils import npy_to_mp4

class SimpleMaze(BaseMujocoEnv):
  """Simple Maze Navigation Env"""

  fixed_w1 = -0.1
  fixed_w2 = 0.1

  def __init__(self, env_params_dict, reset_state=None):
    params_dict = copy.deepcopy(env_params_dict)
    _hp = self._default_hparams()
    for name, value in params_dict.items():
      print('setting param {} to value {}'.format(name, value))
      _hp.set_hparam(name, value)
      
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'simple_maze.xml')
    super().__init__(filename, _hp)
    self._adim = 2
    self.difficulty = _hp.difficulty
    self._hp = _hp

  def default_ncam():
    return 1

  def _default_hparams(self):
    default_dict = {'verbose':False, 'difficulty': None, 'fix_walls': False}
    parent_params = super()._default_hparams()
    for k in default_dict.keys():
      parent_params.add_hparam(k, default_dict[k])
    return parent_params
  
  def reset(self, reset_state=None):
    self.t = 0
    #if reset_state is not None:
    #  self.sim.data.qpos[:] = reset_state
    #  self.sim.data.qvel[:]= 0
    #  self.sim.step()
    if False:
      pass
    else:
      if self.difficulty is None:
        self.sim.data.qpos[0] = np.random.uniform(-0.27, 0.27)
      elif self.difficulty == 'e':
        self.sim.data.qpos[0] = np.random.uniform(0.15, 0.27)
      elif self.difficulty == 'm':
        self.sim.data.qpos[0] = np.random.uniform(-0.15, 0.15)
      elif self.difficulty == 'h':
        self.sim.data.qpos[0] = np.random.uniform(-0.27, -0.15)
      self.sim.data.qpos[1] = np.random.uniform(-0.27, 0.27)

      self.goal = np.zeros((2,))
      self.goal[0] = np.random.uniform(-0.27, 0.27)
      self.goal[1] = np.random.uniform(-0.27, 0.27)

      # Randomize wal positions
      if self._hp.fix_walls:
        w1, w2 = self.fixed_w1, self.fixed_w2
      else:
        w1 = np.random.uniform(-0.2, 0.2)
        w2 = np.random.uniform(-0.2, 0.2)
  #     print(self.sim.model.geom_pos[:])
  #     print(self.sim.model.geom_pos[:].shape)
      self.sim.model.geom_pos[5, 1] = 0.25 + w1
      self.sim.model.geom_pos[7, 1] = -0.25 + w1
      self.sim.model.geom_pos[6, 1] = 0.25 + w2
      self.sim.model.geom_pos[8, 1] = -0.25 + w2
    return self._get_obs(), self.sim.data.qpos.flat.copy()

  def step(self, action):
    self.sim.data.qvel[:] = 0
    self.sim.data.ctrl[:] = action
    for _ in range(500):
      self.sim.step()
    obs = self._get_obs()
#     import matplotlib.pyplot as plt
#     plt.imsave(f'ims/obs{self.t}.png', obs['images'][0])
    self.t += 1
    self.sim.data.qvel[:] = 0
    return obs
  
  def render(self):
    return super().render().copy()
  
  def _get_obs(self):
    obs = {}
    #joint poisitions and velocities
    obs['qpos'] = copy.deepcopy(self.sim.data.qpos[:].squeeze())
    obs['qvel'] = copy.deepcopy(self.sim.data.qvel[:].squeeze())

    obs['state'] = np.concatenate([copy.deepcopy(self.sim.data.qpos[:self._sdim].squeeze()),
                                           copy.deepcopy(self.sim.data.qvel[:self._sdim].squeeze())])

    #copy non-image data for environment's use (if needed)
    self._last_obs = copy.deepcopy(obs)
    self._obs_history.append(copy.deepcopy(obs))

    #get images
    obs['images'] = self.render()
    obs['env_done'] = False

    return obs
  
  def valid_rollout(self):
    return True

  def current_obs(self):
    return self._get_obs(finger_force)

  def set_goal(self, obj_pose, arm_pose):
    return
    self.goal = arm_pose[:]

  def get_goal(self):
    #return self.goal[None]
    print(self.goal)
    curr_qpos = self.sim.data.qpos[:].copy()
    self.sim.data.qpos[:] = self.goal
    self.sim.step()
    goalim = self.render()
    self.sim.data.qpos[:] = curr_qpos
    self.sim.step()
    return goalim
  
  def has_goal(self):
    return True

  def goal_reached(self):
    d = np.sqrt(np.mean((self.goal - self.sim.data.qpos[:])**2))
    if d < 0.05:
      return True
    return False
   
  def get_distance_score(self):
    """
        :return:  mean of the distances between all objects and goals
        """
    d = np.sqrt(np.mean((self.goal - self.sim.data.qpos[:])**2))
    print("********", d)
    if d < 0.1:
      return 1.0
    else:
      return 0.0
