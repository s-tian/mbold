import numpy as np
import copy
import os

from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from visual_mpc.envs.mujoco_env.base_mujoco_env import BaseMujocoEnv


class FrankaDesk(BaseMujocoEnv, SawyerXYZEnv):
    """Tabletop Manip (Metaworld) Env"""

    def __init__(self, env_params_dict, reset_state=None):
        hand_low = (0.05, -0.75, 0.73)
        hand_high = (0.6, -0.48, 1.0)
        obj_low=(0.15, -0.4, 0.6)
        obj_high=(0.4, -0.6, 0.6)

        dirname = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        params_dict = copy.deepcopy(env_params_dict)
        _hp = self._default_hparams()
        for name, value in params_dict.items():
            print('setting param {} to value {}'.format(name, value))
            _hp.set_hparam(name, value)

        filename = os.path.join(dirname, "./playrooms/rooms/playroom/playroom.xml")

        BaseMujocoEnv.__init__(self, filename, _hp)
        SawyerXYZEnv.__init__(
                self,
                frame_skip=15,
                action_scale=1./5,
                hand_low=hand_low,
                hand_high=hand_high,
                model_name=filename
            )
        goal_low = self.hand_low
        goal_high = self.hand_high
        self.obj_low = obj_low
        self.obj_high = obj_high
        self._adim = 4
        self._hp = _hp
        self.liftThresh = 0.04
        self.max_path_length = 100
        self.hand_init_pos = np.array([0.6, -0.48, 1.0])

    def default_ncam():
        return 1

    def _default_hparams(self):
        default_dict = {
            'verbose': False,
            'difficulty': None,
            'render_imgs': True,
            'include_objects': False,
        }
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
          parent_params.add_hparam(k, default_dict[k])
        return parent_params
  
    default_mocap_quat = np.array([0, 1, 0.5, 0])

    def set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]

        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', self.default_mocap_quat.copy())
    
    def _reset_hand(self, goal=False):
        pos = self.hand_init_pos.copy()
        pos[0] = np.random.uniform(0.2, self.hand_high[0])
        pos[1] = np.random.uniform(-0.6, -0.5)
        pos[2] = np.random.uniform(self.hand_low[2], self.hand_high[2])
        for _ in range(20):
          self.data.set_mocap_pos('mocap', pos)
          self.data.set_mocap_quat('mocap', self.default_mocap_quat.copy())
          self.do_simulation([-1,1], self.frame_skip)
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def get_body_pos(self, bodyName):
        _id = self.model.body_names.index(bodyName)
        return self.data.body_xpos[_id].copy()

    def reset(self, reset_state=None):
        self._reset_hand()
        if reset_state is not None:
            if reset_state.shape[0] == 41:
                target_qpos = reset_state
                target_qvel = np.zeros_like(self.data.qvel)
            else:
                target_qpos = reset_state[:41]
                target_qvel = reset_state[41:]
            self.set_state(target_qpos, target_qvel)
        else:
            for i in range(3):
                self.targetobj = i
                init_pos = np.random.uniform(
                    self.obj_low,
                    self.obj_high,
                )
                if not self._hp.include_objects:
                    init_pos = [2, 2, 2]
                self.obj_init_pos = init_pos
                self.data.qpos[9+7*i:12+7*i] = init_pos

                self.data.qpos[40] = np.random.uniform(0, 0.3)
                for _ in range(10):
                     self.do_simulation([0.0, 0.0])
        self.update_mocap_pos()
        self._obs_history = []
        o = self._get_obs()
        self._reset_eval()

        return o, self.sim.data.qpos.flat.copy()

    def step(self, action):
        self.set_xyz_action(action[:3])
        for i in range(10):
            self.do_simulation([action[-1], -action[-1]])
        self.update_mocap_pos()
        self.do_simulation([action[-1], -action[-1]], 1)
        obs = self._get_obs()
        print('current', obs['state'][40])
        print(self._goal_obj_pose[-1])
        return obs

    def update_mocap_pos(self):
        self.data.set_mocap_pos('mocap', self.get_endeff_pos())

    def render(self):
        if not self._hp.render_imgs:
            return np.zeros((1, self._hp.viewer_image_height, self._hp.viewer_image_width, 3))
        return super().render().copy()[..., 25:-25, :]

    def set_goal(self, goal_obj_pose, goal_arm_pose):
        print(f'Setting goals to {goal_obj_pose} and {goal_arm_pose}!')
        super(FrankaDesk, self).set_goal(goal_obj_pose, goal_arm_pose)

    def get_distance_score(self):
        """
        :return:  mean of the distances between all objects and goals
        """
        curr_drawer_pos = self.sim.data.qpos[-1]
        goal_drawer_pos = self._goal_obj_pose[-1]
        drawer_dist = np.abs(curr_drawer_pos-goal_drawer_pos)
        print(f'Door distance score is {drawer_dist}')
        return drawer_dist

    def has_goal(self):
        return True

    def get_endeff_pos(self):
        return self.get_body_pos('hand').copy()

    def goal_reached(self):
        og_pos = self._obs_history[0]['qpos']
        return np.abs(og_pos[40] - self.sim.data.qpos.flat[40]) > 0.25

    def _get_obs(self):
        obs = {}
        #joint poisitions and velocities
        obs['qpos'] = copy.deepcopy(self.sim.data.qpos[:].squeeze())
        obs['qvel'] = copy.deepcopy(self.sim.data.qvel[:].squeeze())
        obs['gripper'] = self.get_endeff_pos()
        obs['state'] = np.concatenate([copy.deepcopy(self.sim.data.qpos[:].squeeze()),
                                       copy.deepcopy(self.sim.data.qvel[:].squeeze())])
        obs['object_qpos'] = copy.deepcopy(self.sim.data.qpos[9:].squeeze())

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
        return self._get_obs()
  
    def get_goal(self):
        return self.goalim
  
    def has_goal(self):
        return True

    def reset_model(self):
        pass


def generate_test_set(data_dir):
    """
    Using an existing set of trajectories collected in this env, generate a set of test tasks using random starting and
    ending door locations. Note that this _does_ modify the data (pkl files and images) in the inputted directory.
    """

    import pickle
    import cv2
    env_params = {
        # resolution sufficient for 16x anti-aliasing
        'viewer_image_height': 192,
        'viewer_image_width': 256,
    }
    env = FrankaDesk(env_params)
    env.reset()
    for i in range(100):
        dir = f'{data_dir}/raw/traj_group0/traj{i}/'
        with open(dir + 'obs_dict.pkl', 'rb') as f:
            obs_dict = pickle.load(f)
        with open(dir + 'agent_data.pkl', 'rb') as file:
            agent_data = pickle.load(file)
        door_center = np.random.uniform(0.10, 0.30)
        # Generate initial and goal door position
        if np.random.random() < 0.5 or door_center > 0.15:
            door_init, door_final = door_center - 0.15, door_center + 0.15
        else:
            door_final, door_init = door_center - 0.15, door_center + 0.15

        state_final = obs_dict['state'][-1]
        state_final[40] = door_final
        # Set new final state using generated door position
        env.reset(state_final)
        env._reset_hand()

        obs_dict['state'][-1] = env._get_obs()['state']
        obs_dict['object_qpos'][-1] = env._get_obs()['object_qpos']
        rend = env.render()[0]
        # Render new final image
        cv2.imwrite(dir + 'images0/im_30.png', cv2.resize(rend[:, :, ::-1], (64, 64), interpolation=cv2.INTER_AREA))
        state = obs_dict['state'][-1][:]
        state[40] = door_init
        env.reset(state)
        # Render new starting image
        rend = env.render()[0]
        cv2.imwrite(dir + 'images0/im_0.png', cv2.resize(rend[:, :, ::-1], (64, 64), interpolation=cv2.INTER_AREA))
        obs_dict['state'][0] = state

        # Update reset state for controller to load from
        agent_data['reset_state'] = state[:41]

        # Write back files
        with open(dir + 'obs_dict.pkl', 'wb') as f:
            pickle.dump(obs_dict, f)
        with open(dir + 'agent_data.pkl', 'wb') as file:
            pickle.dump(agent_data, file)

