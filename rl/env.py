import numpy as np
import mujoco 
import gymnasium as gym
from gymnasium import spaces
import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from robot.robot import Robot
from control.joint_controller import JointController

class RobotReachEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.robot = Robot()

        n_joints = self.robot.num_joints
        obs_dim = n_joints *2+3+3

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32)
        self.target_pos = np.array([0.3, 0.0, 0.5])  # Target position for the end-effector
        self.max_steps = 500
        self._step_count = 0

        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.robot.model, self.robot.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.reset()
        self._step_count = 0

        rng = np.random.default_rng(seed)
        self.target_pos = np.array([
            rng.uniform(0.2, 0.4),  # x
            rng.uniform(-0.2, 0.2), # y
            rng.uniform(0.3, 0.6)   # z
        ],dtype=np.float32)

        target_body_id = mujoco.mj_name2id(
            self.robot.model,mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        if target_body_id>=0:

           self.robot.model.body_pos[target_body_id] = self.target_pos
           mujoco.mj_forward(self.robot.model, self.robot.data)
        obs = self._get_obs()
        info={}

        return obs,info
    
    def step(self, action:np.ndarray):
        action = np.clip(action,-1.0,1.0)
        self.robot.set_ctrl(action)
        self.robot.step()
        self._step_count += 1

        obs = self._get_obs()
        reward,terminated = self._compute_reward()
        truncated = self._step_count >= self.max_steps
        info = {
            "distance":np.linalg.norm(
                self.robot.get_ee_rotation() - self.target_pos
            )
        }

        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        q = self.robot.get_joint_positions().astype(np.float32)
        qd = self.robot.get_joint_velocities().astype(np.float32)
        ee = self.robot.get_end_effector_position().astype(np.float32)
        tgt = self.target_pos.astype(np.float32)
        return np.concatenate([q, qd, ee, tgt])
    
    def _compute_reward(self) -> tuple[float, bool]:
        ee_pos = self.robot.get_end_effector_position()
        dist = np.linalg.norm(ee_pos - self.target_pos)
        reward = -dist
        terminated = False
        if dist < 0.02:
            reward += 10.0
            terminated = True

        return float(reward), terminated
    def render(self):
        if self.render_mode == "human" and hasattr(self,"viewer"):
            self.viewer.sync()
    def close(self):
        if hasattr(self,"viewer"):
            self.viewer.close()