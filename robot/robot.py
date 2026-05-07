import mujoco
import mujoco.viewer
import numpy as np
import yaml
from pathlib import Path

class Robot:
    def __init__(self, config_path:str="robot/configs/robot_config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        
        model_path = self.cfg["robot"]["model_path"]
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.joint_names = self.cfg["joints"]["name"]
        self.ee_site = self.cfg["robot"]["end_effector_link"]
        self.num_joints = self.cfg["robot"]["num_joints"]

        # cache joint and actuators IDs
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)
    
    def get_joint_positions(self) -> np.ndarray:
        return np.array([self.data.qpos[jid] for jid in self.joint_ids])
    
    def get_joint_velocities(self) -> np.ndarray:
        return np.array([self.data.qvel[jid] for jid in self.joint_ids])
    
    def get_end_effector_position(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()
    
    def get_ee_rotation(self) -> np.ndarray:
        return self.data.site_xmat[self.ee_site_id].copy().reshape(3, 3).copy()
    
    def get_jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        return jacp, jacr
    
    def set_ctrl(self, ctrl:np.ndarray):
        self.data.ctrl[:] = np.clip(ctrl,-1.0, 1.0)

    def set_joint_positions_direct(self, qpos:np.ndarray):
        for i, jid in enumerate(self.joint_ids):
            self.data.qpos[jid] = qpos[i]
        mujoco.mj_forward(self.model, self.data)

    @property
    def dof(self) ->float:
        return self.model.opt.timestep

