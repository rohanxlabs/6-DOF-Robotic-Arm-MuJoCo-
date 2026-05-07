import numpy as np
from robot.robot import Robot

class JointController:
    def __init__(self, robot: Robot, kp: float=100, kd: float=10):
        self.robot = robot
        self.kp = kp
        self.kd = kd

    def compute(self, target_qpos: np.ndarray) -> np.ndarray:
        q = self.robot.get_joint_positions()
        qd = self.robot.get_joint_velocities()

        error = target_qpos - q
        ctrl = self.kp * error - self.kd * qd

        ctrl = ctrl/100.0  # Scale down the control signal to prevent instability
        return np.clip(ctrl, -1.0, 1.0)  # Clip control signal to valid range
    
    def is_at_target(self, target_qpos: np.ndarray, tol: float=0.01) -> bool:
        q = self.robot.get_joint_positions()
        return np.allclose(q, target_qpos, atol=tol)
    
    def move_to(self, target_qpos: np.ndarray, max_steps: int=2000):
        for _ in range(max_steps):
            ctrl = self.compute(target_qpos)
            self.robot.set_ctrl(ctrl)
            if self.is_at_target(target_qpos):
                break