import numpy as np
from robot.robot import Robot

class IKController:
    def __init__(
            self,
            robot: Robot,
            step_size: float = 0.5,
            tol: float = 0.001,
            max_iter: int = 100,
            damping: float = 0.01,
    ):
        self.robot = robot
        self.step_size = step_size
        self.tol = tol
        self.max_iter = max_iter
        self.damping = damping

    def solve(self, target_pos: np.ndarray) -> np.ndarray:
        q = self.robot.get_joint_positions()
        for _ in range(self.max_iter):
            ee_pos = self.robot.get_end_effector_position()
            error = target_pos - ee_pos
            if np.linalg.norm(error) < self.tol:
                return q
            
            jacp,_ = self.robot.get_jacobian()
            j = jacp[:,:self.robot.num_joints]

            JJt = j @ j.T
            damped = JJt + self.damping ** 2 * np.eye(3)
            dq = self.step_size * j.T @ np.linalg.solve(damped, error)
            q += dq
            q=self._clip_joints(q)
            self.robot.set_joint_positions_direct(q)

        return q
    
    def _clip_joints(self, q: np.ndarray) -> np.ndarray:
        limits = self.robot.cfg["joints"]["limits"]
        for i, name in enumerate(self.robot.joint_names):
            lo, hi = limits[name]
            q[i] = np.clip(q[i], lo, hi)
        return q
    def compute_ctrl(self, target_pos: np.ndarray) -> np.ndarray:
        from control.joint_controller import JointController
        target_q = self.solve(target_pos)
        if target_q is None:
            return np.zeros(self.robot.num_joints)
        
        jc=JointController(self.robot)
        return jc.compute(target_q)
