import time 
import mujoco
import mujoco.viewer
import numpy as np
import sys
sys.path.insert(0, ".")

from robot.robot import Robot
from control.ik_controller import IKController
from control.joint_controller import JointController

def main():
    print("starting IK control")
    robot = Robot()
    ik = IKController(robot)
    jc = JointController(robot,kp = 300,kd=30)

    targets = [
        np.array([0.25, 0.00, 0.45]),
        np.array([0.2, 0.1, 0.5]),
        np.array([0.3, -0.1, 0.4]),
        np.array([0.2, 0.1, 0.6]),
        np.array([0.3, 0.0, 0.5])
    ]

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.5

        for i,target_pos in enumerate(targets):
            print(f"Moving to target {i+1}: {target_pos}")
            step=0
            while True:
             target_q = ik.compute_ctrl(target_pos)
             if target_q is None:
                print("IK solution not found for target:")
                break
             
             ctrl = jc.compute(target_q)
             robot.set_ctrl(ctrl)

             mujoco.mj_step(robot.model,robot.data)
             viewer.sync()

             ee = robot.get_end_effector_position()
             dist = np.linalg.norm(ee - target_pos)

             print(f"Step{step} | dist: {dist:.3f}",end="\r")

             step += 1

             if dist < 0.05:
                print(f"\nReached target {i+1}")
                time.sleep(0.5)
                break
             
             if step > 2000:
                print(F"\nTimeout (dist={dist:.3f})")
                break
             
        
        print("\nDone!")

        while True:
                mujoco.mj_step(robot.model,robot.data)
                viewer.sync()
                time.sleep(0.01)

if __name__ == "__main__":
    main()