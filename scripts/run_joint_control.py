import time 
import mujoco
import mujoco.viewer
import numpy as np
import sys
sys.path.insert(0,".")

from robot.robot import Robot
from control.joint_controller import JointController

def main():
    print("Starting Simulation")
    robot = Robot()
    robot.reset()
    controller = JointController(robot, kp=150, kd=15)

    waypoints = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, -0.5, 0.3, 0.0, 0.0]),
        np.array([-0.5, 0.4, -0.3, 0.5, 0.3]),
        np.array([1.0, -0.8, 0.5, -0.5, -0.3]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    ]

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.5

        for i, target in enumerate(waypoints):
            print(f"\nMoving to waypoint {i+1}: {np.round(target,2)}")
            step = 0
            while True:
                ctrl= controller.compute(target)
                robot.set_ctrl(ctrl)
                mujoco.mj_step(robot.model,robot.data)
                viewer.sync
                step += 1


                if controller.is_at_target(target, tol=0.05) and step > 100:
                    ee = robot.get_end_effector_position()
                    print(f"Reached waypoint {i+1} after {step} steps | EE: {np.round(ee,2)}")
                    time.sleep(0.5)
                    break

                if step >= 15000:
                    final_error = np.max(np.abs(robot.get_joint_positions() - target))
                    print(f"Timeout reached for waypoint {i+1} | Error: {final_error:.4f}")
                    break

            print("\nCompleted waypoint")
            while True:
                mujoco.mj_step(robot.model,robot.data)
                viewer.sync()
                time.sleep(0.01)

if __name__ =="__main__":
    main()
        