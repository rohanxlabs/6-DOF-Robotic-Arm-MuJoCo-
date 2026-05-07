from fileinput import filename

import numpy as np  
import mujoco
import cv2
from robot.robot import Robot

class Camera:
    def __init__(self,robot: Robot, width=640, height=480,camera_name:str="fixed_cam"):
        self.robot = robot
        self.width = width
        self.height = height
        self.camera_name = camera_name

        self.renderer = mujoco.Renderer(robot.model, height=height, width=width)

    def get_rgb(self) -> np.ndarray:
        self.renderer.update_scene(self.robot.data,camera=self.camera_name)
        return self.renderer.render().copy()
        
    
    def get_depth(self) -> np.ndarray:
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.robot.data,camera=self.camera_name)
        depth= self.renderer.render()
        self.renderer.disable_depth_rendering()
        return depth.copy()
    
    def get_segmentation(self) -> np.ndarray:
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.robot.data,camera=self.camera_name)
        seg = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        return seg.copy()
    
    def show(self,image: np.ndarray, window_name:str="Camera"):
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, bgr)
        cv2.waitKey(1)

    def save(self,image: np.ndarray, path:str):
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)

    def close(self):
        cv2.destroyAllWindows()