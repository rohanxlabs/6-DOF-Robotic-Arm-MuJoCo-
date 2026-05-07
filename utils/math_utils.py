import numpy as np
from scipy.spatial.transform import Rotation

def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to Euler angles (roll, pitch, yaw).
    
    Parameters:
    R (np.ndarray): A 3x3 rotation matrix.
    
    Returns:
    np.ndarray: A 1D array containing the Euler angles [roll, pitch, yaw].
    """
    r = Rotation.from_matrix(R)
    return r.as_euler("xyz")

def euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to a rotation matrix.
    
    Parameters:
    euler (np.ndarray): A 1D array containing the Euler angles [roll, pitch, yaw].
    
    Returns:
    np.ndarray: A 3x3 rotation matrix.
    """
    r = Rotation.from_euler("xyz", euler)
    return r.as_matrix()
def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to a quaternion.
    
    Parameters:
    R (np.ndarray): A 3x3 rotation matrix.
    
    Returns:
    np.ndarray: A 1D array containing the quaternion [x, y, z, w].
    """
    r = Rotation.from_matrix(R)
    xyzw = r.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])  # Return in [x, y, z, w] format

def transform_point(point: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Transform a point using a rotation and translation.
    
    Parameters:
    point (np.ndarray): A 3D point to be transformed.
    R (np.ndarray): A 3x3 rotation matrix.
    t (np.ndarray): A 3D translation vector.
    
    Returns:
    np.ndarray: The transformed point.
    """
    return R @ point + t

def compute_pose_error(
        current_pos: np.ndarray, target_pos: np.ndarray,
        current_R: np.ndarray, target_R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pos_error = target_pos - current_pos
    R_error = target_R @ current_R.T
    r = Rotation.from_matrix(R_error)
    rot_error = r.as_rotvec()

    return pos_error, rot_error
