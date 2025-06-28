"""
Utilities for coordinate transformations and mathematical operations related to poses.
"""

# Third-party imports
import numpy as np
from numpy.typing import NDArray

# Type aliases
from typing import TypeAlias

# Define type aliases
QuaternionArray: TypeAlias = NDArray[np.float64]  # 4-element quaternion [x,y,z,w]
CovarianceMatrix: TypeAlias = NDArray[np.float64] # Square covariance matrix

def rotation_to_quaternion_covariance(
rot_cov: CovarianceMatrix, 
current_quat: QuaternionArray
) -> CovarianceMatrix:
    """
    Convert 3x3 rotation covariance to 4x4 quaternion covariance using the proper Jacobian.
    This takes into account the current orientation.
    
    Parameters:
        rot_cov: 3x3 rotation covariance matrix (from ROS message)
        current_quat: Current quaternion [x,y,z,w] representing the orientation
    
    Returns:
        4x4 quaternion covariance matrix [x,y,z,w]
    """
    # Normalize quaternion to ensure unit length
    current_quat = current_quat / np.linalg.norm(current_quat)
    qx, qy, qz, qw = current_quat
    
    # Jacobian of quaternion wrt rotation angles
    J = np.array([
        [ qw/2,  qz/2, -qy/2],
        [-qz/2,  qw/2,  qx/2],
        [ qy/2, -qx/2,  qw/2],
        [-qx/2, -qy/2, -qz/2]
    ])
    
    # Transform covariance using the Jacobian
    quat_cov = J @ rot_cov @ J.T
    
    return quat_cov
