"""
This module provides functions for simulating sensor data, including position sensors
and zebra identification based on stripe patterns.
"""
# Standard library imports
from typing import List, Optional, Tuple, TypeAlias
import logging

# Third-party imports
import numpy as np
from numpy.typing import NDArray
import pandas as pd

# Local application imports
from multi_target_tracking import PosMeasurement

# Define type aliases
MeasurementArray: TypeAlias = NDArray[np.object_]  # Array of PosMeasurement objects

logger = logging.getLogger(__name__)

def simulate_position_sensor(
    df: pd.DataFrame,
    white_noise_std: Optional[List[float]] = None,
    random_walk: bool = False,
    random_walk_std: Optional[List[float]] = None,
    bounds: Optional[List[float]] = None,
    outlier_percentage: float = 0.0,
    outlier_std: Optional[List[float]] = None,
) -> Tuple[NDArray[np.float64], MeasurementArray]:
    """
    Simulate a position sensor by adding Gaussian white noise to x, y, z across the DataFrame.
    Optionally adds a bounded random walk and random outliers. The function returns a list of 
    PosMeasurement objects.
    
    Args:
        df (pd.DataFrame): DataFrame with at least a 'time' column and x, y, z columns.
        white_noise_std (List[float], optional): Standard deviations for the white noise for x, y, z.
            Default: [2.5, 2.5, 2].
        random_walk (bool): If True, a cumulative random walk is added. Default: False.
        random_walk_std (List[float], optional): Standard deviations for the walk for x, y, z.
            Default: [1, 1, 1] if random_walk is enabled.
        bounds (List[float], optional): Maximum allowed deviation from true value for x, y, z.
            Default: [5, 5, 3] if random_walk is enabled.
        outlier_percentage (float): Percentage of measurements that should be outliers.
            Default: 0.0 (no outliers).
        outlier_std (List[float], optional): Maximum magnitude of outlier offset for x, y, z.
            Default: [10, 10, 5] if outliers are enabled.
            
    Returns:
        Tuple[NDArray[np.float64], MeasurementArray]: A tuple containing:
            - timestamps (NDArray[np.float64]): Array of timestamps for each measurement.
            - measurements (MeasurementArray): Array of simulated position measurements.
    """
    # Set default white noise magnitudes
    if white_noise_std is None:
        white_noise_std = np.array([2.5, 2.5, 2])
    else:
        white_noise_std = np.array(white_noise_std, dtype=float)
        
    if random_walk and random_walk_std is None:
        random_walk_std = [1, 1, 1]
    random_walk_std = np.array(random_walk_std) if random_walk else np.array([0, 0, 0])
    
    if random_walk and bounds is None:
        bounds = [5, 5, 3]
    bounds = np.array(bounds) if bounds is not None else np.array([float('inf')] * 3)
    
    # Set default outlier parameters
    if outlier_percentage < 0 or outlier_percentage > 1:
        raise ValueError("Outlier percentage must be between 0 and 1")
        
    if outlier_percentage > 0 and outlier_std is None:
        outlier_std = [10, 10, 5]
    outlier_std = np.array(outlier_std) if outlier_std is not None else np.array([0, 0, 0])
    
    # Sort DataFrame by time
    df_sorted = df.sort_values("time")

    # Initialize an empty lists to store outputs
    timestamps = []
    measurements = []

    # Initialize cumulative random walk offset for x, y, z
    cumulative_offset = np.zeros(3)
    
    for _, row in df_sorted.iterrows():
        # Generate Gaussian white noise
        white_noise = np.random.normal(0, white_noise_std)
        
        # Random walk component with bounds
        if random_walk:
            walk_step = np.random.normal(0, random_walk_std)
            cumulative_offset += walk_step
            
            # Clip the offset to bounds as a safety measure
            cumulative_offset = np.clip(cumulative_offset, -bounds, bounds)
        
        # Initialize outlier offset as zeros
        outlier_offset = np.zeros(3)
        
        # Determine if this measurement should be an outlier
        if outlier_percentage > 0 and np.random.random() < outlier_percentage:
            # Generate random outlier offset within max magnitude
            outlier_offset = np.random.uniform(-outlier_std, outlier_std)
        
        # Update position with all noise components
        position = np.array([row['x'], row['y'], row['z']]) + white_noise + cumulative_offset + outlier_offset
        
        # Add the white noise variance to the diagonal
        noise_cov = np.diag(white_noise_std**2)
        
        # Add the random walk variance contribution if enabled
        if random_walk:
            # Add bounded random walk variance to the diagonal
            # bounded_variance = np.minimum(random_walk_std**2, (bounds/3)**2)  # use 3-sigma rule
            bounded_variance = (bounds/2)**2 # use 2-sigma rule
            noise_cov += np.diag(bounded_variance)
        
        # Create measurement
        timestamps.append(row['time'])
        measurements.append(PosMeasurement(
            position=position,
            covariance=noise_cov
        ))
    
    # Convert lists to numpy arrays for consistency
    timestamps = np.array(timestamps, dtype=float)
    measurements = np.array(measurements, dtype=object)
    
    return timestamps, measurements

def create_zebra_db(
        zebras_in_db: int = 10,
        zebras_in_gt: int = 5,
        zebra_ids: Optional[List[int]] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Creates a database of zebra IDs and a ground truth set of zebra IDs.
    
    Args:
        zebras_in_db (int): Number of zebras in the database.
        zebras_in_gt (int): Number of zebras in the ground truth set.
        zebra_ids (List[int], optional): List of zebra IDs. If None, random IDs are generated.
        
    Returns:
        Tuple[NDArray[np.int64], NDArray[np.int64]]: Tuple containing two numpy arrays:
            - Zebra IDs in the database
            - Zebra IDs in the ground truth set
    """
    if zebra_ids is None:
        zebra_ids = np.arange(zebras_in_db)
    
    if zebras_in_gt > zebras_in_db:
        raise ValueError("Ground truth zebras cannot exceed database zebras")
    
    gt_zebra_ids = np.random.choice(zebra_ids, size=zebras_in_gt, replace=False)
    
    return zebra_ids, gt_zebra_ids

def simulate_zebra_identification(
    zebra_trajectories: List[MeasurementArray],
    zebra_ids_db: NDArray[np.int64],
    zebra_ids_meas: NDArray[np.int64],
    mean_confidence: float = 0.3,
    confidence_std: float = 0.15,
    outlier_percentage: float = 0.0,
) -> List[MeasurementArray]:
    """
    Simulates an ML algorithm that identifies zebras based on stripe patterns.
    
    For each position measurement, the function assigns confidence scores to each zebra ID:
    - For regular measurements: The true zebra ID receives the highest confidence value, 
      with other zebra IDs sharing the remaining probability mass proportionally
    - For outliers (controlled by outlier_percentage): Either a random zebra ID receives 
      the highest confidence (simulating misidentification) or a completely random confidence 
      distribution is generated
    
    Args:
        zebra_trajectories (List[MeasurementArray]): List of arrays containing PosMeasurement objects.
            Each array represents the trajectory of one zebra.
        zebra_ids_db (NDArray[np.int64]): Numpy array of all unique zebra IDs in the database.
        zebra_ids_meas (NDArray[np.int64]): Numpy array of zebra IDs corresponding to each trajectory
            in `zebra_trajectories`.
        mean_confidence (float, optional): Mean confidence value for the true zebra ID
            (or random ID for outliers). Default: 0.3
        confidence_std (float, optional): Standard deviation of the confidence value, creating
            a normal distribution around the mean. Default: 0.15
        outlier_percentage (float, optional): Percentage of measurements that should be outliers
            with the highest confidence assigned to a random zebra ID or using a random distribution. 
            Default: 0.0
            
    Returns:
        List[MeasurementArray]: The input trajectories with added ID confidence scores in each measurement.
        Each measurement.id contains an array of confidence values corresponding to zebra IDs in zebra_ids_db.
        
    Raises:
        ValueError: If zebra_trajectories is empty, if zebra_ids_meas and zebra_trajectories have
            different lengths, if outlier_percentage is not between 0 and 1, or if mean_confidence
            is not between 0 and 1 (exclusive).
    """
    if len(zebra_trajectories) < 1:
        raise ValueError("At least one zebra trajectory is required")
    
    if len(zebra_ids_meas) != len(zebra_trajectories):
        raise ValueError("Zebra IDs and trajectories must have the same length")
    
    if outlier_percentage < 0 or outlier_percentage > 1:
        raise ValueError("Outlier percentage must be between 0 and 1")
        
    if mean_confidence <= 0 or mean_confidence >= 1:
        raise ValueError("Mean confidence must be between 0 and 1 exclusive")

    # For each zebra trajectory
    for zebra_id, trajectory in zip(zebra_ids_meas, zebra_trajectories):

        total_zebras = len(zebra_ids_db)
        
        # For each measurement in the trajectory
        for measurement in trajectory:
            # Determine if this measurement should be an outlier
            is_outlier = np.random.random() < outlier_percentage
            
            # Generate a high confidence value with controlled mean and deviation
            high_confidence = np.random.normal(mean_confidence, confidence_std)
            # Ensure the confidence is within the valid range
            high_confidence = np.clip(high_confidence, 0.01, 0.99)
            
            def generate_confidences(conf_idx: int, high_conf: float, total: int) -> NDArray[np.float64]:
                """Generate a confidence distribution with high confidence at conf_idx and the others scaled accordingly."""
                other_vals = np.random.random(total - 1)
                other_vals *= (1 - high_conf) / np.sum(other_vals)
                return np.insert(other_vals, conf_idx, high_conf)

            if is_outlier:
                if np.random.random() < 0.5:
                    # Assign high confidence randomly
                    rand_idx = np.random.choice(zebra_ids_db, size=1)[0]
                    confidences = generate_confidences(rand_idx, high_confidence, total_zebras)
                else:
                    # Use an entirely random distribution
                    confidences = np.random.random(total_zebras)
            else:
                # High confidence at the true zebra position
                zebra_idx = np.where(zebra_ids_db == zebra_id)[0]
                confidences = generate_confidences(zebra_idx, high_confidence, total_zebras)
            
            # Normalize to ensure sum is 1.0
            confidences = confidences / np.sum(confidences)
            measurement.id = confidences
    
    return zebra_trajectories