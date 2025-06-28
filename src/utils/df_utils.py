"""
This module provides a collection of utility functions for processing and manipulating
pandas DataFrames, particularly those containing time-series data from robotics
applications (e.g., ROS bag data).

Key functionalities include:
- Data Sampling: Downsampling time-series data based on a specified rate.
- Timestamp Handling: Cleaning duplicate timestamps.
- Data Alignment: Reindexing and interpolating DataFrames to align timestamps,
  facilitating comparisons between different data sources (e.g., ground truth and estimates).
- Error Calculation: Computing position and pose errors between two datasets,
  along with statistical analysis of these errors (mean, std, RMS, percentiles, etc.).
- Data Conversion: Transforming custom measurement and track objects (from the 'mtt' package)
  into pandas DataFrames and vice-versa.
- Kinematic Estimation: Estimating velocity from position data using numerical differentiation.

These utilities are designed to streamline common data processing tasks in robotics
research and development, especially when working with tracking and localization data.
"""
# Standard library imports
from typing import List, Optional, Tuple, TypeAlias
import logging

# Third-party imports
import numpy as np
from numpy.typing import NDArray
import pandas as pd

# Local application imports
from multi_target_tracking import PosMeasurement, PosVelMeasurement, Track

# Define type aliases
MeasurementArray: TypeAlias = NDArray[np.object_]  # Array of PosMeasurement objects

# Configure module logger
logger = logging.getLogger(__name__)

# Dataframe column names for geometry_msgs/PoseWithCovarianceStamped
# cols_pose_with_cov_stamped = [
#     "time", "frame_id",
#     "x", "y", "z",
#     "qx", "qy", "qz", "qw",
#     "covariance"]

def sample_data(df: pd.DataFrame, sample_rate: float = 0.5) -> pd.DataFrame:
    """
    Samples one set of data every 'sample_rate' seconds from the given DataFrame.

    The DataFrame is expected to have a 'time' column (in seconds) derived from a rosbag 
    PoseWithCovarianceStamped message. For each sample_rate period, the first row encountered 
    is selected.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least the 'time' column.
        sample_rate (float, optional): Sampling sample_rate in seconds. Defaults to 0.5.

    Returns:
        pd.DataFrame: A new DataFrame with sampled rows.
    """
    # Group rows by time sample_rates and take the first row of each group
    groups = (df['time'] // sample_rate).astype(int)
    sampled_df = df.groupby(groups).first().reset_index(drop=True)
    return sampled_df


def clean_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate timestamps by keeping the first occurrence.
    
    Args:
        df (pd.DataFrame): DataFrame with possible duplicate timestamps
        
    Returns:
        pd.DataFrame: DataFrame with unique timestamps
    """
    if df['time'].duplicated().any():
        logger.warning(f"Found {df['time'].duplicated().sum()} duplicate timestamps. Keeping first occurrences.")
        return df.loc[~df['time'].duplicated()].copy()
    return df


def reindex_dataframe(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Interpolates the source DataFrame to match the timestamps of the target DataFrame.
    
    Args:
        source_df (pd.DataFrame): DataFrame to be interpolated, must contain 'time' column
        target_df (pd.DataFrame): DataFrame with the desired timestamps, must contain 'time' column
    
    Returns:
        pd.DataFrame: A new DataFrame with source_df data interpolated to target_df timestamps
    """
    # Input validation
    if 'time' not in source_df.columns or 'time' not in target_df.columns:
        raise ValueError("Both DataFrames must contain a 'time' column")
    
    # Clean duplicate timestamps
    logger.info("Cleaning duplicate timestamps in source_df...")
    source_df = clean_duplicate_timestamps(source_df)
    logger.info("Cleaning duplicate timestamps in target_df...")
    target_df = clean_duplicate_timestamps(target_df)
    
    # Check time range
    source_time_min, source_time_max = source_df['time'].min(), source_df['time'].max()
    target_time_min, target_time_max = target_df['time'].min(), target_df['time'].max()
    
    # Find overlapping time range
    overlap_start = max(source_time_min, target_time_min)
    overlap_end = min(source_time_max, target_time_max)
    
    if overlap_end <= overlap_start:
        raise ValueError(
            "No overlap between source and target timestamps. "
            f"Source range: [{source_time_min}, {source_time_max}], "
            f"Target range: [{target_time_min}, {target_time_max}]"
        )
    
    logger.info(f"Using overlapping time range: [{overlap_start}, {overlap_end}]")
    
    # Filter both dataframes to overlapping range
    source_df = source_df[(source_df['time'] >= overlap_start) & (source_df['time'] <= overlap_end)].copy()
    target_df = target_df[(target_df['time'] >= overlap_start) & (target_df['time'] <= overlap_end)].copy()

    # Interpolate source data to target times
    logger.info("Interpolating source_df to target_df timestamps...")
    source_df.set_index('time', inplace=True)
    source_df = source_df.reindex(target_df['time'], method='nearest')
    source_df.reset_index(inplace=True)
    logger.info("Interpolation complete.")
    
    return source_df


def calculate_position_error(gt_df: pd.DataFrame, est_df: pd.DataFrame, abs = False) -> pd.DataFrame:
    """
    Calculates the position error between ground truth and estimated poses.
    
    Args:
        gt_df (pd.DataFrame): Ground truth DataFrame with pose data
        est_df (pd.DataFrame): Estimated pose DataFrame
        abs (bool, optional): If True, return absolute values of errors. Defaults to False.
        
    Returns:
        pd.DataFrame: DataFrame containing the position errors with columns:
            - time: timestamp
            - x: position error in x
            - y: position error in y
            - z: position error in z
            - total: total position error (Euclidean distance)
    """
    logger.info("Calculating position errors...")
    
    # Initialize error DataFrame
    error_df = {'time': est_df['time'].copy()}
    
    # Position errors
    for coord in ['x', 'y', 'z']:
        error_df[coord] = []
        for time_idx, time in enumerate(est_df['time']):
            # Find the closest time index in gt_df
            gt_time_idx = np.abs(gt_df['time'] - time).idxmin()
            error = gt_df[coord][gt_time_idx] - est_df[coord][time_idx]
            if abs:
                error = np.abs(error)
            error_df[coord].append(error)
    
    error_df = pd.DataFrame(error_df)
    # Total position error (Euclidean distance)
    error_df['total'] = np.sqrt(
        error_df['x']**2 + 
        error_df['y']**2 + 
        error_df['z']**2
    )

    logger.debug("Position error calculation complete.")
    return error_df

def calculate_position_error_statistics(error_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates statistical measures for position errors.
    
    Args:
        error_df (pd.DataFrame): DataFrame containing position errors, typically 
                                from calculate_position_error function
        
    Returns:
        pd.DataFrame: DataFrame containing statistical measures for each error component:
            - std: standard deviation of errors
            - std_filtered: standard deviation without outliers (using MAE-based detection)
            - min: minimum error
            - max: maximum error
            - rms: root mean square error
            - mae: mean absolute error
            - outlier_ratio: ratio of outliers detected
    """
    # Error components to analyze
    error_components = ['x', 'y', 'z', 'total']
    
    # Initialize dictionary to store statistics
    stats = {}
    
    # calculate outliers
    abs_error = np.abs(error_df['total'])
    mae = np.mean(abs_error)
    outliers = np.abs(abs_error - mae) > 2 * abs_error.std()
    outlier_ratio = np.sum(outliers) / len(error_df['total'])

    # Calculate statistics for each component
    for component in error_components:
        if component in error_df.columns:
            abs_error = np.abs(error_df[component])
            mae = np.mean(abs_error)
            filtered_errors = abs_error[~outliers]
            std_filtered = filtered_errors.std()
            component_stats = {
                'std': error_df[component].std(),
                'std_filtered': std_filtered,
                'min': error_df[component].min(),
                'max': error_df[component].max(),
                'rms': np.sqrt(np.mean(error_df[component] ** 2)),
                'mae': mae,
                'outlier_ratio': outlier_ratio,
            }
            stats[component] = component_stats
    
    # Convert to DataFrame for better visualization
    stats_df = pd.DataFrame(stats)
    
    return stats_df

def calculate_absolute_pose_error(gt_df: pd.DataFrame, est_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the absolute error between ground truth and estimated poses using only 
    common time indices between the two DataFrames.
    
    Note:
        Both DataFrames must have a 'time' column. The function will only use timestamps
        that exist in both DataFrames. Any timestamps that don't match between the 
        DataFrames will be excluded from the error calculation.
    
    Args:
        gt_df (pd.DataFrame): Ground truth DataFrame with pose data
        est_df (pd.DataFrame): Estimated pose DataFrame
        
    Returns:
        pd.DataFrame: DataFrame containing the absolute errors with columns:
            - time: timestamp
            - x: absolute position error in x
            - y: absolute position error in y
            - z: absolute position error in z
            - total: total position error (Euclidean distance)
            - error_qx: absolute orientation error in qx
            - error_qy: absolute orientation error in qy
            - error_qz: absolute orientation error in qz
            - error_qw: absolute orientation error in qw
    """
    logger.info("Calculating absolute pose errors...")
    logger.info("Analyzing time indices...")
    
    # Find common time indices
    # Clean duplicate timestamps
    logger.info("Cleaning duplicate timestamps in gt_df...")
    gt_filtered = clean_duplicate_timestamps(gt_df).copy()
    logger.info("Cleaning duplicate timestamps in est_df...")
    est_filtered = clean_duplicate_timestamps(est_df).copy()
    gt_times = set(gt_filtered['time'])
    est_times = set(est_filtered['time'])
    common_times = sorted(gt_times.intersection(est_times))
    
    # Report on matching
    gt_unmatched = len(gt_times) - len(common_times)
    est_unmatched = len(est_times) - len(common_times)
    logger.info(f"Ground truth timestamps without matches: {gt_unmatched}")
    logger.info(f"Estimated timestamps without matches: {est_unmatched}")
    logger.info(f"Common timestamps for error calculation: {len(common_times)}")
    
    if len(common_times) == 0:
        raise ValueError("No common timestamps found between ground truth and estimated data!")
    
    # Filter both DataFrames to common times
    gt_filtered = gt_filtered[gt_filtered['time'].isin(common_times)]
    est_filtered = est_filtered[est_filtered['time'].isin(common_times)]
    
    # Initialize error DataFrame
    error_df = pd.DataFrame()
    error_df['time'] = gt_filtered['time']
    
    # Position errors
    for coord in ['x', 'y', 'z']:
        error_df[coord] = abs(gt_filtered[coord] - est_filtered[coord])
    
    # Total position error (Euclidean distance)
    error_df['total'] = np.sqrt(
        error_df['x']**2 + 
        error_df['y']**2 + 
        error_df['z']**2
    )
    
    # Orientation errors
    for coord in ['qx', 'qy', 'qz', 'qw']:
        error_df[f'error_{coord}'] = abs(
            gt_filtered[coord] - est_filtered[coord]
        )
    
    logger.info("Pose error calculation complete.")
    return error_df


def is_pos_vel_measurement(obj):
    return hasattr(obj, "position") and hasattr(obj, "velocity")


def measurements_to_dataframe(
measurements: MeasurementArray,
timestamp: Optional[NDArray[np.float64]] = None,
) -> pd.DataFrame:
    """
    Convert a list of PosMeasurement or PosVelMeasurement objects into a pandas DataFrame.
    
    Args:
        measurements: Array of PosMeasurement or PosVelMeasurement objects
        
    Returns:
        pd.DataFrame: DataFrame containing all measurement data with columns:
            - time: timestamps
            - x, y, z: position components
            - vx, vy, vz: velocity components (if available)
            - covariance: position covariance matrix or full state covariance matrix
    """
    if not measurements.any():
        return pd.DataFrame(columns=['time', 'x', 'y', 'z', 'covariance'])
    
    if timestamp is not None:
        # Check if the provided timestamp matches the measurements
        if len(measurements) != len(timestamp):
            raise ValueError("Length of measurements and timestamp array must match.")
    
    # Check if any measurements have velocity data
    has_velocity = any(is_pos_vel_measurement(m) for m in measurements)
    
    data = {
        'x': [], 'y': [], 'z': [],
        'covariance': []
    }

    if timestamp is not None:
        data['time'] = timestamp.tolist()
    
    # Add velocity columns if needed
    if has_velocity:
        data['vx'] = []
        data['vy'] = []
        data['vz'] = []
    
    for m in measurements:
        # Add timestamp and position
        data['x'].append(float(m.position[0]))
        data['y'].append(float(m.position[1]))
        data['z'].append(float(m.position[2]))
        
        # Add velocity if available
        if is_pos_vel_measurement(m):
            data['vx'].append(float(m.velocity[0]))
            data['vy'].append(float(m.velocity[1]))
            data['vz'].append(float(m.velocity[2]))
        elif has_velocity:
            # Fill with NaN if this measurement doesn't have velocity
            data['vx'].append(np.nan)
            data['vy'].append(np.nan)
            data['vz'].append(np.nan)
        
        # Add covariance matrix
        data['covariance'].append(m.covariance)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df


def track_to_dataframe(track: Track) -> pd.DataFrame:
    """
    Convert a Track object into a pandas DataFrame.
    
    Args:
        track: A Track object containing state estimates
        
    Returns:
        pd.DataFrame: DataFrame containing track data with columns:
            - time: timestamps
            - x, y, z: position components
            - vx, vy, vz: velocity components (if available)
            - covariance: state covariance matrices
            - track_id: identifier for the track
    """
    data = {
        'time': [],
        'x': [], 'y': [], 'z': [],
        'vx': [], 'vy': [], 'vz': [],
        'covariance': [],
        'track_id': track.id
    }
    
    for timestamp, state, cov in zip(
        track.timestamp_history, 
        track.state_history, 
        track.covariance_history
        ):
        data['time'].append(timestamp)
        data['x'].append(float(state[0]))
        data['y'].append(float(state[1]))
        data['z'].append(float(state[2]))
        data['vx'].append(float(state[3]))
        data['vy'].append(float(state[4]))
        data['vz'].append(float(state[5]))
        data['covariance'].append(cov)
    
    df = pd.DataFrame(data)
    return df


def estimate_velocity(
    df: pd.DataFrame,
    window_size: int = 1,
    position_cols: List[str] = ['x', 'y', 'z']
) -> pd.DataFrame:
    """
    Estimate velocity components from position data using central difference method.
    Optionally applies moving average smoothing.

    Args:
        df (pd.DataFrame): DataFrame containing 'time' column and position columns
        window_size (int, optional): Size of moving average window. Default is 1 (no smoothing)
        position_cols (List[str], optional): Names of position columns. Defaults to ['x', 'y', 'z']

    Returns:
        pd.DataFrame: DataFrame with original columns plus velocity columns ('vx', 'vy', 'vz')
        and speed magnitude ('v_total')

    Raises:
        ValueError: If required columns are missing or if DataFrame has less than 2 points
    """
    # Input validation
    required_cols = ['time'] + position_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if len(df) < 2:
        raise ValueError("At least 2 data points are required for velocity estimation")

    # Create copy of input DataFrame
    result_df = df.copy()

    # Initialize velocity columns
    vel_cols = [f'v{col}' for col in position_cols]
    for vel_col in vel_cols:
        result_df[vel_col] = np.nan

    # Calculate velocities using central difference
    time_diff = np.diff(result_df['time'])
    
    for pos_col, vel_col in zip(position_cols, vel_cols):
        # Calculate position differences
        pos_diff = np.diff(result_df[pos_col])
        
        # Central difference for interior points
        velocities = pos_diff / time_diff
        
        # Forward difference for first point
        v_start = (result_df[pos_col].iloc[1] - result_df[pos_col].iloc[0]) / (
            result_df['time'].iloc[1] - result_df['time'].iloc[0])
        
        # Backward difference for last point
        # v_end = (result_df[pos_col].iloc[-1] - result_df[pos_col].iloc[-2]) / (
        #     result_df['time'].iloc[-1] - result_df['time'].iloc[-2])
        
        # Combine all velocities
        all_velocities = np.concatenate([[v_start], velocities])
        
        # Apply moving average if window_size > 1
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            all_velocities = np.convolve(all_velocities, kernel, mode='same')
        
        # Store in DataFrame
        result_df[vel_col] = all_velocities

    # Calculate total speed (magnitude of velocity vector)
    result_df['v_total'] = np.sqrt(
        sum(result_df[vel_col]**2 for vel_col in vel_cols)
    )

    return result_df


def dataframe_to_measurements(
    df: pd.DataFrame, 
    velocity: bool = False,
    position_cols: List[str] = ['x', 'y', 'z'],
    velocity_cols: List[str] = ['vx', 'vy', 'vz']
) -> Tuple[NDArray[np.float64], NDArray[np.object_]]:
    """
    Convert DataFrame rows into a list of PosMeasurement or PosVelMeasurement objects.
    
    Args:
        df (pd.DataFrame): DataFrame containing position and optionally velocity data
        velocity (bool, optional): Whether to include velocity in measurements. Defaults to False.
        position_cols (List[str], optional): Column names for position. Defaults to ['x', 'y', 'z'].
        velocity_cols (List[str], optional): Column names for velocity. Defaults to ['vx', 'vy', 'vz'].
    
    Returns:
        Tuple[NDArray[np.float64], NDArray[np.object_]]: Tuple of arrays containing timestamps and measurement objects.
        
    Raises:
        ValueError: If required columns are missing from the DataFrame
    """
    # Verify required columns exist
    required_cols = ['time'] + position_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check if velocity data is requested and available
    use_velocity = velocity
    if use_velocity:
        # Check if velocity columns exist
        missing_vel_cols = [col for col in velocity_cols if col not in df.columns]
        if missing_vel_cols:
            logger.warning(f"Velocity columns {missing_vel_cols} not found in DataFrame. Falling back to PosMeasurement.")
            use_velocity = False
    
    timestamps = []
    measurements = []
    
    # Process each row
    for _, row in df.iterrows():
        # Extract timestamp and position vector
        timestamps.append(row['time'])
        position = np.array([row[col] for col in position_cols], dtype=float)
        
        # Extract covariance matrix if available, otherwise use default
        if 'covariance' in row:
            covariance = row['covariance']
        else:
            # Create a diagonal covariance matrix with default values
            covariance = np.diag([1.0] * (3 if not use_velocity else 6))
        
        # Create measurement based on type
        if use_velocity:
            # Extract velocity vector
            velocity_vector = np.array([row[col] for col in velocity_cols], dtype=float)
            
            # Check if any velocity values are NaN
            if np.isnan(velocity_vector).any():
                # Create position measurement if velocity contains NaN
                measurements.append(PosMeasurement(
                    position=position,
                    covariance=covariance[:3, :3] if covariance.shape[0] > 3 else covariance
                ))
            else:
                # Create position-velocity measurement
                measurements.append(PosVelMeasurement(
                    position=position,
                    velocity=velocity_vector,
                    covariance=covariance
                ))
        else:
            # Create position measurement
            measurements.append(PosMeasurement(
                position=position,
                covariance=covariance[:3, :3] if covariance.shape[0] > 3 else covariance
            ))
    
    timestamps = np.array(timestamps, dtype=np.float64)
    measurements = np.array(measurements, dtype=object)

    return timestamps, measurements

def analyze_measurement_covariance(
    measurements: MeasurementArray
) -> pd.DataFrame:
    """
    Analyzes the covariance matrices of a list of PosMeasurement objects.
    
    Args:
        measurements (MeasurementArray): Array of PosMeasurement objects
        
    Returns:
        pd.DataFrame: DataFrame containing statistical analysis of measurement covariances:
            - std_x: standard deviation in x dimension
            - std_y: standard deviation in y dimension
            - std_z: standard deviation in z dimension
            - std_total: Euclidean standard deviation
    """
    if not measurements.any():
        return pd.DataFrame(columns=['std_x', 'std_y', 'std_z', 'std_total'])
    
    # Initialize arrays to store standard deviations
    std_x = []
    std_y = []
    std_z = []
    std_total = []
    
    # Process each measurement
    for m in measurements:
        # Extract diagonal elements (variances) from covariance matrix
        variances = np.diag(m.covariance[:3, :3])
        
        # Calculate standard deviations (square root of variances)
        std_x.append(np.sqrt(variances[0]))
        std_y.append(np.sqrt(variances[1]))
        std_z.append(np.sqrt(variances[2]))
        
        # Calculate total standard deviation (Euclidean)
        std_total.append(np.sqrt(np.sum(variances)))
    
    # Create DataFrame with results
    result_df = pd.DataFrame({
        'std_x': std_x,
        'std_y': std_y,
        'std_z': std_z,
        'std_total': std_total
    })
    
    # Calculate summary statistics
    summary = {
        'mean': result_df.mean(),
        'min': result_df.min(),
        'max': result_df.max(),
        'median': result_df.median()
    }
    
    return pd.DataFrame(summary)