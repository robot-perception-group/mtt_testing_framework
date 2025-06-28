"""
Utilities for generating visualizations of drone and zebra tracking data from ROS bags.

This module provides functions to:
- Visualize drone and zebra position/orientation data
- Create animated visualizations of tracking data
- Generate video files from visualization frames
"""

# Standard library imports
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import cv2
from tqdm import tqdm

# Configure module logger
logger = logging.getLogger(__name__)


def estimate_heading_from_positions(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """
    Estimates heading based on position changes over time and smooths it.
    
    Args:
        df: DataFrame containing 'time', 'x', 'y', 'z' columns
        window_size: Size of the window for smoothing
        
    Returns:
        DataFrame with additional columns 'vx', 'vy', 'vz', 'heading'
    """
    if len(df) < 2:
        raise ValueError("At least 2 data points are required for heading estimation")
        
    # Create a copy of the input DataFrame
    result_df = df.copy()
    
    # Calculate velocities
    time_diff = np.diff(result_df['time'])
    pos_diff_x = np.diff(result_df['x'])
    pos_diff_y = np.diff(result_df['y']) 
    pos_diff_z = np.diff(result_df['z'])
    
    # Calculate velocities using central difference
    vx = pos_diff_x / time_diff
    vy = pos_diff_y / time_diff
    vz = pos_diff_z / time_diff
    
    # Forward difference for first point
    v_start_x = vx[0]
    v_start_y = vy[0]
    v_start_z = vz[0]
    
    # Combine all velocities
    all_vx = np.concatenate([[v_start_x], vx])
    all_vy = np.concatenate([[v_start_y], vy])
    all_vz = np.concatenate([[v_start_z], vz])
    
    # Apply moving average for smoothing
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        all_vx = np.convolve(all_vx, kernel, mode='same')
        all_vy = np.convolve(all_vy, kernel, mode='same')
        all_vz = np.convolve(all_vz, kernel, mode='same')
    
    # Store in DataFrame
    result_df['vx'] = all_vx
    result_df['vy'] = all_vy
    result_df['vz'] = all_vz
    
    # Calculate heading (angle in xy-plane)
    result_df['heading'] = np.arctan2(all_vy, all_vx)
    
    return result_df


def find_common_timerange(gt_df: pd.DataFrame, drone_dfs: List[pd.DataFrame]) -> Tuple[float, float]:
    """
    Find the common time range across all dataframes.
    
    Args:
        gt_df: Ground truth dataframe with zebra positions
        drone_dfs: List of drone dataframes
        
    Returns:
        Tuple of (start_time, end_time)
    """
    # Get time limits for ground truth
    gt_start, gt_end = gt_df['time'].min(), gt_df['time'].max()
    
    # Get time limits for all drone dataframes
    drone_starts = [df['time'].min() for df in drone_dfs]
    drone_ends = [df['time'].max() for df in drone_dfs]
    
    # Find the common range
    start_time = max([gt_start] + drone_starts)
    end_time = min([gt_end] + drone_ends)
    
    if start_time >= end_time:
        raise ValueError("No common time range found between ground truth and drone data")
    
    logger.info(f"Common time range: {start_time:.2f}s to {end_time:.2f}s")
    return start_time, end_time


def interpolate_dataframe(df: pd.DataFrame, target_times: np.ndarray) -> pd.DataFrame:
    """
    Interpolate dataframe values to align with target timestamps.
    
    Args:
        df: Source dataframe with 'time' column
        target_times: Array of target timestamps
        
    Returns:
        New dataframe with interpolated values
    """
    # Create interpolation functions for each numerical column
    interp_funcs = {}
    for col in df.columns:
        if col == 'time' or col == 'frame_id' or col == 'covariance':
            continue  # Skip non-numerical or special columns
        
        # Use linear interpolation for position and velocity data
        interp_funcs[col] = interp1d(
            df['time'].values,
            df[col].values,
            kind='linear', 
            bounds_error=False,
            fill_value=(df[col].iloc[0], df[col].iloc[-1])
        )
    
    # Create new dataframe with interpolated values
    result = {'time': target_times}
    
    for col, func in interp_funcs.items():
        result[col] = func(target_times)
    
    # Handle covariance matrix interpolation separately if present
    if 'covariance' in df.columns:
        # Get list of all covariance matrices
        cov_matrices = np.array(df['covariance'].tolist())
        
        # Create interpolation function for each element of the matrix
        cov_shape = cov_matrices[0].shape
        cov_interps = np.zeros((cov_shape[0], cov_shape[1], len(target_times)))
        
        # Reshape matrices for interpolation
        for i in range(cov_shape[0]):
            for j in range(cov_shape[1]):
                # Extract element (i,j) from each matrix
                elements = cov_matrices[:, i, j]
                # Create interpolation function for this element
                interp_func = interp1d(
                    df['time'].values,
                    elements,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(elements[0], elements[-1])
                )
                # Store interpolated values
                cov_interps[i, j] = interp_func(target_times)
        
        # Create new covariance matrices
        result['covariance'] = [
            np.array([[cov_interps[i, j, k] for j in range(cov_shape[1])] 
                     for i in range(cov_shape[0])])
            for k in range(len(target_times))
        ]
    
    # For frame_id, use the nearest neighbor
    if 'frame_id' in df.columns:
        # Find closest time index
        nearest_indices = np.array([
            np.abs(df['time'].values - t).argmin() 
            for t in target_times
        ])
        result['frame_id'] = df['frame_id'].iloc[nearest_indices].values
    
    return pd.DataFrame(result)


def create_visualization_frame(
    ax: Axes,
    gt_data: pd.DataFrame,
    drone_data: List[pd.DataFrame],
    timestamp: float,
    zebra_colors: Optional[Dict[int, str]] = None,
    drone_colors: Optional[List[str]] = None,
    show_history: bool = True,
    history_seconds: float = 2.0
) -> None:
    """
    Create a single visualization frame at the specified timestamp.
    
    Args:
        ax: Matplotlib axes for plotting
        gt_data: Ground truth data with zebra positions
        drone_data: List of dataframes with drone data
        timestamp: Current timestamp for the frame
        zebra_colors: Dictionary mapping zebra IDs to colors
        drone_colors: List of colors for drones
        show_history: Whether to show trajectory history
        history_seconds: How many seconds of history to show
    """
    # Clear previous frame
    ax.clear()
    
    # Filter data for the current timestamp
    time_filter = gt_data['time'] == timestamp
    current_gt = gt_data[time_filter]
    
    # Set default colors if not provided
    if not zebra_colors:
        zebra_colors = {i: plt.cm.tab10(i % 10) for i in range(len(current_gt))}
    
    if not drone_colors:
        drone_colors = ['red', 'blue', 'green', 'purple', 'orange'][:len(drone_data)]
    
    # Plot ground truth zebra positions
    for i, (_, zebra) in enumerate(current_gt.iterrows()):
        # Plot zebra position
        ax.scatter(zebra['x'], zebra['y'], color=zebra_colors[i], s=100, label=f"Zebra {i+1}")
        
        # Plot zebra heading arrow
        if 'heading' in zebra:
            arrow_length = 1.0  # meters
            dx = arrow_length * np.cos(zebra['heading'])
            dy = arrow_length * np.sin(zebra['heading'])
            ax.arrow(
                zebra['x'], zebra['y'], dx, dy,
                head_width=0.3, head_length=0.3,
                fc=zebra_colors[i], ec=zebra_colors[i], alpha=0.7
            )
        
        # Plot history if enabled
        if show_history:
            history_start = timestamp - history_seconds
            history_mask = (gt_data['time'] >= history_start) & (gt_data['time'] <= timestamp)
            history_data = gt_data[history_mask]
            
            if not history_data.empty:
                ax.plot(
                    history_data['x'], history_data['y'],
                    color=zebra_colors[i], alpha=0.5, linewidth=1.5, linestyle=':'
                )
    
    # Plot drone positions and orientations
    for i, drone_df in enumerate(drone_data):
        time_filter = drone_df['time'] == timestamp
        current_drone = drone_df[time_filter]
        
        if current_drone.empty:
            continue
        
        drone = current_drone.iloc[0]
        color = drone_colors[i % len(drone_colors)]
        
        # Plot drone position
        ax.scatter(drone['x'], drone['y'], color=color, s=120, marker='s', label=f"Drone {i+1}")
        
        # Plot drone orientation
        if all(q in drone for q in ['qx', 'qy', 'qz', 'qw']):
            # Create rotation matrix from quaternion
            quat = np.array([drone['qx'], drone['qy'], drone['qz'], drone['qw']])
            rotation = Rotation.from_quat(quat)
            
            # Get orientation vector (assuming drone's forward direction is along x-axis)
            orientation = rotation.apply([1, 0, 0])
            
            # Scale the orientation arrow
            arrow_length = 2.0  # meters
            dx, dy = orientation[0] * arrow_length, orientation[1] * arrow_length
            
            # Draw orientation arrow
            ax.arrow(
                drone['x'], drone['y'], dx, dy,
                head_width=0.5, head_length=0.5,
                fc=color, ec=color, alpha=0.8
            )
        
        # Plot history if enabled
        if show_history:
            history_start = timestamp - history_seconds
            history_mask = (drone_df['time'] >= history_start) & (drone_df['time'] <= timestamp)
            history_data = drone_df[history_mask]
            
            if not history_data.empty:
                ax.plot(
                    history_data['x'], history_data['y'],
                    color=color, alpha=0.5, linewidth=1.5, linestyle='--'
                )
    
    # Configure axes
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f'Time: {timestamp:.2f}s')
    ax.grid(True)
    ax.axis('equal')
    
    # Add legend (only for unique entries)
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) 
              if l not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper right')


def generate_frames(
    gt_df: pd.DataFrame,
    drone_dfs: List[pd.DataFrame],
    output_dir: str,
    fps: int = 24,
    zebra_colors: Optional[Dict[int, str]] = None,
    drone_colors: Optional[List[str]] = None,
    show_history: bool = True,
    history_seconds: float = 2.0,
    dpi: int = 100,
    figsize: Tuple[int, int] = (12, 10)
) -> List[str]:
    """
    Generate visualization frames and save them as images.
    
    Args:
        gt_df: Ground truth data with zebra positions
        drone_dfs: List of dataframes with drone data
        output_dir: Directory to save output frames
        fps: Frames per second
        zebra_colors: Dictionary mapping zebra IDs to colors
        drone_colors: List of colors for drones
        show_history: Whether to show trajectory history
        history_seconds: How many seconds of history to show
        dpi: DPI for saved images
        figsize: Figure size (width, height) in inches
        
    Returns:
        List of paths to generated frame images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find common time range
    start_time, end_time = find_common_timerange(gt_df, drone_dfs)
    
    # Calculate timestamps for frames
    duration = end_time - start_time
    num_frames = int(duration * fps)
    timestamps = np.linspace(start_time, end_time, num_frames)
    
    # Estimate zebra heading if not present
    if 'heading' not in gt_df.columns:
        gt_df = estimate_heading_from_positions(gt_df)
    
    # Create interpolated dataframes
    interp_gt = interpolate_dataframe(gt_df, timestamps)
    interp_drones = [interpolate_dataframe(df, timestamps) for df in drone_dfs]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate frames
    frame_paths = []
    
    for i, timestamp in enumerate(tqdm(timestamps, desc="Generating frames")):
        # Create the visualization
        create_visualization_frame(
            ax, interp_gt, interp_drones, timestamp,
            zebra_colors, drone_colors, show_history, history_seconds
        )
        
        # Save frame
        frame_path = os.path.join(output_dir, f'frame_{i:04d}.png')
        fig.savefig(frame_path, dpi=dpi, bbox_inches='tight')
        frame_paths.append(frame_path)
    
    plt.close(fig)
    return frame_paths


def create_video_from_frames(
    frame_paths: List[str],
    output_path: str,
    fps: int = 24,
    codec: str = 'mp4v'
) -> str:
    """
    Create a video from a sequence of image frames.
    
    Args:
        frame_paths: List of paths to frame images
        output_path: Path to save the output video
        fps: Frames per second
        codec: FourCC code for the video codec
        
    Returns:
        Path to the created video file
    """
    if not frame_paths:
        raise ValueError("No frames provided")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get frame dimensions from the first image
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Could not read the first frame: {frame_paths[0]}")
        
    target_height, target_width, _ = first_frame.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    # Process all frames
    size_mismatch_detected = False
    
    for path in tqdm(frame_paths, desc="Creating video"):
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f"Could not read frame: {path}")
            
        height, width, _ = frame.shape
        
        # Check if resize is needed
        if height != target_height or width != target_width:
            if not size_mismatch_detected:
                logger.debug(f"Size mismatch detected. Resizing frames to {target_width}x{target_height}")
                size_mismatch_detected = True
                
            frame = cv2.resize(frame, (target_width, target_height))
                
        video.write(frame)
    
    # Release video writer
    video.release()
    logger.info(f"Video saved to {output_path}")
    
    return output_path


def visualize_tracking_data(
    gt_zebra_data: pd.DataFrame,
    drone_data_list: List[pd.DataFrame],
    output_dir: str,
    create_video: bool = True,
    fps: int = 24,
    zebra_colors: Optional[Dict[int, str]] = None,
    drone_colors: Optional[List[str]] = None,
    show_history: bool = True,
    history_seconds: float = 2.0,
    dpi: int = 100,
    figsize: Tuple[int, int] = (12, 10)
) -> Optional[str]:
    """
    Visualize zebra and drone tracking data and optionally create a video.
    
    Args:
        gt_zebra_data: DataFrame with ground truth zebra positions
        drone_data_list: List of DataFrames with drone positions from uav_msgs/msg/uav_pose
        output_dir: Directory to save output frames and video
        create_video: Whether to create a video from the frames
        fps: Frames per second
        zebra_colors: Dictionary mapping zebra IDs to colors
        drone_colors: List of colors for drones
        show_history: Whether to show trajectory history
        history_seconds: How many seconds of history to show
        dpi: DPI for saved images
        figsize: Figure size (width, height) in inches
        
    Returns:
        Path to created video if create_video=True, otherwise None
    """
    # Validate inputs
    if gt_zebra_data.empty:
        raise ValueError("Ground truth data is empty")
    
    if not drone_data_list:
        raise ValueError("No drone data provided")
    
    # Create output directory
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Generate frames
    logger.info("Generating visualization frames...")
    frame_paths = generate_frames(
        gt_zebra_data, drone_data_list, frames_dir,
        fps, zebra_colors, drone_colors,
        show_history, history_seconds,
        dpi, figsize
    )
    
    # Create video if requested
    if create_video and frame_paths:
        logger.info("Creating video from frames...")
        video_path = os.path.join(output_dir, 'tracking_visualization.mp4')
        return create_video_from_frames(frame_paths, video_path, fps)
    
    return None