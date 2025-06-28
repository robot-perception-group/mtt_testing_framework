"""
This module provides plotting utilities for visualizing rosbag imported data using matplotlib.
"""

# Standard library imports
import sys
import logging
from typing import Optional, Tuple, Dict, List
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from scipy import stats

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from .transform_utils import rotation_to_quaternion_covariance

# Configure module logger
logger = logging.getLogger(__name__)

def plot_xyz_over_time(df: pd.DataFrame, 
                       fig: Optional[plt.Figure] = None, 
                       label: Optional[str] = None, 
                       noisy_limits: bool = False,
                       show_uncertainty: bool = False, 
                       coords: List[str] = ['x', 'y', 'z'],
                       cov_indices: Dict = {'x': 0, 'y': 1, 'z': 2}, 
                       color: Optional[str] = None, 
                       alpha: float = 1.0,
                       linestyle: str = "-",
                       linewidth: float = 1.0,
                       marker: Optional[str] = None
                       ) -> plt.Figure:
    """
    Plots position coordinates over time from the provided DataFrame.

    Creates one subplot for each coordinate in the coords list (defaults to x, y, and z).
    If a matplotlib Figure is provided, the data is added to it;
    otherwise, a new figure with appropriate subplots is created.

    Parameters:
        df (pd.DataFrame): DataFrame containing at least the columns 'time' and position coordinates
                          specified in the coords parameter.
                          If show_uncertainty=True, also needs 'covariance' column containing
                          covariance matrices.
        fig (Optional[plt.Figure]): An optional matplotlib Figure to plot on.
        label (Optional[str]): Label for the plotted lines. If provided, a shared legend will be shown.
        noisy_limits (bool): If True, sets axis limits based on 95% percentile range.
                           If limits exist, they will only be expanded, not shrunk.
        show_uncertainty (bool): If True, plots uncertainty bands using covariance data.
        coords (List[str]): List of coordinate column names to plot (default: ['x', 'y', 'z']).
        cov_indices (Dict): Dictionary mapping coordinate names to their indices in the covariance matrix.
                           Only used when show_uncertainty=True. Default: {'x': 0, 'y': 1, 'z': 2}.
        color (Optional[str]): Color for the plotted lines. If None, uses matplotlib's default color cycle.
        alpha (float): Transparency of the plotted lines and uncertainty bands. Range: 0.0 (transparent) to 1.0 (opaque).
        linestyle (str): Line style for the plotted lines. Default: "-". Options: "-", "--", "-.", ":", etc.
        linewidth (float): Width of the plotted lines. Default: 1.0.
        marker (Optional[str]): Marker style for the plotted points. Default: None. Options: "o", "s", "^", "v", etc.

    Returns:
        plt.Figure: The matplotlib Figure object containing the plot.
    """
    n_coords = len(coords)
    
    # Create a new figure with appropriate number of subplots if none is provided.
    if fig is None:
        fig, axs = plt.subplots(nrows=n_coords, ncols=1, sharex=True, figsize=(10, 2.5*n_coords))
        
        # Handle case where only one subplot is created (axs won't be a list)
        if n_coords == 1:
            axs = [axs]
        
        # Define a uniform style for all plots.
        for i, (ax, coord) in enumerate(zip(axs, coords)):
            ax.set_ylabel(f"{coord} [m]")
            ax.grid(True)
            
            # Only add x-label to the bottom subplot
            if i == n_coords - 1:
                ax.set_xlabel("Time (s)")
        
        plt.tight_layout()
    else:
        axs = fig.get_axes()

    # Define a uniform style for all plots.
    time_array = df["time"].to_numpy()

    # Calculate new axis limits based on percentile range
    new_limits = {}
    if noisy_limits:
        # Handle axis limits for each subplot
        for ax, coord in zip(axs, coords):
            coord_min, coord_max = np.percentile(df[coord], [1, 95])
            current_limits = ax.get_ylim()
            
            # If limits exist, only expand them
            if ax.get_ylim() != (0.0, 1.0):  # Default matplotlib limits
                new_min = min(current_limits[0], coord_min-5)
                new_max = max(current_limits[1], coord_max+5)
            else:
                new_min, new_max = coord_min, coord_max
            new_limits[coord] = (new_min, new_max)

    # Plot the data for each coordinate
    for ax, coord in zip(axs, coords):
        coord_data = df[coord].to_numpy()
        line = ax.plot(time_array, coord_data, label=label, linestyle=linestyle, 
                      color=color, alpha=alpha, linewidth=linewidth, marker=marker)[0]
        
        if show_uncertainty:
            # Get the standard deviation from the corresponding diagonal element of the covariance matrix
            cov_idx = cov_indices[coord]
            std_dev = np.array([cov[cov_idx, cov_idx] for cov in df['covariance']])
            std_dev = np.sqrt(std_dev)
            
            # Create uncertainty bands (±2σ for 95% confidence interval)
            upper_bound = coord_data + 2 * std_dev
            lower_bound = coord_data - 2 * std_dev

            # Convert any numpy.float64 objects to regular floats if needed
            if isinstance(lower_bound[0], np.float64) or isinstance(upper_bound[0], np.float64):
                lower_bound = lower_bound.astype(float)
                upper_bound = upper_bound.astype(float)
            
            ax.fill_between(time_array, lower_bound, upper_bound, 
                          color=line.get_color(), alpha=alpha * 0.5)  # Make uncertainty bands more transparent

    # Set axis limits
    if new_limits:
        for ax, coord in zip(axs, coords):
            ax.set_ylim(new_limits[coord])

    return fig


def plot_xyz_3d(df: pd.DataFrame, fig: Optional[plt.Figure] = None, label: Optional[str] = None) -> plt.Figure:
    """
    Plots xyz data as a 3D line plot. Optionally, a second line can be added.

    Parameters:
        df (pd.DataFrame): DataFrame containing the columns 'x', 'y', 'z' for the primary line.
        fig (Optional[plt.Figure]): An optional matplotlib Figure to plot on.
        label (Optional[str]): Label for the plotted line. If provided, a legend will be shown.
            
    Returns:
        plt.Figure: The matplotlib Figure object containing the plot.
    """
    if fig is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.grid(True)
    else:
        ax = fig.get_axes()[0]
    
    linestyle = "-"
    # Plot the primary 3D line
    ax.plot(df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy(), label=label, linestyle=linestyle)
    
    if label:
        ax.legend()
    
    return fig

def plot_orientation_over_time(df: pd.DataFrame, fig: Optional[plt.Figure] = None,
                             label: Optional[str] = None, noisy_limits: bool = False,
                             show_uncertainty: bool = False, use_euler: bool = False) -> plt.Figure:
    """
    Plots orientation data over time from the provided DataFrame.

    Creates one subplot for each orientation component. By default, plots quaternion
    components (w,x,y,z). If use_euler=True, plots Euler angles (roll,pitch,yaw).

    Parameters:
        df (pd.DataFrame): DataFrame containing orientation data as defined in rosbag_utils.py.
                         Quaternion columns: 'qx', 'qy', 'qz', 'qw'
                         Covariance: 'covariance_21' to 'covariance_35' for orientation
        fig (Optional[plt.Figure]): An optional matplotlib Figure to plot on.
        label (Optional[str]): Label for the plotted lines. If provided, a shared legend will be shown.
        noisy_limits (bool): If True, sets axis limits based on 95% percentile range.
        show_uncertainty (bool): If True, plots uncertainty bands using covariance data.
        use_euler (bool): If True, converts quaternions to Euler angles (roll,pitch,yaw).

    Returns:
        plt.Figure: The matplotlib Figure object containing the plot.
    """
    # Process orientation data
    if use_euler:
        # Convert quaternions to euler angles
        quats = df[['qx', 'qy', 'qz', 'qw']].to_numpy()  # Extract all quaternions
        rot = Rotation.from_quat(quats)
        euler = rot.as_euler('xyz', degrees=False)
        
        coords = ['roll', 'pitch', 'yaw']
        ylabels = ['Roll [rad]', 'Pitch [rad]', 'Yaw [rad]']
        titles = ['Roll Angle Over Time', 'Pitch Angle Over Time', 'Yaw Angle Over Time']
        
        # Create temporary DataFrame with euler angles
        plot_data = pd.DataFrame({
            'time': df['time'],
            'roll': euler[:, 0],
            'pitch': euler[:, 1],
            'yaw': euler[:, 2]
        })

        if show_uncertainty:
            # Extract standard deviations directly from the rotation part of the covariance matrix
            std_devs = {
                'roll': np.sqrt(np.array([cov[3, 3] for cov in df['covariance']])),   # roll variance
                'pitch': np.sqrt(np.array([cov[4, 4] for cov in df['covariance']])),  # pitch variance
                'yaw': np.sqrt(np.array([cov[5, 5] for cov in df['covariance']]))     # yaw variance
            }
    else:
        coords = ['qx', 'qy', 'qz', 'qw']
        ylabels = ['x', 'y', 'z', 'w']
        titles = ['Quaternion X Component', 'Quaternion Y Component',
                 'Quaternion Z Component', 'Quaternion W Component']
        plot_data = df
        
        if show_uncertainty:
            # Calculate quaternion covariance for each timestep
            std_devs = {coord: np.zeros(len(df)) for coord in coords}
            
            for i in range(len(df)):
                # Extract current quaternion
                current_quat = np.array([
                    df['qx'].iloc[i], df['qy'].iloc[i],
                    df['qz'].iloc[i], df['qw'].iloc[i]
                ])
                
                # Extract rotation covariance at this timestep
                rot_cov = np.zeros((3, 3))
                # Extract the rotation elements from the matrix covariance
                rot_cov[0, 0] = df['covariance'].iloc[i][3, 3]  # roll-roll
                rot_cov[0, 1] = df['covariance'].iloc[i][3, 4]  # roll-pitch
                rot_cov[0, 2] = df['covariance'].iloc[i][3, 5]  # roll-yaw
                rot_cov[1, 0] = df['covariance'].iloc[i][4, 3]  # pitch-roll
                rot_cov[1, 1] = df['covariance'].iloc[i][4, 4]  # pitch-pitch
                rot_cov[1, 2] = df['covariance'].iloc[i][4, 5]  # pitch-yaw
                rot_cov[2, 0] = df['covariance'].iloc[i][5, 3]  # yaw-roll
                rot_cov[2, 1] = df['covariance'].iloc[i][5, 4]  # yaw-pitch
                rot_cov[2, 2] = df['covariance'].iloc[i][5, 5]  # yaw-yaw
                
                # Transform covariance
                quat_cov = rotation_to_quaternion_covariance(rot_cov, current_quat)
                
                # Store standard deviations
                std_devs['qx'][i] = np.sqrt(abs(quat_cov[0, 0]))
                std_devs['qy'][i] = np.sqrt(abs(quat_cov[1, 1]))
                std_devs['qz'][i] = np.sqrt(abs(quat_cov[2, 2]))
                std_devs['qw'][i] = np.sqrt(abs(quat_cov[3, 3]))

    n_plots = len(coords)

    # Create a new figure with subplots if none is provided
    if fig is None:
        fig, axs = plt.subplots(nrows=n_plots, ncols=1, sharex=True, figsize=(10, 2.5*n_plots))
        # Define a uniform style for all plots
        for ax, ylabel, title in zip(axs, ylabels, titles):
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True)
        axs[-1].set_xlabel("Time (s)")
        plt.tight_layout()
    else:
        axs = fig.get_axes()

    # Define a uniform style for all plots
    linestyle = "-"
    time_array = df["time"].to_numpy()

    # Calculate new axis limits based on percentile range
    new_limits = {}
    if noisy_limits:
        for ax, coord in zip(axs, coords):
            # Use plot_data instead of df for accessing coordinates
            coord_min, coord_max = np.percentile(plot_data[coord], [1, 99])
            current_limits = ax.get_ylim()
            
            if ax.get_ylim() != (0.0, 1.0):
                new_min = min(current_limits[0], coord_min)
                new_max = max(current_limits[1], coord_max)
            else:
                new_min, new_max = coord_min, coord_max
            new_limits[coord] = (new_min, new_max)

    # Plot the data for each coordinate
    for idx, (ax, coord) in enumerate(zip(axs, coords)):
        coord_data = plot_data[coord].to_numpy()
        line = ax.plot(time_array, coord_data, label=label, linestyle=linestyle, 
                      color=color, alpha=alpha, linewidth=linewidth, marker=marker)[0]
        
        if show_uncertainty and coord in std_devs:
            std_dev = std_devs[coord]
            upper_bound = coord_data + 2 * std_dev
            lower_bound = coord_data - 2 * std_dev
            ax.fill_between(time_array, lower_bound, upper_bound,
                          color=line.get_color(), alpha=0.2)

    # Set axis limits
    if new_limits:
        for ax, coord in zip(axs, coords):
            ax.set_ylim(new_limits[coord])

    if label:
        axs[0].legend()

    return fig

def check_data_synchronization(gt_times: np.ndarray, meas_times_list: List[np.ndarray], 
                               tolerance: float = 0.01) -> bool:
    """
    Check if ground truth and measurement timestamps are synchronized.
    
    Args:
        gt_times: Array of ground truth timestamps
        meas_times_list: List of arrays containing measurement timestamps
        tolerance: Maximum allowed time difference between timestamps
        
    Returns:
        True if synchronized, raises ValueError otherwise
    """
    for i, meas_times in enumerate(meas_times_list):
        gt_start, gt_end = gt_times[0], gt_times[-1]
        meas_start, meas_end = meas_times[0], meas_times[-1]
        
        # Check if measurement time range is contained within ground truth time range
        if meas_start < gt_start - tolerance:
            logger.warning(f"Measurement set {i+1} starts at {meas_start:.3f}s, which is before ground truth start at {gt_start:.3f}s")
        
        if meas_end > gt_end + tolerance:
            logger.warning(f"Measurement set {i+1} ends at {meas_end:.3f}s, which is after ground truth end at {gt_end:.3f}s")
        
        # Check for time gaps
        max_diff = np.max(np.abs(gt_times - meas_times))
        
        if abs(max_diff) > tolerance:
            logger.warning(f"Timestamps not synchronized for measurement set {i+1}. "
                           f"Maximum time difference: {max_diff:.4f}s (tolerance: {tolerance:.4f}s)")
    
    return True

def confidence_ellipse(cov: np.ndarray, center: Tuple[float, float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate parameters for drawing an ellipse representing a covariance matrix at a specific confidence level.
    
    Args:
        cov: 2x2 covariance matrix
        center: (x, y) center coordinates of ellipse
        confidence: Confidence level (0-1)
        
    Returns:
        width, height, angle_degrees tuple representing ellipse parameters
    """
    # Extract the 2D position covariance (assuming cov is at least 2x2)
    cov_2d = cov[:2, :2]
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
    
    # Sort by eigenvalue in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Chi-square value for desired confidence level with 2 degrees of freedom
    chisquare_val = stats.chi2.ppf(confidence, 2)
    
    # Calculate width and height of ellipse
    width = 2 * np.sqrt(chisquare_val * eigenvalues[0])
    height = 2 * np.sqrt(chisquare_val * eigenvalues[1])
    
    # Calculate rotation angle in degrees
    angle_radians = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    angle_degrees = np.degrees(angle_radians)
    
    return width, height, angle_degrees

def create_covariance_ellipse(x: float, y: float, covariance: np.ndarray, 
                              confidence: float = 0.95, alpha: float = 0.3, 
                              color: str = 'blue') -> Ellipse:
    """
    Create an ellipse artist for visualizing a 2D covariance.
    
    Args:
        x, y: Center coordinates
        covariance: Covariance matrix (at least 2x2)
        confidence: Confidence interval (0-1)
        alpha: Ellipse transparency
        color: Ellipse color
        
    Returns:
        matplotlib Ellipse patch
    """
    width, height, angle = confidence_ellipse(covariance, (x, y), confidence)
    ellipse = Ellipse((x, y), width=width, height=height, angle=angle,
                      facecolor=color, alpha=alpha, edgecolor=color, linewidth=1)
    return ellipse

def animate_2d_trajectory(
    gt_df: pd.DataFrame,
    meas1_df: pd.DataFrame,
    meas2_df: Optional[pd.DataFrame] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    labels: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
    show_covariance: bool = True,
    noisy_limits: bool = False,
    history_length: int = 100,
    interval: int = 50,
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> animation.FuncAnimation:
    """
    Create an animation showing ground truth and measurement data over time using DataFrames.
    
    Parameters:
        gt_df: DataFrame with ground truth data containing 'time', 'x', 'y' (and optionally 'z') columns
        meas1_df: DataFrame with first measurement set containing 'time', 'x', 'y' columns and optionally 'covariance'
        meas2_df: Optional DataFrame with second measurement set (same format as meas1_df)
        fig, ax: Optional figure and axes to plot on
        labels: Dict with keys 'gt', 'meas1', 'meas2' for custom labels
        colors: Dict with keys 'gt', 'meas1', 'meas2' for custom colors
        show_covariance: Whether to show covariance ellipses if covariance data is available
        history_length: Number of past points to display in trajectory
        interval: Animation interval in milliseconds
        save_path: If provided, save animation to this path
        title: Optional custom title for the plot
        
    Returns:
        matplotlib.animation.FuncAnimation object
    """
    # Default values for labels and colors
    if labels is None:
        labels = ['Ground Truth', 'Measurement 1', 'Measurement 2']
    
    if colors is None:
        colors = ['black', 'blue', 'red']
    
    # Extract timestamps
    gt_times = gt_df['time'].to_numpy()
    meas1_times = meas1_df['time'].to_numpy()
    
    # Check data synchronization
    meas_times_list = [meas1_times]
    
    if meas2_df is not None:
        meas2_times = meas2_df['time'].to_numpy()
        meas_times_list.append(meas2_times)
    
    try:
        check_data_synchronization(gt_times, meas_times_list)
    except ValueError as e:
        logger.warning(f"Data synchronization issue: {e}")
        logger.warning("Continuing with animation despite synchronization issues.")
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract position data
    gt_pos_2d = gt_df[['x', 'y']].to_numpy()
    meas1_pos_2d = meas1_df[['x', 'y']].to_numpy()
    
    if meas2_df is not None:
        meas2_pos_2d = meas2_df[['x', 'y']].to_numpy()
    
    # Set initial plot limits with some margin
    all_data_x = np.concatenate([gt_pos_2d[:, 0], meas1_pos_2d[:, 0]] + 
                              ([meas2_pos_2d[:, 0]] if meas2_df is not None else []))
    all_data_y = np.concatenate([gt_pos_2d[:, 1], meas1_pos_2d[:, 1]] + 
                              ([meas2_pos_2d[:, 1]] if meas2_df is not None else []))
    
    if noisy_limits:
        x_min, x_max = np.percentile(all_data_x, [1, 99])
        y_min, y_max = np.percentile(all_data_y, [1, 99])
    else:
        x_min, x_max = np.min(all_data_x), np.max(all_data_x)
        y_min, y_max = np.min(all_data_y), np.max(all_data_y)
    
    # Add 10% padding
    x_padding = 0.1 * (x_max - x_min)
    y_padding = 0.1 * (y_max - y_min)
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Initialize plot elements
    gt_line, = ax.plot([], [], '-', color=colors[0], label=labels[0], linewidth=2)
    gt_point, = ax.plot([], [], 'o', color=colors[0], markersize=8)
    
    meas1_line, = ax.plot([], [], '-', color=colors[1], label=labels[1], alpha=0.7)
    meas1_point, = ax.plot([], [], 'o', color=colors[1], markersize=6)
    
    # Create fixed ellipse artists that we'll update instead of recreating
    meas1_ellipse = Ellipse((0, 0), 0, 0, color=colors[1], alpha=0.3)
    meas1_ellipse_artist = ax.add_patch(meas1_ellipse)
    
    plot_elements = {
        'gt_line': gt_line,
        'gt_point': gt_point,
        'meas1_line': meas1_line,
        'meas1_point': meas1_point,
        'meas1_ellipse': meas1_ellipse_artist
    }
    
    if meas2_df is not None:
        meas2_line, = ax.plot([], [], '-', color=colors[2], label=labels[2], alpha=0.7)
        meas2_point, = ax.plot([], [], 'o', color=colors[2], markersize=6)
        
        # Create fixed ellipse artist for measurement 2
        meas2_ellipse = Ellipse((0, 0), 0, 0, color=colors[2], alpha=0.3)
        meas2_ellipse_artist = ax.add_patch(meas2_ellipse)
        
        plot_elements.update({
            'meas2_line': meas2_line,
            'meas2_point': meas2_point,
            'meas2_ellipse': meas2_ellipse_artist
        })
    
    # Add time text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    plot_elements['time_text'] = time_text
    
    # Add legend
    ax.legend(loc='upper right')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('2D Trajectory' if title is None else title)
    ax.grid(True)
    
    # Check if covariance data is available
    meas1_has_cov = 'covariance' in meas1_df.columns
    meas2_has_cov = meas2_df is not None and 'covariance' in meas2_df.columns
    
    if show_covariance and not meas1_has_cov:
        logger.warning("Covariance visualization requested but 'covariance' column not found in meas1_df")
    if show_covariance and meas2_df is not None and not meas2_has_cov:
        logger.warning("Covariance visualization requested but 'covariance' column not found in meas2_df")
    
    # Update function for animation
    def update(frame):
        if frame >= len(gt_times):
            return list(plot_elements.values())
        
        # Determine history window
        start_idx = max(0, frame - history_length)
        end_idx = frame + 1
        
        # Update ground truth line
        x_hist = gt_pos_2d[start_idx:end_idx, 0]
        y_hist = gt_pos_2d[start_idx:end_idx, 1]
        gt_line.set_data(x_hist, y_hist)
        
        # Update ground truth point
        if frame < len(gt_pos_2d):
            gt_point.set_data([gt_pos_2d[frame, 0]], [gt_pos_2d[frame, 1]])
        
        # Update measurement 1 line
        if start_idx < len(meas1_pos_2d) and end_idx <= len(meas1_pos_2d):
            x_hist_meas1 = meas1_pos_2d[start_idx:end_idx, 0]
            y_hist_meas1 = meas1_pos_2d[start_idx:end_idx, 1]
            meas1_line.set_data(x_hist_meas1, y_hist_meas1)
        
        # Update measurement 1 point
        if frame < len(meas1_pos_2d):
            meas1_point.set_data([meas1_pos_2d[frame, 0]], [meas1_pos_2d[frame, 1]])
            
            # Update covariance ellipse for measurement 1
            if show_covariance and meas1_has_cov and frame < len(meas1_df):
                # Calculate ellipse parameters
                x = meas1_pos_2d[frame, 0]
                y = meas1_pos_2d[frame, 1]
                covariance = meas1_df['covariance'].iloc[frame]
                width, height, angle = confidence_ellipse(covariance, (x, y))
                
                # Update existing ellipse properties instead of creating a new one
                meas1_ellipse.set_center((x, y))
                meas1_ellipse.width = width
                meas1_ellipse.height = height
                meas1_ellipse.angle = angle
                meas1_ellipse.set_visible(True)
            else:
                meas1_ellipse.set_visible(False)
        
        # Update measurement 2 if available
        if meas2_df is not None and frame < len(meas2_pos_2d):
            # Update measurement 2 line
            if start_idx < len(meas2_pos_2d) and end_idx <= len(meas2_pos_2d):
                x_hist_meas2 = meas2_pos_2d[start_idx:end_idx, 0]
                y_hist_meas2 = meas2_pos_2d[start_idx:end_idx, 1]
                meas2_line.set_data(x_hist_meas2, y_hist_meas2)
            
            # Update measurement 2 point
            meas2_point.set_data([meas2_pos_2d[frame, 0]], [meas2_pos_2d[frame, 1]])
            
            # Update covariance ellipse for measurement 2
            if show_covariance and meas2_has_cov and frame < len(meas2_df):
                # Calculate ellipse parameters
                x = meas2_pos_2d[frame, 0]
                y = meas2_pos_2d[frame, 1]
                covariance = meas2_df['covariance'].iloc[frame]
                width, height, angle = confidence_ellipse(covariance, (x, y))
                
                # Update existing ellipse
                meas2_ellipse.set_center((x, y))
                meas2_ellipse.width = width
                meas2_ellipse.height = height
                meas2_ellipse.angle = angle
                meas2_ellipse.set_visible(True)
            else:
                meas2_ellipse.set_visible(False)
        
        # Update time text
        time_text.set_text(f'Time: {gt_times[frame]:.2f}s')
        
        # Return all updated elements
        return list(plot_elements.values())
    
    # Create animation
    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=len(gt_times),
        interval=interval, blit=True
    )
    
    # Save animation if path provided
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=1000/interval)
        print(f"Animation saved successfully.")
    
    return anim

def animate_2d_targets(
    target_dfs: List[pd.DataFrame],
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    show_covariance: bool = True,
    noisy_limits: bool = False,
    history_length: int = 100,
    interval: int = 50,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    time_step: float = 0.05
) -> animation.FuncAnimation:
    """
    Create an animation showing multiple target trajectories over time with synchronized timing.
    
    Parameters:
        target_dfs: List of DataFrames, each containing 'time', 'x', 'y' columns and optionally 'covariance'
        fig, ax: Optional figure and axes to plot on
        labels: List of custom labels for each target (default: Target 1, Target 2, ...)
        colors: List of custom colors for each target
        show_covariance: Whether to show covariance ellipses if covariance data is available
        noisy_limits: If True, sets axis limits based on 95% percentile range
        history_length: Number of past points to display in trajectory
        interval: Animation interval in milliseconds
        save_path: If provided, save animation to this path
        title: Optional custom title for the plot
        time_step: Time step for animation frames in seconds (default: 0.05)
        
    Returns:
        matplotlib.animation.FuncAnimation object
    """
    if not target_dfs:
        raise ValueError("At least one target DataFrame must be provided")
    
    n_targets = len(target_dfs)
    
    # Default values for labels and colors
    if labels is None:
        labels = [f'Target {i+1}' for i in range(n_targets)]
    elif len(labels) != n_targets:
        raise ValueError(f"Number of labels ({len(labels)}) must match number of target DataFrames ({n_targets})")
    
    if colors is None:
        # Use matplotlib's default color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        # Extend colors if we have more targets than default colors
        while len(colors) < n_targets:
            colors.extend(colors)
        colors = colors[:n_targets]
    elif len(colors) != n_targets:
        raise ValueError(f"Number of colors ({len(colors)}) must match number of target DataFrames ({n_targets})")
    
    # Find the overall time range
    all_start_times = [df['time'].min() for df in target_dfs]
    all_end_times = [df['time'].max() for df in target_dfs]
    
    global_start_time = min(all_start_times)
    global_end_time = max(all_end_times)
    
    # Create time array for animation frames
    animation_times = np.arange(global_start_time, global_end_time + time_step, time_step)
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate plot limits
    all_x_data = np.concatenate([df['x'].to_numpy() for df in target_dfs])
    all_y_data = np.concatenate([df['y'].to_numpy() for df in target_dfs])
    
    if noisy_limits:
        x_min, x_max = np.percentile(all_x_data, [1, 99])
        y_min, y_max = np.percentile(all_y_data, [1, 99])
    else:
        x_min, x_max = np.min(all_x_data), np.max(all_x_data)
        y_min, y_max = np.min(all_y_data), np.max(all_y_data)
    
    # Add 10% padding
    x_padding = 0.1 * (x_max - x_min)
    y_padding = 0.1 * (y_max - y_min)
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Initialize plot elements for each target
    plot_elements = {}
    target_lines = []
    target_points = []
    target_ellipses = []
    
    for i in range(n_targets):
        # Create line and point for each target
        line, = ax.plot([], [], '-', color=colors[i], label=labels[i], alpha=0.8)
        point, = ax.plot([], [], 'o', color=colors[i], markersize=6)
        
        # Create ellipse for covariance if needed
        ellipse = Ellipse((0, 0), 0, 0, color=colors[i], alpha=0.3)
        ellipse_artist = ax.add_patch(ellipse)
        ellipse.set_visible(False)  # Initially hidden
        
        target_lines.append(line)
        target_points.append(point)
        target_ellipses.append(ellipse_artist)
        
        plot_elements[f'target_{i}_line'] = line
        plot_elements[f'target_{i}_point'] = point
        plot_elements[f'target_{i}_ellipse'] = ellipse_artist
    
    # Add time text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    plot_elements['time_text'] = time_text
    
    # Add legend and labels
    ax.legend(loc='upper right')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Multi-Target 2D Trajectories' if title is None else title)
    ax.grid(True)
    
    # Check covariance availability for each target
    target_has_cov = ['covariance' in df.columns for df in target_dfs]
    
    if show_covariance:
        for i, has_cov in enumerate(target_has_cov):
            if not has_cov:
                logger.warning(f"Covariance visualization requested but 'covariance' column not found in target_dfs[{i}]")
    
    # Precompute indices for efficient lookup during animation
    target_indices = []
    for df in target_dfs:
        times = df['time'].to_numpy()
        indices = []
        for anim_time in animation_times:
            # Find the closest timestamp that is <= anim_time
            valid_indices = np.where(times <= anim_time + time_step/2)[0]  # Small tolerance
            if len(valid_indices) > 0:
                indices.append(valid_indices[-1])  # Take the latest valid index
            else:
                indices.append(-1)  # No valid data yet
        target_indices.append(indices)
    
    def update(frame):
        if frame >= len(animation_times):
            return list(plot_elements.values())
        
        current_time = animation_times[frame]
        
        for target_idx in range(n_targets):
            df = target_dfs[target_idx]
            line = target_lines[target_idx]
            point = target_points[target_idx]
            ellipse = target_ellipses[target_idx]
            
            # Get the precomputed index for this frame and target
            data_idx = target_indices[target_idx][frame]
            
            # Check if target is active (has data and hasn't exceeded max time)
            target_start = df['time'].min()
            target_end = df['time'].max()
            
            if current_time < target_start or data_idx == -1:
                # Target hasn't started yet - hide everything
                line.set_data([], [])
                point.set_data([], [])
                ellipse.set_visible(False)
                continue
            elif current_time > target_end:
                # Target has ended - hide everything
                line.set_data([], [])
                point.set_data([], [])
                ellipse.set_visible(False)
                continue
            
            # Target is active - show data
            # Determine history window
            start_frame = max(0, frame - history_length)
            
            # Collect valid positions for the history window
            x_hist = []
            y_hist = []
            
            for hist_frame in range(start_frame, frame + 1):
                if hist_frame < len(animation_times):
                    hist_idx = target_indices[target_idx][hist_frame]
                    if hist_idx >= 0 and animation_times[hist_frame] <= target_end:
                        x_hist.append(df['x'].iloc[hist_idx])
                        y_hist.append(df['y'].iloc[hist_idx])
            
            # Update line with history
            if x_hist and y_hist:
                line.set_data(x_hist, y_hist)
            else:
                line.set_data([], [])
            
            # Update current point
            current_x = df['x'].iloc[data_idx]
            current_y = df['y'].iloc[data_idx]
            point.set_data([current_x], [current_y])
            
            # Update covariance ellipse
            if show_covariance and target_has_cov[target_idx]:
                covariance = df['covariance'].iloc[data_idx]
                try:
                    width, height, angle = confidence_ellipse(covariance, (current_x, current_y))
                    ellipse.set_center((current_x, current_y))
                    ellipse.width = width
                    ellipse.height = height
                    ellipse.angle = angle
                    ellipse.set_visible(True)
                except:
                    ellipse.set_visible(False)
            else:
                ellipse.set_visible(False)
        
        # Update time text
        time_text.set_text(f'Time: {current_time:.2f}s')
        
        return list(plot_elements.values())
    
    # Create animation
    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=len(animation_times),
        interval=interval, blit=True, repeat=True
    )
    
    # Save animation if path provided
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=1000/interval)
        print(f"Animation saved successfully.")
    
    return anim
