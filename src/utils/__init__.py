"""
Initialization file for the utils package.
Exposes common utility functions for easy import while keeping implementation details private.
"""

# Import from df_utils
from .df_utils import (
    sample_data,
    clean_duplicate_timestamps,
    reindex_dataframe,
    calculate_position_error,
    calculate_position_error_statistics,
    calculate_absolute_pose_error,
    measurements_to_dataframe,
    track_to_dataframe,
    estimate_velocity,
    dataframe_to_measurements,
    analyze_measurement_covariance,
)

# Import from plotting_utils
# Note: check_data_synchronization and confidence_ellipse are kept private as they're internal helper functions
from .plotting_utils import (
    plot_xyz_over_time,
    plot_xyz_3d,
    plot_orientation_over_time,
    create_covariance_ellipse,
    animate_2d_trajectory,
    animate_2d_targets,
)

# Import from rosbag_utils
from .rosbag_utils import (
    load_rosbag,
    create_df_from_bag,
    import_rosbag_data,
    display_rosbag_info,
    import_detections_array_data,
    import_target_array_data,
)

# Import from sensor_sim
from .sensor_sim import (
    simulate_position_sensor,
    create_zebra_db,
    simulate_zebra_identification,
)

# Import from transform_utils
from .transform_utils import rotation_to_quaternion_covariance

# Import from log_cleaner
from .log_cleaner import cleanup_log_files

# Import from visualization_utils
from .visualization_utils import (
    visualize_tracking_data,
    create_video_from_frames,
)

__all__ = [
    # df_utils
    'sample_data', 'clean_duplicate_timestamps', 'reindex_dataframe',
    'calculate_position_error', 'calculate_position_error_statistics',
    'calculate_absolute_pose_error', 'measurements_to_dataframe', 
    'track_to_dataframe', 'estimate_velocity', 'dataframe_to_measurements',
    'analyze_measurement_covariance',
    
    # plotting_utils
    'plot_xyz_over_time', 'plot_xyz_3d', 'plot_orientation_over_time', 
    'create_covariance_ellipse', 'animate_2d_trajectory',
    
    # rosbag_utils
    'load_rosbag', 'create_df_from_bag', 'import_rosbag_data', 
    'display_rosbag_info', 'import_detections_array_data',
    'import_target_array_data',

    # sensor_sim
    'simulate_position_sensor',
    'create_zebra_db', 
    'simulate_zebra_identification',
    
    # transform_utils
    'rotation_to_quaternion_covariance',
    
    # log_cleaner
    'cleanup_log_files',

    # visualization_utils
    'visualize_tracking_data',
    'create_video_from_frames',
]
