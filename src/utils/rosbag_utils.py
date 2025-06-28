"""
This file defines functions for importing data from a rosbag.
"""

# Standard library imports
import logging
from pathlib import Path
from typing import Any, Dict, List, Union
import re  # For regex pattern matching

# Third-party imports
import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader

# Local application imports
from multi_target_tracking import PosMeasurement

# Configure module logger
logger = logging.getLogger(__name__)

# Define the columns for the pose with covariance stamped message
cols_pose_with_cov_stamped = [
    "time",
    "frame_id",
    "x", "y", "z",
    "qx", "qy", "qz", "qw",
    "covariance"  # Changed to store a single covariance matrix instead of 36 individual elements
]

# Define the columns for the uav pose message
cols_uav_pose = [
    "time",
    "frame_id",
    "x", "y", "z",
    "vx", "vy", "vz",
    "qx", "qy", "qz", "qw",
    "covariance"
]

def load_rosbag(file_path: str) -> AnyReader:
    """
    Load a rosbag file using AnyReader.

    Args:
        file_path (str): Path to the rosbag file

    Returns:
        AnyReader: Reader object for the rosbag file
    """
    path = Path(file_path)
    reader = AnyReader([path])
    logger.debug(f"Loaded rosbag file: {path}")
    return reader

def create_df_from_bag(reader: AnyReader, topic: str) -> pd.DataFrame:
    """
    Convert messages from a topic into a pandas DataFrame.

    Parameters:
        reader (AnyReader): The ROS bag reader from which to read the messages.
        topic (str): The topic to filter the messages.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted data.
    
    Note:
        Supports messages of type PoseWithCovarianceStamped and uav_msgs/msg/uav_pose.
        Returns an empty DataFrame with an error message if the message type is not supported.

        The returned DataFrame will contain the following columns depending on the message type:
        - time: The time of the message.
        - frame_id: The frame ID of the message.
        - x, y, z: The position coordinates.
        - vx, vy, vz: The velocity coordinates (only for uav_pose).
        - qx, qy, qz, qw: The orientation coordinates.
        - covariance: A 6x6 numpy matrix representing the covariance.
    """
    with reader:
        connections = [x for x in reader.connections if x.topic == topic]
        if not connections:
            return pd.DataFrame()
        
        # Get the message type
        msg_type = connections[0].msgtype

        logger.info(f"Processing topic: '{topic}' with message type: '{msg_type}'")
        
        # Initialize data dictionary based on message type
        if "uav_msgs/msg/uav_pose" in msg_type:
            columns = cols_uav_pose
        elif "geometry_msgs/msg/PoseWithCovarianceStamped" in msg_type:
            columns = cols_pose_with_cov_stamped
        else:
            logger.error(f"Unsupported message type: {msg_type}")
            return pd.DataFrame()
            
        data: Dict[str, List[Any]] = {name: [] for name in columns}
            
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            # Process based on message type
            if "uav_msgs/msg/uav_pose" in connection.msgtype:
                # Position
                data['x'].append(msg.position.x)
                data['y'].append(msg.position.y)
                data['z'].append(msg.position.z)
                
                # Velocity
                data['vx'].append(msg.velocity.x)
                data['vy'].append(msg.velocity.y)
                data['vz'].append(msg.velocity.z)
                
                # Orientation
                data['qx'].append(msg.orientation.x)
                data['qy'].append(msg.orientation.y)
                data['qz'].append(msg.orientation.z)
                data['qw'].append(msg.orientation.w)
                
                # Covariance - store as 6x6 matrix if available
                cov_matrix = np.array(msg.covariance).reshape(10, 10)[:6, :6]  # Take the top-left 6x6 portion
                data['covariance'].append(cov_matrix)
                
                # Timestamp and frame_id
                data['time'].append(timestamp / 1e9)  # Convert nanoseconds to seconds
                data['frame_id'].append(msg.header.frame_id)
                
            elif "geometry_msgs/msg/PoseWithCovarianceStamped" in connection.msgtype:
                # Position
                data['x'].append(msg.pose.pose.position.x)
                data['y'].append(msg.pose.pose.position.y)
                data['z'].append(msg.pose.pose.position.z)
                
                # Orientation
                data['qx'].append(msg.pose.pose.orientation.x)
                data['qy'].append(msg.pose.pose.orientation.y)
                data['qz'].append(msg.pose.pose.orientation.z)
                data['qw'].append(msg.pose.pose.orientation.w)
                
                # Covariance - store as 6x6 matrix
                cov_matrix = np.array(msg.pose.covariance).reshape(6, 6)
                data['covariance'].append(cov_matrix)
                
                # Timestamp and frame_id
                data['time'].append(timestamp / 1e9)  # Convert nanoseconds to seconds
                data['frame_id'].append(msg.header.frame_id)
            else:
                logger.error(f"Encountered unsupported message type during processing: {connection.msgtype}")
                return pd.DataFrame()
    
    return pd.DataFrame(data)

def import_rosbag_data(file_path: str, topic: str) -> pd.DataFrame:
    """
    Imports data from a rosbag file for a specific topic and returns a DataFrame.

    Args:
        file_path (str): Path to the rosbag file.
        topic (str): The topic to extract data from.

    Returns:
        pd.DataFrame: The processed data extracted from the rosbag.
    """
    reader = load_rosbag(file_path)
    try:
        df = create_df_from_bag(reader, topic=topic)
    finally:
        reader.close()
    return df

def display_rosbag_info(reader: AnyReader) -> None:
    """
    Displays topics with their message types and message counts.
    
    Args:
        reader (AnyReader): The ROS bag reader
    """
    with reader:
        topics = reader.topics
        connections = reader.connections
        
        # Count messages per topic
        topic_counts = {}
        for connection in connections:
            if connection.topic in topic_counts:
                topic_counts[connection.topic] += connection.msgcount
            else:
                topic_counts[connection.topic] = connection.msgcount
        
        print("#################### Topics and message counts: ####################")
        for topic, msg_type in topics.items():
            print(f"Topic: {topic}")
            print(f"    Message type: {msg_type}")
            print(f"    Message count: {topic_counts.get(topic, 0)}")
        print("#####################################################################\n")

def import_detections_array_data(reader: AnyReader, topics: Union[str, List[str]]) -> List[tuple]:
    """
    Imports array of projected detections from a rosbag file for one or multiple topics.
    
    Args:
        reader (AnyReader): The ROS bag reader from which to read the messages.
        topics (Union[str, List[str]]): The topic or list of topics to extract data from.
            All topics must contain messages of type geometry_msgs/PoseWithCovarianceStamped[].

    Returns:
        List[tuple]: A list of tuples (timestamp, measurements_array, drone_id), where:
            - timestamp is the message timestamp in nanoseconds
            - measurements_array is a numpy array of PosMeasurement objects
            - drone_id is an integer extracted from topic name if it matches /machine_i/*
              or None if no ID could be extracted
        The list is sorted by message timestamp.
    """
    logger.info(f"Importing detections array data from topics: {topics}")
    # Normalize topics to a list
    if isinstance(topics, str):
        topics_list = [topics]
    else:
        topics_list = topics
    
    result_list = []
    
    with reader:
        # Get all relevant connections
        connections = []
        for topic in topics_list:
            topic_connections = [x for x in reader.connections if x.topic == topic]
            if not topic_connections:
                logger.warning(f"No messages found for topic: {topic}")
                continue
            connections.extend(topic_connections)
        
        if not connections:
            logger.error(f"No valid connections found for specified topics")
            return []
        
        # Process all messages
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            # Extract drone_id from topic name if it matches /machine_i/*
            drone_id = None
            topic_name = connection.topic
            match = re.match(r'/machine_(\d+)/', topic_name)
            if match:
                drone_id = int(match.group(1))
            
            # Create a list of PosMeasurement objects for each pose in the array
            measurements = []
            for pose_stamped in msg.detections:
                # Extract position
                position = np.array([
                    pose_stamped.pose.pose.position.x,
                    pose_stamped.pose.pose.position.y,
                    pose_stamped.pose.pose.position.z
                ])
                
                # Extract covariance as 6x6 matrix, then get the 3x3 position part
                full_covariance = np.array(pose_stamped.pose.covariance).reshape(6, 6)
                position_covariance = full_covariance[:3, :3]
                
                # Create a PosMeasurement object
                measurement = PosMeasurement(
                    position=position,
                    covariance=position_covariance
                )
                measurements.append(measurement)
            
            # Only add non-empty arrays to the result
            if measurements:
                # Create a numpy array from the measurements
                measurements_array = np.array(measurements)
                
                # Add a tuple of (timestamp, measurements_array, drone_id) to sort later
                result_list.append((timestamp / 1e9, measurements_array, drone_id))
    
    # Sort by timestamp and extract only the measurement arrays and drone_ids
    result_list.sort(key=lambda x: x[0])
    return result_list

def import_target_array_data(reader: AnyReader, topic: str) -> List[pd.DataFrame]:
    """
    Imports TargetArray messages from a rosbag and converts them into DataFrames for each track.
    
    Args:
        reader (AnyReader): The ROS bag reader from which to read the messages.
        topic (str): The topic containing TargetArray messages.
    
    Returns:
        List[pd.DataFrame]: A list of DataFrames, one for each unique track_id found.
        Each DataFrame has the same structure as track_to_dataframe output with columns:
            - time: timestamps
            - x, y, z: position components
            - vx, vy, vz: velocity components
            - covariance: state covariance matrices (6x6)
            - track_id: identifier for the track
    """
    logger.info(f"Importing TargetArray data from topic: {topic}")
    
    # Dictionary to store data for each track_id
    tracks_data = {}
    
    with reader:
        # Get connections for the specified topic
        connections = [x for x in reader.connections if x.topic == topic]
        if not connections:
            logger.warning(f"No messages found for topic: {topic}")
            return []
        
        # Verify message type
        msg_type = connections[0].msgtype
        if "TargetArray" not in msg_type:
            logger.error(f"Expected TargetArray message type, got: {msg_type}")
            return []
        
        # Process all messages
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            # Convert timestamp from nanoseconds to seconds
            time_seconds = timestamp / 1e9
            
            # Process each target in the array
            for target in msg.targets:
                track_id = target.track_id
                
                # Initialize track data if this is the first time we see this track_id
                if track_id not in tracks_data:
                    tracks_data[track_id] = {
                        'time': [],
                        'x': [], 'y': [], 'z': [],
                        'vx': [], 'vy': [], 'vz': [],
                        'covariance': [],
                        'track_id': track_id
                    }
                
                # Extract position
                tracks_data[track_id]['time'].append(time_seconds)
                tracks_data[track_id]['x'].append(float(target.position.x))
                tracks_data[track_id]['y'].append(float(target.position.y))
                tracks_data[track_id]['z'].append(float(target.position.z))
                
                # Extract velocity
                tracks_data[track_id]['vx'].append(float(target.velocity.x))
                tracks_data[track_id]['vy'].append(float(target.velocity.y))
                tracks_data[track_id]['vz'].append(float(target.velocity.z))
                
                # Extract covariance matrix (reshape from 36-element array to 6x6 matrix)
                covariance_matrix = np.array(target.covariance).reshape(6, 6)
                tracks_data[track_id]['covariance'].append(covariance_matrix)
    
    # Convert each track's data to a DataFrame
    result_dataframes = []
    for track_id, data in tracks_data.items():
        df = pd.DataFrame(data)
        # Sort by time to ensure chronological order
        result_dataframes.append(df)
        logger.debug(f"Created DataFrame for track_id {track_id} with {len(df)} entries")
    
    logger.info(f"Successfully imported {len(result_dataframes)} tracks from TargetArray messages")
    return result_dataframes