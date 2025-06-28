# MTT_TestingFramework

A Python-based testing framework for processing ROS bag files and evaluating multi-target tracking algorithms. This testing framework was developed to specifically test the multi-target tracking package included in this repository and is part of the [WildCap](https://www.aamirahmad.de/projects/wildcap/) project.

## Features

- ROS bag file processing and analysis
- Ground truth data extraction and visualization
- Sensor measurement simulation from ground truth
- 2D and 3D trajectory visualization
- Coordinate transformation utilities
- Multi-target tracking algorithm testing environment
- Extended Kalman Filter (EKF) testing
- Single and multi-drone tracking scenarios
- Real-world data analysis capabilities

## Requirements

- Python 3.9 or higher
- numpy (>=1.24.0)
- pandas (>=2.0.0)
- matplotlib (>=3.7.0)
- scipy (>=1.10.0)
- rosbags (>=0.9.15)
- jupyter (==1.0.0)
- notebook (>=7.0.0)
- ipywidgets (>=8.0.0)
- tqdm (>=4.67.1)
- opencv-python (>=4.11.0.86)

### Build Dependencies (for C++ components)
- CMake (>=3.10)
- C++ compiler with C++11 support
- Eigen3 development files
- pybind11 (>=2.10.0)

## Installation

### System Dependencies (Ubuntu)
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libeigen3-dev python3-pybind11
```

### Install Package
```bash
# Clone the repository
git clone <repository-url>
cd <your-repo>
git submodule update --init --recursive

# Install the package
pip install .

# For development
pip install -e .
```

**Note**:  
Ensure that your `pip` is up to date before installing this package.  
Older versions of pip (before 25.1.1) may cause errors such as:  
`pip._vendor.packaging.requirements.InvalidRequirement: Invalid URL given`  
 
To upgrade pip, run:  
```bash
python -m pip install --upgrade pip
```

## Usage

The testing framework is primarily used through Jupyter notebooks located in `examples/notebooks/`. Each notebook demonstrates specific testing scenarios:

| Notebook | Description |
|----------|-------------|
| `01_error_analysis.ipynb` | Error analysis and performance evaluation |
| `02_EKF.ipynb` | Extended Kalman Filter testing and validation |
| `03_modular_STT.ipynb` | Single Target Tracker (STT) component testing |
| `04_JPDA.ipynb` | Joint Probabilistic Data Association (JPDA) tracker evaluation |
| `05_MHT.ipynb` | Multiple Hypothesis Tracker (MHT) testing |
| `06_multi_drone_MTT.ipynb` | Multi-drone multi-target tracking scenarios |
| `07_real_data_MTT.ipynb` | Real-world data analysis and tracking performance |

### Quick Start
```bash
# Start Jupyter
jupyter notebook examples/notebooks/

# Open any notebook and follow the instructions
```

## Data Sources

The simulated rosbag files used for testing were created using the Airship-MPC ROS simulation from [https://github.com/robot-perception-group/Airship-MPC](https://github.com/robot-perception-group/Airship-MPC). These bags contain simulated drone flights with ground truth data and sensor measurements for comprehensive testing of tracking algorithms.

The specific rosbag files used in the notebooks will be available at [this repository](https://keeper.mpdl.mpg.de/d/212a60d7c0694fb4986d/). The following rosbag files are included in this data:

### Simulated Data
- `1_zebra_3_drones_with_gps_drift.bag`: 1 zebra tracked by 3 drones
- `2_zebras_3_drones_gps_drift.bag`: 2 zebras tracked by 3 drones
- `6_zebras_3_drones_gps_drift.bag`: 6 zebras tracked by 3 drones
- `gazebo_flight__2025-02-13-14-57-05.bag`: 1 zebra tracked with 1 drone
- `test_JPDA_node.bag`: Target tracking with JPDA ROS node, 2 zebras, 3 drones

### Real-World Data
This data was recorded during test flights in Kenya in April 2025.
- `0404_1elephants1_2025-04-04_11_10_26_1.bag`: A herd of elephants and a herd of zebras tracked with 1 drone
- `0406_1donkey1_2025-04-06_08_22_54_1.bag`: A herd of donkeys, drone 1
- `0406_1donkey1_2025-04-06_08_23_10_2.bag`: A herd of donkeys, drone 2
- `0411_zebras_2025-04-11_13_50_40_1.bag`: A single zebra with 1 drone

## Project Structure

```
MTT_TestingFramework/
├── data/                       # Sample rosbag files for testing
├── examples/
│   └── notebooks/              # Test notebooks
├── src/
│   ├── multi_target_tracking/      # Multi-target tracking package (see mtt/README.md)
│   └── utils/                      # Testing utilities
│       ├── df_utils.py             # Pandas DataFrame manipulation
│       ├── rosbag_utils.py         # ROS bag processing
│       ├── sensor_sim.py           # Sensor simulation
│       ├── plotting_utils.py       # Visualization tools
│       ├── visualization_utils.py  # High-level visualization functions
│       ├── transform_utils.py      # Coordinate transformations
│       ├── log_cleaner.py          # Log file cleanup
│       └── bag_video_export.py     # Export images/video from rosbags
├── results/                    # Test results output
└── logs/                       # Test execution logs
```

## Multi-Target Tracking

For detailed information about the multi-target tracking algorithms and implementations, please refer to the [MTT package documentation](src/multi_target_tracking/README.md).

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Murty Algorithm**: Integrates [Jonatan Olofsson's C++ implementation](https://github.com/jonatanolofsson/mht) via pybind11
- **EHM Algorithm**: Uses [pyehm](https://github.com/sglvladi/pyehm) by Lyudmil Vladimirov as a dependency
- **Simulation Data**: Generated using [Airship-MPC](https://github.com/robot-perception-group/Airship-MPC) ROS simulation
