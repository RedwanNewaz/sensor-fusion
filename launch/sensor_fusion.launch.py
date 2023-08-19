import argparse
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

import os
import sys

from ament_index_python.packages import get_package_share_directory

ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
]

params = {
    "ctrv_mtx": [0.9725,  0.0, 0.0, 0.9725], #[0.725,  0.0, 0.0, 0.725],
    "lidar_mtx": [0.0275,0.0, 0.0, 0.0275], 
    "radar_mtx": [0.9725,  0.000,  0.00, 
                  0.000,  0.9725,  0.00, 
                  0.000,  0.000,  0.025]
}

def generate_launch_description():
    ld = LaunchDescription(ARGUMENTS)
    namespace = LaunchConfiguration('namespace')
    sensor_fusion_dir = get_package_share_directory('sensor_fusion')
    robotName = 'ac32'
    if len(sys.argv) > 4:
        robotName = sys.argv[4].split("=")[-1]

    ld.add_action(Node(
        package='multicam_tag_state_estimator', executable='multicam_tag_state_estimator_node', 
        name="multicam_tag_state_estimator"
    ))
    ld.add_action(Node(
        package='sensor_fusion', executable='sensor_fusion_node', output='screen',
        name="sensor_fusion_node",
        namespace=namespace, 
        parameters=[params]
    ))

    return ld