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
def generate_launch_description():
    ld = LaunchDescription(ARGUMENTS)
    namespace = LaunchConfiguration('namespace')
    sensor_fusion_dir = get_package_share_directory('sensor_fusion')
    robotName = 'ac32'
    if len(sys.argv) > 4:
        robotName = sys.argv[4].split("=")[-1]

    # get path to params file
    params_path = os.path.join(
        sensor_fusion_dir,
        'config',
        '{}_params.yaml'.format(robotName)
    )

    print(params_path)
    ld.add_action(Node(
        package='sensor_fusion', executable='sensor_fusion_node', output='screen',
        name="sensor_fusion_node",
        namespace=namespace,
        parameters=[params_path]
    ))

    return ld