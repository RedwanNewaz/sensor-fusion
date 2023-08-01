//
// Created by redwan on 7/31/23.
//
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#include <Eigen/Dense>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/msg/marker.hpp>

using namespace std;
using namespace Eigen;
const auto qos = rclcpp::QoS(rclcpp::KeepLast(1), rmw_qos_profile_sensor_data);
namespace airlab
{
    class apriltag_fusion: public rclcpp::Node
    {
        using Mat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
    public:
        apriltag_fusion(const rclcpp::NodeOptions& options): rclcpp::Node("apriltag_fusion", options)
        {


            create3_state_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/ekf/apriltag/viz", 10);

            vector<double>logitec_pmat{933.4418334960938, 0, 978.0901233670083, 0, 0, 995.1202392578125, 490.9420947208673, 0, 0, 0, 1, 0};
            vector<double> nexigo_pmat{863.1061401367188, 0, 946.3947846149531, 0, 0, 903.219482421875, 411.1189551965581, 0, 0, 0, 1, 0};
            Mat3 Pinv_logi = Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(logitec_pmat.data()).leftCols<3>().inverse();
            Mat3 Pinv_nexigo = Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(nexigo_pmat.data()).leftCols<3>().inverse();
            cam_info_["logitec_cam"] = Pinv_logi;
            cam_info_["nexigo_cam"] = Pinv_nexigo;
            const double tag_size = 0.2; //200 mm
            logitec_tag_sub_ = this->create_subscription<apriltag_msgs::msg::AprilTagDetectionArray>("/apriltag_logitec/detections", qos, [&]
                    (apriltag_msgs::msg::AprilTagDetectionArray::SharedPtr msg){

                for(const auto& detection: msg->detections)
                {
                    geometry_msgs::msg::Transform transform;
                    getPose(detection.homography, cam_info_[msg->header.frame_id], transform, tag_size);
                    // Assuming you have the TransformStamped message.
                    tf2::Transform tf2Transform;
                    tf2::fromMsg(transform, tf2Transform);
                    state_callback(tf2Transform, RED);
                }
            });

            nexigo_tag_sub_ = this->create_subscription<apriltag_msgs::msg::AprilTagDetectionArray>("/apriltag_nexigo/detections", qos, [&]
                    (apriltag_msgs::msg::AprilTagDetectionArray::SharedPtr msg){

                for(const auto& detection: msg->detections)
                {
                    geometry_msgs::msg::Transform transform;
                    getPose(detection.homography, cam_info_[msg->header.frame_id], transform, tag_size);
                    // Assuming you have the TransformStamped message.
                    tf2::Transform tf2Transform;
                    tf2::fromMsg(transform, tf2Transform);
                    state_callback(tf2Transform, GREEN);
                }
            });

            RCLCPP_INFO(get_logger(), "apriltag fusion node started");
        }
    private:
        rclcpp::Subscription<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr logitec_tag_sub_, nexigo_tag_sub_;
        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr create3_state_pub_;
        unordered_map<string, Mat3> cam_info_;
        enum COLOR{
            RED,
            GREEN,
            DEFAULT
        };

    private:
        void getPose(const std::array<double, 9>& H,
                     const Mat3& Pinv,
                     geometry_msgs::msg::Transform& t,
                     const double size)
        {
            // compute extrinsic camera parameter
            // https://dsp.stackexchange.com/a/2737/31703
            // H = K * T  =>  T = K^(-1) * H
            const Mat3 T = Pinv * Eigen::Map<const Mat3>(H.data());
            Mat3 R;
            R.col(0) = T.col(0).normalized();
            R.col(1) = T.col(1).normalized();
            R.col(2) = R.col(0).cross(R.col(1));

            // rotate by half rotation about x-axis to have z-axis
            // point upwards orthogonal to the tag plane
            R.col(1) *= -1;
            R.col(2) *= -1;

            // the corner coordinates of the tag in the canonical frame are (+/-1, +/-1)
            // hence the scale is half of the edge size
            const Eigen::Vector3d tt = T.rightCols<1>() / ((T.col(0).norm() + T.col(0).norm()) / 2.0) * (size / 2.0);

            const Eigen::Quaterniond q(R);

            RCLCPP_INFO_STREAM(get_logger(), tt);

            t.translation.x = tt.x();
            t.translation.y = tt.y();
            t.translation.z = tt.z();
            t.rotation.w = q.w();
            t.rotation.x = q.x();
            t.rotation.y = q.y();
            t.rotation.z = q.z();
        }

        void state_callback(const tf2::Transform& tf, const COLOR& color)
        {
            // convert odom to viz marker
            visualization_msgs::msg::Marker marker;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.header.stamp  = get_clock()->now();
            marker.header.frame_id  = "map";
            marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
            marker.mesh_resource = "package://irobot_create_description/meshes/body_visual.dae";
            marker.id = static_cast<int>(color);
            marker.ns = get_namespace();
            marker.scale.x = marker.scale.y = marker.scale.z = 1.0; // arrow scale 0.2 roomba scale 1.0
            switch (color) {
                case RED: marker.color.r = 1; break;
                case GREEN: marker.color.g = 1; break;
                default:
                    marker.color.r = marker.color.g  = marker.color.b = 0.66;

            }
            marker.color.a = 0.85;
            tf2::toMsg(tf, marker.pose);
            create3_state_pub_->publish(marker);


        }

    };
}

RCLCPP_COMPONENTS_REGISTER_NODE(airlab::apriltag_fusion)