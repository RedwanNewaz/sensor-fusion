#pragma once 
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <Eigen/Dense>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <filters.hpp>
#include <definitions.hpp>
#include <process_models.hpp>
#include <fusion.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "../filters/UnscentedKalmanFilter.hpp"

using namespace std;
using namespace Eigen;
using namespace ser94mor::sensor_fusion;
const auto qos = rclcpp::QoS(rclcpp::KeepLast(1), rmw_qos_profile_sensor_data);



namespace airlab{
    inline void printTransformation(const tf2::Transform& transform)
    {
        std::cout << "Translation: (" << transform.getOrigin().x()
                << ", " << transform.getOrigin().y()
                << ", " << transform.getOrigin().z() << ")" << std::endl;
        std::cout << "Rotation: (" << transform.getRotation().x()
                << ", " << transform.getRotation().y()
                << ", " << transform.getRotation().z()
                << ", " << transform.getRotation().w() << ")" << std::endl;
    }


    class apriltag_fusion: public rclcpp::Node
    {
        using Mat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
        using UKF = UnscentedKalmanFilter<CTRV::ProcessModel, Radar::MeasurementModel<CTRV::ProcessModel>>;
        using BEL = Belief<CTRV::StateVector, CTRV::StateCovarianceMatrix>;
        enum COLOR{
            RED,
            GREEN,
            DEFAULT
        };
    public:
        apriltag_fusion(const rclcpp::NodeOptions& options);
    
    protected:
        void ukfInit();
        void rvizTimerCallback();
        void estimatorTimerCallback(); 
        void filterInput(const uint32_t timestamp, const tf2::Transform& tf2Transform);
        Radar::Measurement stateToRadar(const uint32_t timestamp, const tf2::Transform& tf2Transform);
        Lidar::Measurement stateToLidar(const uint32_t timestamp, const tf2::Transform& tf2Transform);
        tf2::Transform beliefToTF(const BEL& belief);
        void pub_robot_state(const tf2::Transform& t);
        void state_callback(const tf2::Transform& tf, const COLOR& color);
        void publish_traj();
        void apriltag_callback(geometry_msgs::msg::PoseArray::SharedPtr msg);
    private:
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
        rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
        rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr pose_sub_;
        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr create3_state_pub_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr robot_state_pub_;
        unordered_map<string, Mat3> cam_info_;
        unordered_map<string, tf2::Transform> to_map_;

        bool filterInit_;
        bool sensorFusion_;
        tf2::Transform robotState_;
        tf2::Transform apriltagMeas_; 
        unique_ptr<UKF_CTRV_LIDAR_RADAR_Fusion> ukf_;
        u_int64_t prev_timestamp_;
        double rho_dot_, yaw_rate_;
        CTRV::ControlVector control_vector_;
        CTRV::ProcessModel pm_;
        Radar::MeasurementModel<CTRV::ProcessModel> radar_mm_;
        rclcpp::TimerBase::SharedPtr timer_, estimator_loop_;
        int tag_id_;

        double tag_size_;

        std::unordered_map<COLOR, std::vector<geometry_msgs::msg::Point>> pubData_;

    };
}