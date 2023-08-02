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
#include <nav_msgs/msg/odometry.hpp>
#include "filters.hpp"
#include "definitions.hpp"
#include "process_models.hpp"
#include <fusion.hpp>

using namespace std;
using namespace Eigen;
using namespace ser94mor::sensor_fusion;
const auto qos = rclcpp::QoS(rclcpp::KeepLast(1), rmw_qos_profile_sensor_data);


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


namespace airlab
{
    class apriltag_fusion: public rclcpp::Node
    {
        using Mat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
        using UKF = UnscentedKalmanFilter<CTRV::ProcessModel, Radar::MeasurementModel<CTRV::ProcessModel>>;
        using BEL = Belief<CTRV::StateVector, CTRV::StateCovarianceMatrix>;
    public:
        apriltag_fusion(const rclcpp::NodeOptions& options): rclcpp::Node("apriltag_fusion", options), filterInit_(false)
        {
            this->declare_parameter("tag_id", 7);
            this->declare_parameter("sensor_fusion", 0);

            ukfInit();

            readTransformation();

            create3_state_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/apriltag/viz", 10);
            tag_id_ = this->get_parameter("tag_id").get_parameter_value().get<int>();
            sensorFusion_ = this->get_parameter("sensor_fusion").get_parameter_value().get<int>();

            robot_state_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("apriltag/state", 10);


            // compute cmd_vel and rho_dot  from ros topics
            control_vector_ = CTRV::ControlVector::Zero();
            rho_dot_ = yaw_rate_ = 0;

            odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("odom", qos, [&](nav_msgs::msg::Odometry::SharedPtr msg)
            {
                auto linear_vel = msg->twist.twist.linear;
                rho_dot_ = sqrt(linear_vel.x * linear_vel.x + linear_vel.y * linear_vel.y);
                yaw_rate_ = msg->twist.twist.angular.z;
            });

            cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>("cmd_vel", qos   , [&](geometry_msgs::msg::Twist::SharedPtr msg){
                auto acc_linear = msg->linear.x;
                auto acc_angular = msg->angular.z;
                control_vector_ << acc_linear, acc_linear, acc_linear, acc_angular, acc_angular;
            });


            logitec_tag_sub_ = this->create_subscription<apriltag_msgs::msg::AprilTagDetectionArray>("/apriltag_logitec/detections", qos, [&]
                    (apriltag_msgs::msg::AprilTagDetectionArray::SharedPtr msg){
                    for(const auto& detection: msg->detections)
                    {
                        if(detection.id == tag_id_)
                            transformToMapCoordinate(msg->header.stamp.nanosec, detection.homography, msg->header.frame_id);
                    }
            });

            nexigo_tag_sub_ = this->create_subscription<apriltag_msgs::msg::AprilTagDetectionArray>("/apriltag_nexigo/detections", qos, [&]
                    (apriltag_msgs::msg::AprilTagDetectionArray::SharedPtr msg){
                    for(const auto& detection: msg->detections)
                    {
                        if(detection.id == tag_id_)
                            transformToMapCoordinate(msg->header.stamp.nanosec, detection.homography, msg->header.frame_id);
                    }
            });

            timer_ = this->create_wall_timer(1s, [this] { publish_traj(); });

            RCLCPP_INFO(get_logger(), "apriltag fusion node started");
        }
    private:
        rclcpp::Subscription<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr logitec_tag_sub_, nexigo_tag_sub_;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
        rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr create3_state_pub_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr robot_state_pub_;
        unordered_map<string, Mat3> cam_info_;
        unordered_map<string, tf2::Transform> to_map_;

        bool filterInit_;
        bool sensorFusion_;
        unique_ptr<UKF_CTRV_LIDAR_RADAR_Fusion> ukf_;
        uint32_t prev_timestamp_;
        double rho_dot_, yaw_rate_;
        CTRV::ControlVector control_vector_;
        CTRV::ProcessModel pm_;
        rclcpp::TimerBase::SharedPtr timer_;
        int tag_id_;

        double tag_size_;
        enum COLOR{
            RED,
            GREEN,
            DEFAULT
        };
        std::unordered_map<COLOR, std::vector<geometry_msgs::msg::Point>> pubData_;

    private:

        void ukfInit()
        {
            this->declare_parameter("ctrv_mtx", vector<double>{0.005,  0.0, 0.0, 0.005});
            this->declare_parameter("lidar_mtx", vector<double>{0.0225,0.0, 0.0, 0.0225});
            this->declare_parameter("radar_mtx", vector<double>{2.050,  0.000,  0.00, 0.000,  2.050,  0.00, 0.000,  0.000,  0.09});

            // retrieve sensor fusion
            vector<double> ctrv_vec, lidar_vec, radar_vec;
            ctrv_vec = this->get_parameter("ctrv_mtx").get_parameter_value().get<vector<double>>();
            lidar_vec = this->get_parameter("lidar_mtx").get_parameter_value().get<vector<double>>();
            radar_vec = this->get_parameter("radar_mtx").get_parameter_value().get<vector<double>>();

            CTRV::ProcessNoiseCovarianceMatrix ctrv_mtx = Eigen::Map<CTRV::ProcessNoiseCovarianceMatrix>(ctrv_vec.data());
            Lidar::MeasurementCovarianceMatrix lidar_mtx = Eigen::Map<Lidar::MeasurementCovarianceMatrix>(lidar_vec.data());
            Radar::MeasurementCovarianceMatrix radar_mtx = Eigen::Map<Radar::MeasurementCovarianceMatrix>(radar_vec.data());

            pm_.SetProcessNoiseCovarianceMatrix(ctrv_mtx);
            ukf_= std::make_unique<UKF_CTRV_LIDAR_RADAR_Fusion>(ctrv_mtx, lidar_mtx, radar_mtx);
        }

        void readTransformation()
        {
            this->declare_parameter("logitec_pmat", vector<double>{933.4418334960938, 0, 978.0901233670083, 0, 0, 995.1202392578125, 490.9420947208673, 0, 0, 0, 1, 0});
            this->declare_parameter("nexigo_pmat", vector<double>{863.1061401367188, 0, 946.3947846149531, 0, 0, 903.219482421875, 411.1189551965581, 0, 0, 0, 1, 0});
            this->declare_parameter("logitec_to_map", vector<double>{-0.232, 1.258, 3.098, 0.996, -0.013, -0.026, 0.073});
            this->declare_parameter("nexigo_to_map", vector<double>{0.259, 1.737, 3.070, -0.014, 0.970, 0.226, 0.080});
            this->declare_parameter("tag_size", 0.2);

            tag_size_ = this->get_parameter("tag_size").get_parameter_value().get<double>();; // 200 mm

            auto toTransform = [](const vector<double>& vecTf)
            {
                tf2::Transform res;
                res.setOrigin(tf2::Vector3(vecTf[0], vecTf[1], vecTf[2]));
                res.setRotation(tf2::Quaternion(vecTf[3], vecTf[4], vecTf[5], vecTf[6]));
                return res;
            };

            vector<double> logitec_pmat, nexigo_pmat, logitec_to_map, nexigo_to_map;

            logitec_pmat = this->get_parameter("logitec_pmat").get_parameter_value().get<vector<double>>();
            nexigo_pmat = this->get_parameter("nexigo_pmat").get_parameter_value().get<vector<double>>();
            logitec_to_map = this->get_parameter("logitec_to_map").get_parameter_value().get<vector<double>>();
            nexigo_to_map = this->get_parameter("nexigo_to_map").get_parameter_value().get<vector<double>>();


            Mat3 Pinv_logi = Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(logitec_pmat.data()).leftCols<3>().inverse();
            Mat3 Pinv_nexigo = Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(nexigo_pmat.data()).leftCols<3>().inverse();

            cam_info_["logitec_cam"] = Pinv_logi;
            cam_info_["nexigo_cam"] = Pinv_nexigo;

            to_map_["logitec_cam"] = toTransform(logitec_to_map);
            to_map_["nexigo_cam"] = toTransform(nexigo_to_map);
        }


        void transformToMapCoordinate(const uint32_t timestamp, const std::array<double, 9>& H, const string& frame_id)
        {

            geometry_msgs::msg::Transform transform;
            getPose(H, cam_info_[frame_id], transform, tag_size_);

            // convert geometry msg to tf2::Transform class
            tf2::Transform tf2Transform;
            tf2::fromMsg(transform, tf2Transform);

            //  M_t_R = M_t_X * X_t_R
            tf2::Vector3 MapOrigin(0, 0, 0);
            tf2Transform = to_map_[frame_id].inverseTimes(tf2Transform);
            auto q = tf2Transform.getRotation();
            tf2::Matrix3x3 mat(q);
            double roll, pitch, yaw;
            mat.getRPY(roll, pitch, yaw);
            q.setRPY(0, 0, yaw + M_PI);
            tf2Transform.setRotation(q);



            if(sensorFusion_)
                filterInput(timestamp, tf2Transform);
            else
            {
                pub_robot_state(tf2Transform);
                state_callback(tf2Transform, DEFAULT);
            }

//            state_callback(tf2Transform, DEFAULT);

        }
        Radar::Measurement stateToRadar(const uint32_t timestamp, const tf2::Transform& tf2Transform)
        {
            auto origin = tf2Transform.getOrigin();
            double rho = sqrt(origin.x() * origin.x() + origin.y() * origin.y());
            double alpha = atan2(origin.y(), origin.x());

            Radar::MeasurementVector mv;
            mv << rho, alpha, rho_dot_;
            auto time = timestamp / 1.0e9;
            Radar::Measurement radar_meas{time, mv};
            return radar_meas;
        }

        Lidar::Measurement stateToLidar(const uint32_t timestamp, const tf2::Transform& tf2Transform)
        {
            auto origin = tf2Transform.getOrigin();
            Lidar::MeasurementVector mv;
            mv << origin.x(), origin.y();
            auto time = timestamp / 1.0e9;
            Lidar::Measurement lidar_meas{time, mv};
            return lidar_meas;
        }


        void filterInput(const uint32_t timestamp, const tf2::Transform& tf2Transform)
        {

            if(!filterInit_)
            {
                prev_timestamp_ = timestamp;
                filterInit_ = true;
                return;
            }

            auto beliefToTf = [&](const BEL& belief)
            {
                const auto& sv{belief.mu()};
                CTRV::ROStateVectorView state_vector_view{sv};

                tf2::Transform state;
                auto origin = tf2::Vector3(state_vector_view.px(), state_vector_view.py(), 0);
                state.setRotation(tf2Transform.getRotation());
                state.setOrigin(origin);
                return state;
            };

            // convert time from nanosecond scale to sec scale
            auto time = timestamp / 1.0e9;
            auto dt = (timestamp - prev_timestamp_) / 1.0e9;

            // initial obsrvation is very noisy use lidar point observation model to filter noise
            // process liadr measurement to make prediction
            Lidar::Measurement lidar_meas = stateToLidar(timestamp, tf2Transform);
            BEL belief_initial{ukf_->ProcessMeasurement(lidar_meas)};
            //use control and process model to make prediction
            auto belief_prior{UKF::Predict(belief_initial, control_vector_ ,   dt, pm_)};

            // fuse apriltag with odom velocity information with the radar model to update state
            Radar::Measurement radar_meas = stateToRadar(timestamp, beliefToTf(belief_prior));
            auto belief{ukf_->ProcessMeasurement(radar_meas)};
            auto state = beliefToTf(belief);
            pub_robot_state(state);
            state_callback(state, DEFAULT);
            prev_timestamp_ = timestamp;
        }

        void pub_robot_state(const tf2::Transform& t)
        {
            nav_msgs::msg::Odometry odom;
            auto pos = t.getOrigin();
            auto ori = t.getRotation();
            odom.header.stamp = get_clock()->now();
            odom.header.frame_id = "map";

            odom.pose.pose.position.x = pos.x();
            odom.pose.pose.position.y = pos.y();
            odom.pose.pose.position.z = pos.z();

            odom.pose.pose.orientation.x = ori.x();
            odom.pose.pose.orientation.y = ori.y();
            odom.pose.pose.orientation.z = ori.z();
            odom.pose.pose.orientation.w = ori.w();

            robot_state_pub_->publish(odom);

        }




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

            // RCLCPP_INFO_STREAM(get_logger(), tt.transpose());

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
            // convert Transform to viz marker
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
            pubData_[color].push_back(marker.pose.position);
        }

        void publish_traj()
        {
            // it is problematic if we continuously publish long trajectory use timer function to periodically publish trajectory
            auto color = DEFAULT;
            if(pubData_[color].empty())
                return;

            visualization_msgs::msg::Marker trajMarker;
            std::copy(pubData_[color].begin(), pubData_[color].end(), std::back_inserter(trajMarker.points));
            trajMarker.id = 202;
            trajMarker.action = visualization_msgs::msg::Marker::ADD;
            trajMarker.header.stamp = get_clock()->now();
            trajMarker.header.frame_id = "map";

            trajMarker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
            trajMarker.ns = get_namespace();
            trajMarker.color.g = 1;
            trajMarker.color.b = 1;
            trajMarker.color.a = 0.7;
            trajMarker.scale.x = trajMarker.scale.y = trajMarker.scale.z = 0.08;
            create3_state_pub_->publish(trajMarker);
        }

    };
}

RCLCPP_COMPONENTS_REGISTER_NODE(airlab::apriltag_fusion)