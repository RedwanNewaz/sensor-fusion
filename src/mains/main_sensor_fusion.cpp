//
// Created by redwan on 7/31/23.
//
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <Eigen/Dense>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/msg/marker.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <filters.hpp>
#include <definitions.hpp>
#include <process_models.hpp>
#include <fusion.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

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
            this->declare_parameter("tag_id", 32);
            this->declare_parameter("sensor_fusion", 0);

            ukfInit();

            create3_state_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/apriltag/viz/filtered", 10);
            tag_id_ = this->get_parameter("tag_id").get_parameter_value().get<int>();
            sensorFusion_ = this->get_parameter("sensor_fusion").get_parameter_value().get<int>();

            robot_state_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("apriltag/state/filtered", 10);


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
                if(acc_linear != 0)
                    control_vector_ << acc_linear, acc_linear, acc_linear, acc_angular, acc_angular;
            });

            pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>("/apriltag/pose_detections", qos, [&](geometry_msgs::msg::PoseArray::SharedPtr msg){
                auto frame_id = msg->header.frame_id;  
                stringstream ss(frame_id);
                vector<string> ids;
                while (ss.good()) {
                    string substr;
                    getline(ss, substr, ',');
                    ids.push_back(substr);
                }

                for(int i = 0; i < ids.size() - 1; ++i)
                {
                    auto pose = msg->poses.at(i);
                    auto _id = atoi(ids[i].c_str());
                    if(tag_id_ == _id)
                    {
                        // tf2::Transform tf; 
                        tf2::fromMsg(pose, robotState_);
                        if (!filterInit_)
                        {
                            filterInit_ = true; 
                        }
                        prev_timestamp_ = this->get_clock()->now().nanoseconds();

                    }
                }
    

            });


            timer_ = this->create_wall_timer(250ms, [this] { 
                if(filterInit_)
                {
                    timerCallback();
                }                
            });
            estimator_loop_ = this->create_wall_timer(1ms, [this] {
                auto currentTime = this->get_clock()->now().nanoseconds();
                if (filterInit_)
                {
                    filterInput(currentTime, robotState_);
                }
            });
            RCLCPP_INFO(get_logger(), "apriltag fusion node started");
        }
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
        unique_ptr<UKF_CTRV_LIDAR_RADAR_Fusion> ukf_;
        uint32_t prev_timestamp_;
        double rho_dot_, yaw_rate_;
        CTRV::ControlVector control_vector_;
        CTRV::ProcessModel pm_;
        rclcpp::TimerBase::SharedPtr timer_, estimator_loop_;
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

        tf2::Transform beliefToTF(const BEL& belief)
        {
            const auto& sv{belief.mu()};
            CTRV::ROStateVectorView state_vector_view{sv};

            tf2::Transform state;
            auto origin = tf2::Vector3(state_vector_view.px(), state_vector_view.py(), 0);
            state.setRotation(robotState_.getRotation());
            state.setOrigin(origin);
            return state;
        }

        void timerCallback()
        {
            auto belief = ukf_->GetBelief();
            auto state = beliefToTF(belief);
            
            state_callback(state, DEFAULT);
            publish_traj();
        }


        void filterInput(const uint32_t timestamp, const tf2::Transform& tf2Transform)
        {
            // convert time from nanosecond scale to sec scale
            // auto dt = (timestamp - prev_timestamp_) / 1.0e9;
        
            // initial obsrvation is very noisy use lidar point observation model to filter noise
            // process liadr measurement to make prediction
            Lidar::Measurement lidar_meas = stateToLidar(timestamp, tf2Transform);
            BEL belief_initial{ukf_->ProcessMeasurement(lidar_meas)};
            //use control and process model to make prediction
            // auto belief_prior{UKF::Predict(belief_initial, control_vector_,   dt, pm_)};
            // fuse apriltag with odom velocity information with the radar model to update state
            Radar::Measurement radar_meas = stateToRadar(timestamp, beliefToTF(belief_initial));
            auto belief{ukf_->ProcessMeasurement(radar_meas)};
            auto state = beliefToTF(belief);
            pub_robot_state(state);
           

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