//
// Created by redwan on 7/31/23.
//

#include "apriltag_fusion.h"
#include "timestamp_logger.h"





namespace airlab
{
    apriltag_fusion::apriltag_fusion(const rclcpp::NodeOptions& options): rclcpp::Node("apriltag_fusion", options), filterInit_(false)
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
            control_vector_ << acc_linear, acc_linear, acc_linear, acc_angular, acc_angular;
        });

        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>("/apriltag/pose_detections", qos, [&](geometry_msgs::msg::PoseArray::SharedPtr msg){
            apriltag_callback(msg);
        });


        timer_ = this->create_wall_timer(250ms, [this] { 
            if(filterInit_)
            {
                rvizTimerCallback();
            }                
        });
        estimator_loop_ = this->create_wall_timer(10ms, [this] {

            if (filterInit_)
            {
                estimatorTimerCallback();
            }
        });
        RCLCPP_INFO(get_logger(), "apriltag fusion node started");
    }

    void apriltag_fusion::estimatorTimerCallback()
    {
        auto currentTime = TimestampLogger::getInstance().getElapsedTime();
        auto dt = (currentTime - prev_timestamp_) / 1.0e3; // sec 
        dt =  dt / 100.0; // sec
        // std::cout << dt1 << std::endl;  


        filterInput(currentTime, apriltagMeas_);
        auto belief_initial = ukf_->GetBelief();
        auto belief_prior{UKF::Predict(belief_initial, control_vector_,   dt, pm_)};
        Radar::Measurement radar_meas = stateToRadar(currentTime, beliefToTF(belief_prior));
        BEL belief_posterior{UKF::Update(belief_prior, radar_meas, radar_mm_)};        
        robotState_ = beliefToTF(belief_posterior);
        ukf_->SetBelief(belief_posterior);
        pub_robot_state(robotState_);    
    }

    void apriltag_fusion::apriltag_callback(geometry_msgs::msg::PoseArray::SharedPtr msg)
    {
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
                auto currentTime = TimestampLogger::getInstance().getElapsedTime();
                
                tf2::fromMsg(pose, apriltagMeas_);
                if (!filterInit_)
                {
                    filterInit_ = true;
                }
                
                prev_timestamp_ = currentTime;

            }
        }
    }
 

    void apriltag_fusion::ukfInit()
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
        radar_mm_.SetMeasurementCovarianceMatrix(radar_mtx);
        ukf_= std::make_unique<UKF_CTRV_LIDAR_RADAR_Fusion>(ctrv_mtx, lidar_mtx, radar_mtx);
    }



    
    Radar::Measurement apriltag_fusion::stateToRadar(const uint32_t timestamp, const tf2::Transform& tf2Transform)
    {
        auto origin = tf2Transform.getOrigin();
        double rho = sqrt(origin.x() * origin.x() + origin.y() * origin.y());
        double alpha = atan2(origin.y(), origin.x());

        Radar::MeasurementVector mv;
        mv << rho, alpha, rho_dot_;
        auto time = timestamp / 1.0e3;
        Radar::Measurement radar_meas{time, mv};
        return radar_meas;
    }

    Lidar::Measurement apriltag_fusion::stateToLidar(const uint32_t timestamp, const tf2::Transform& tf2Transform)
    {
        auto origin = tf2Transform.getOrigin();
        Lidar::MeasurementVector mv;
        mv << origin.x(), origin.y();
        auto time = timestamp / 1.0e3;
        Lidar::Measurement lidar_meas{time, mv};
        return lidar_meas;
    }

    tf2::Transform apriltag_fusion::beliefToTF(const BEL& belief)
    {
        const auto& sv{belief.mu()};
        CTRV::ROStateVectorView state_vector_view{sv};

        tf2::Transform state;
        auto origin = tf2::Vector3(state_vector_view.px(), state_vector_view.py(), 0);
        state.setRotation(apriltagMeas_.getRotation());
        state.setOrigin(origin);
        return state;
    }

    void apriltag_fusion::rvizTimerCallback()
    {
        state_callback(robotState_, DEFAULT);
        publish_traj();
    }


    void apriltag_fusion::filterInput(const uint32_t timestamp, const tf2::Transform& tf2Transform)
    {
        // initial obsrvation is very noisy use lidar point observation model to filter noise
        // process liadr measurement to make prediction
        Lidar::Measurement lidar_meas = stateToLidar(timestamp, tf2Transform);
        BEL belief_initial{ukf_->ProcessMeasurement(lidar_meas)};
        // fuse apriltag with odom velocity information with the radar model to update state
        Radar::Measurement radar_meas = stateToRadar(timestamp, beliefToTF(belief_initial));
        auto belief{ukf_->ProcessMeasurement(radar_meas)};
    }

    void apriltag_fusion::pub_robot_state(const tf2::Transform& t)
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




    
    void apriltag_fusion::state_callback(const tf2::Transform& tf, const COLOR& color)
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

    void apriltag_fusion::publish_traj()
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
}

RCLCPP_COMPONENTS_REGISTER_NODE(airlab::apriltag_fusion)