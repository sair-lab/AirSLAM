/**
*
*/

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// AirVO
#include "read_configs.h"
#include "dataset.h"
#include "map_builder.h"

MapBuilder* p_map_builder;

//////////////////////////////////////////////////
// Functions
//////////////////////////////////////////////////

void GrabStereo(const sensor_msgs::ImageConstPtr& imgLeft, const sensor_msgs::ImageConstPtr& imgRight)
{
    // Copy the ros image messages to cvMat
    cv_bridge::CvImageConstPtr cv_ptrLeft, cv_ptrRight;
    try
    {
        cv_ptrLeft = cv_bridge::toCvShare(imgLeft);
        cv_ptrRight = cv_bridge::toCvShare(imgRight);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
    static int frame_id = 0;
    auto before_infer = std::chrono::steady_clock::now();

    InputDataPtr input_data = std::shared_ptr<InputData>(new InputData());
    input_data->index = frame_id;
    input_data->image_left = cv_ptrLeft->image.clone();
    input_data->image_right = cv_ptrRight->image.clone();
    input_data->time = imgLeft->header.stamp.toSec();

    if(input_data == nullptr) return;
    p_map_builder->AddInput(input_data);

    auto after_infer = std::chrono::steady_clock::now();
    auto cost_time = std::chrono::duration_cast<std::chrono::milliseconds>
    (after_infer - before_infer).count();

    std::cout << "i ===== " << frame_id++ << " Processing Time: " << cost_time << " ms." << std::endl;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "air_vo_ros");

    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    if (argc > 1)
    {
        ROS_WARN ("Arguments supplied via command line are ignored.");
    }

    // ROS
    ros::NodeHandle node_handler;
    message_filters::Subscriber<sensor_msgs::Image> sub_img_left(node_handler, "/cam0/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> sub_img_right(node_handler, "/cam1/image_raw", 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), sub_img_left, sub_img_right);
    sync.registerCallback(boost::bind(&GrabStereo, _1, _2));

    // AirVO
    std::string config_path, model_dir;
    ros::param::get("~config_path", config_path);
    ros::param::get("~model_dir", model_dir);
    Configs configs(config_path, model_dir);
    ros::param::get("~dataroot", configs.dataroot); // unused for real-time operation
    ros::param::get("~camera_config_path", configs.camera_config_path);
    ros::param::get("~saving_dir", configs.saving_dir);
    std::string traj_path;
    ros::param::get("~traj_path", traj_path);

    p_map_builder = new MapBuilder(configs);

    // Starts the operation
    ros::spin();

    // Shutting down
    std::cout << "Saving trajectory to " << traj_path << std::endl;
    p_map_builder->SaveTrajectory(traj_path);

    ros::shutdown();

    return 0;
}