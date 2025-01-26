#include "mppi_controller/mppi_controller_ros.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "mppi_controller");
    ROS_INFO("MPPI Controller node initializing...");
    mppi::MPPIControllerROS mppi_controller;
    ROS_INFO("MPPI Controller node successfully initialized.");
    ros::spin();
    return 0;
};
