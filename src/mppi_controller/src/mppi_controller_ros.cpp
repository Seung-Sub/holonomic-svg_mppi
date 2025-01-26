#include "mppi_controller/mppi_controller_ros.hpp"

namespace mppi {
MPPIControllerROS::MPPIControllerROS() : nh_(""), private_nh_("~"), tf_listener_(tf_buffer_) {

    
    // set parameters from ros parameter server
    private_nh_.param("is_simulation", is_simulation_, false);
    // private_nh_.param("is_localize_less_mode", is_localize_less_mode_, false);

    private_nh_.param("control_sampling_time", control_sampling_time_, 0.1);

    private_nh_.param("is_visualize_mppi", is_visualize_mppi_, false);
    // private_nh_.param("constant_speed_mode", constant_speed_mode_, false);

    std::string control_cmd_topic;
    std::string in_reference_sdf_topic;
    std::string in_odom_topic;
    std::string is_activate_ad_topic;
    std::string occupacy_grid_id;
    std::string local_costmap_id;
    std::string in_backward_point_topic;

    private_nh_.param("control_cmd_topic", control_cmd_topic, static_cast<std::string>("/robot/move_base/cmd_vel"));
    private_nh_.param("in_reference_sdf_topic", in_reference_sdf_topic, static_cast<std::string>("/robot/move_base/GlobalPlanner/plan"));
    private_nh_.param("in_odom_topic", in_odom_topic, static_cast<std::string>("robot/robotnik_base_control/odom"));
    private_nh_.param("robot_frame_id", robot_frame_id_, static_cast<std::string>("robot_base_link"));
    private_nh_.param("map_frame_id", map_frame_id_, static_cast<std::string>("robot_map"));
    private_nh_.param("costmap_id", occupacy_grid_id, static_cast<std::string>("costmap"));
    private_nh_.param("local_costmap_id", local_costmap_id, static_cast<std::string>("local_costmap"));
    private_nh_.param("backward_point_topic", in_backward_point_topic, static_cast<std::string>("backward_point"));

    private_nh_.param("is_activate_ad_topic", is_activate_ad_topic, static_cast<std::string>("is_active_ad"));
    private_nh_.param("collision_rate_threshold", collision_rate_threshold_, 0.95);
    private_nh_.param("speed_queue_size", speed_deque_size_, static_cast<int>(10));
    private_nh_.param("stuck_speed_threshold", stuck_speed_threshold_, static_cast<float>(0.1));

    // fill deque
    for (int i = 0; i < speed_deque_size_; i++) {
        const double enough_speed = stuck_speed_threshold_ + 1.0;
        speed_deque_.push_back(enough_speed);
    }

    // load params
    Params params;

    private_nh_.param("common/thread_num", params.common.thread_num, 12);
    private_nh_.param("common/prediction_step_size", params.common.prediction_step_size, 10);
    private_nh_.param("common/prediction_interval", params.common.prediction_interval, 0.1);


    private_nh_.param("common/q_dist", params.common.q_dist, 1.0);
    private_nh_.param("common/q_angle", params.common.q_angle, 0.0);
    private_nh_.param("common/collision_weight", params.common.collision_weight, 0.01);
    private_nh_.param("common/q_terminal_dist", params.common.q_terminal_dist, 1.0);
    private_nh_.param("common/q_terminal_angle", params.common.q_terminal_angle, 0.0);

    // Forward MPPI params
    private_nh_.param("forward_mppi/sample_batch_num", params.forward_mppi.sample_batch_num, 1000);
    private_nh_.param("forward_mppi/lambda", params.forward_mppi.lambda, 10.0);
    private_nh_.param("forward_mppi/alpha", params.forward_mppi.alpha, 0.1);
    private_nh_.param("forward_mppi/non_biased_sampling_rate", params.forward_mppi.non_biased_sampling_rate, 0.1);
    private_nh_.param("forward_mppi/Vx_cov", params.forward_mppi.Vx_cov, 0.01);
    private_nh_.param("forward_mppi/Vy_cov", params.forward_mppi.Vy_cov, 0.01);
    private_nh_.param("forward_mppi/w_cov", params.forward_mppi.w_cov, 0.01);
    private_nh_.param("forward_mppi/num_itr_for_grad_estimation", params.forward_mppi.num_itr_for_grad_estimation, 1);
    private_nh_.param("forward_mppi/step_size_for_grad_estimation", params.forward_mppi.step_size_for_grad_estimation, 0.1);
    private_nh_.param("forward_mppi/sample_num_for_grad_estimation", params.forward_mppi.sample_num_for_grad_estimation, 10);
    private_nh_.param("forward_mppi/Vx_cov_for_grad_estimation", params.forward_mppi.Vx_cov_for_grad_estimation, 0.01);
    private_nh_.param("forward_mppi/Vy_cov_for_grad_estimation", params.forward_mppi.Vy_cov_for_grad_estimation, 0.01);
    private_nh_.param("forward_mppi/w_cov_for_grad_estimation", params.forward_mppi.w_cov_for_grad_estimation, 0.01);

    // Stein Variational Guided MPPI params
    private_nh_.param("svg_mppi/sample_batch_num", params.svg_mppi.sample_batch_num, 1000);
    private_nh_.param("svg_mppi/lambda", params.svg_mppi.lambda, 10.0);
    private_nh_.param("svg_mppi/alpha", params.svg_mppi.alpha, 0.1);
    private_nh_.param("svg_mppi/non_biased_sampling_rate", params.svg_mppi.non_biased_sampling_rate, 0.1);
    // private_nh_.param("svg_mppi/steer_cov", params.svg_mppi.steer_cov, 0.5);
    // private_nh_.param("svg_mppi/accel_cov", params.accel_cov, 0.5);
    private_nh_.param("svg_mppi/Vx_cov", params.svg_mppi.Vx_cov, 0.01);
    private_nh_.param("svg_mppi/Vy_cov", params.svg_mppi.Vy_cov, 0.01);
    private_nh_.param("svg_mppi/w_cov", params.svg_mppi.w_cov, 0.01);

    private_nh_.param("svg_mppi/guide_sample_num", params.svg_mppi.guide_sample_num, 1);
    private_nh_.param("svg_mppi/grad_lambda", params.svg_mppi.grad_lambda, 3.0);
    private_nh_.param("svg_mppi/sample_num_for_grad_estimation", params.svg_mppi.sample_num_for_grad_estimation, 10);
    // private_nh_.param("svg_mppi/steer_cov_for_grad_estimation", params.svg_mppi.steer_cov_for_grad_estimation, 0.05);
    private_nh_.param("svg_mppi/Vx_cov_for_grad_estimation", params.svg_mppi.Vx_cov_for_grad_estimation, 0.01);
    private_nh_.param("svg_mppi/Vy_cov_for_grad_estimation", params.svg_mppi.Vy_cov_for_grad_estimation, 0.01);
    private_nh_.param("svg_mppi/w_cov_for_grad_estimation", params.svg_mppi.w_cov_for_grad_estimation, 0.01);

    private_nh_.param("svg_mppi/svgd_step_size", params.svg_mppi.svgd_step_size, 0.1);
    private_nh_.param("svg_mppi/num_svgd_iteration", params.svg_mppi.num_svgd_iteration, 100);
    private_nh_.param("svg_mppi/is_use_nominal_solution", params.svg_mppi.is_use_nominal_solution, true);
    private_nh_.param("svg_mppi/is_covariance_adaptation", params.svg_mppi.is_covariance_adaptation, true);
    private_nh_.param("svg_mppi/gaussian_fitting_lambda", params.svg_mppi.gaussian_fitting_lambda, 0.1);
    // private_nh_.param("svg_mppi/min_steer_cov", params.svg_mppi.min_steer_cov, 0.001);
    // private_nh_.param("svg_mppi/max_steer_cov", params.svg_mppi.max_steer_cov, 0.1);
    private_nh_.param("svg_mppi/max_Vx_cov", params.svg_mppi.max_Vx_cov, 0.1);
    private_nh_.param("svg_mppi/min_Vx_cov", params.svg_mppi.min_Vx_cov, 0.001);
    private_nh_.param("svg_mppi/max_Vy_cov", params.svg_mppi.max_Vy_cov, 0.1);
    private_nh_.param("svg_mppi/min_Vy_cov", params.svg_mppi.min_Vy_cov, 0.001);
    private_nh_.param("svg_mppi/max_w_cov", params.svg_mppi.max_w_cov, 0.1);
    private_nh_.param("svg_mppi/min_w_cov", params.svg_mppi.min_w_cov, 0.001);

    // Initialize MPC solver based on mpc_mode
    const std::string mpc_mode = private_nh_.param<std::string>("mpc_mode", "");
    if (mpc_mode == "forward_mppi") {
        mpc_solver_ptr_ = std::make_unique<mppi::cpu::ForwardMPPI>(params.common, params.forward_mppi);
    // } else if (mpc_mode == "reverse_mppi") {
    //     mpc_solver_ptr_ = std::make_unique<mppi::cpu::ReverseMPPI>(params.common, params.reverse_mppi);
    // } else if (mpc_mode == "sv_mpc") {
    //     mpc_solver_ptr_ = std::make_unique<mppi::cpu::SteinVariationalMPC>(params.common, params.stein_variational_mpc);
    } else if (mpc_mode == "svg_mppi") {
        mpc_solver_ptr_ = std::make_unique<mppi::cpu::SVGuidedMPPI>(params.common, params.svg_mppi);
    } else {
        ROS_ERROR("Invalid MPC mode: %s", mpc_mode.c_str());
        exit(1);
    }

    // set publishers and subscribers
    pub_cmd_vel_ = nh_.advertise<geometry_msgs::Twist>(control_cmd_topic, 1);
    
    // global path subscriber (nav_msgs::Path)
    sub_global_path_ = nh_.subscribe(in_reference_sdf_topic, 1,&MPPIControllerROS::callbackGlobalPath, this);

    // obstacle map
    sub_grid_map_ = nh_.subscribe(local_costmap_id, 1, &MPPIControllerROS::callback_grid_map, this);

    // odom
    sub_odom_ = nh_.subscribe(in_odom_topic, 1, &MPPIControllerROS::callback_odom_with_pose, this);

    // activate AD
    sub_activated_ = nh_.subscribe(is_activate_ad_topic, 1, &MPPIControllerROS::callback_activate_signal, this);
    
    // timer for control
    timer_control_ = nh_.createTimer(ros::Duration(control_sampling_time_), &MPPIControllerROS::timer_callback, this);
    
    sub_start_cmd_ = nh_.subscribe("mppi/start", 1, &MPPIControllerROS::start_cmd_callback, this);
    sub_stop_cmd_ = nh_.subscribe("mppi/stop", 1, &MPPIControllerROS::stop_cmd_callback, this);

    // For debug
    pub_best_path_ = nh_.advertise<visualization_msgs::MarkerArray>("mppi/best_path", 1, true);
    pub_nominal_path_ = nh_.advertise<visualization_msgs::MarkerArray>("mppi/nominal_path", 1, true);
    pub_candidate_paths_ = nh_.advertise<visualization_msgs::MarkerArray>("mppi/candidate_paths", 1, true);
    pub_proposal_state_distributions_ = nh_.advertise<visualization_msgs::MarkerArray>("mppi/proposal_state_distributions", 1, true);
    pub_control_covariances_ = nh_.advertise<visualization_msgs::MarkerArray>("mppi/control_covariances", 1, true);
    pub_calculation_time_ = nh_.advertise<std_msgs::Float32>("mppi/calculation_time", 1, true);
    pub_speed_ = nh_.advertise<std_msgs::Float32>("mppi/speed", 1, true);
    pub_collision_rate_ = nh_.advertise<std_msgs::Float32>("mppi/collision_rate", 1, true);
    // Publishers and Subscribers
    pub_cost_ = nh_.advertise<std_msgs::Float32>("mppi/cost", 1, true);
    pub_mppi_metrics_ = nh_.advertise<mppi_metrics_msgs::MPPIMetrics>("mppi/eval_metrics", 1, true);

    // init state
    robot_state_.x = 0.0;
    robot_state_.y = 0.0;
    robot_state_.yaw= 0.0;
    is_robot_state_ok_ = false;
    is_costmap_ok_ = false;
    is_activate_ad_ = false;
    is_start_ = false;

    // Log parameter loading completion
    ROS_INFO_THROTTLE(1.0, "Parameters loaded: is_simulation=%s", is_simulation_ ? "true" : "false");
    ROS_INFO_THROTTLE(1.0, "Parameters loaded: control_sampling_time=%.2f", control_sampling_time_);
    ROS_INFO_THROTTLE(1.0, "Parameters loaded: mpc_mode=%s", mpc_mode.c_str());

    // Log MPC solver initialization
    if (mpc_solver_ptr_) {
        ROS_INFO_THROTTLE(1.0, "MPC solver (%s) initialized", mpc_mode.c_str());
    } else {
        ROS_ERROR("MPC solver initialization failed");
    }

    
}

/**
 * @brief callback for global path
 */
void MPPIControllerROS::callbackGlobalPath(const nav_msgs::PathConstPtr& msg)
{
    if (!mpc_solver_ptr_) {
        ROS_WARN("[MPPIControllerROS] mpc_solver_ptr_ not ready");
        return;
    }
    mpc_solver_ptr_->set_reference_path(*msg);
    // ROS_INFO_THROTTLE(1.0"[MPPIControllerROS] Global path received, size=%zu", msg->poses.size());  
    // 추가: 참조 경로의 첫 포인트와 로봇 위치 비교
    if (!msg->poses.empty()) {
        double ref_x = msg->poses[0].pose.position.x;
        double ref_y = msg->poses[0].pose.position.y;
        // ROS_INFO_THROTTLE(2.0, "Reference Path First Point: x=%.2f, y=%.2f", ref_x, ref_y);
    }
}

// Get current pose and velocity used with localization model
void MPPIControllerROS::callback_odom_with_pose(const nav_msgs::Odometry& odom) {
    /*Get current pose via tf*/
    geometry_msgs::TransformStamped trans_form_stamped;
    try {
        trans_form_stamped = tf_buffer_.lookupTransform(map_frame_id_, robot_frame_id_, ros::Time(0));
    } catch (const tf2::TransformException& ex) {
        ROS_WARN_THROTTLE(3.0, "[MPPIControllerROS] TF transform failed: %s", ex.what());
        return;
    };

    // Print transform to see if it's valid
    ROS_INFO_THROTTLE(1.0, "[MPPIControllerROS] transform %s -> %s at time=%.3f",
                      map_frame_id_.c_str(), robot_frame_id_.c_str(),
                      trans_form_stamped.header.stamp.toSec());
    ROS_INFO_THROTTLE(1.0, "  translation=(%.3f, %.3f, %.3f)",
                      trans_form_stamped.transform.translation.x,
                      trans_form_stamped.transform.translation.y,
                      trans_form_stamped.transform.translation.z);
    // ROS_INFO_THROTTLE(1.0, "  rotation=(%.3f, %.3f, %.3f, %.3f)",
    //                   trans_form_stamped.transform.rotation.x,
    //                   trans_form_stamped.transform.rotation.y,
    //                   trans_form_stamped.transform.rotation.z,
    //                   trans_form_stamped.transform.rotation.w);
    /*Update status*/
    robot_state_.x = trans_form_stamped.transform.translation.x;
    robot_state_.y = trans_form_stamped.transform.translation.y;
    const double _yaw = tf2::getYaw(trans_form_stamped.transform.rotation);
    robot_state_.yaw = std::atan2(std::sin(_yaw), std::cos(_yaw));
    

    is_robot_state_ok_ = true;

    // Confirm final x,y,yaw
    ROS_INFO_THROTTLE(1.0, "[MPPIControllerROS] Robot state updated: x=%.3f, y=%.3f, yaw=%.3f",
                      robot_state_.x, robot_state_.y, robot_state_.yaw);
}


void MPPIControllerROS::callback_activate_signal(const std_msgs::Bool& msg)
{
    is_activate_ad_ = msg.data;
}

void MPPIControllerROS::callback_grid_map(const grid_map_msgs::GridMap& grid_map) {
    // make grid map for obstacle layer
    if (!grid_map::GridMapRosConverter::fromMessage(grid_map, obstacle_map_)) {
        ROS_ERROR("[MPPIControllerROS]Failed to convert grid map to grid map");
        return;
    }

    is_costmap_ok_ = true;
    // ROS_INFO_THROTTLE(2.0, "Obstacle map updated with size: [%u x %u]", obstacle_map_.getSize()(0), obstacle_map_.getSize()(1));
}


void MPPIControllerROS::start_cmd_callback([[maybe_unused]] const std_msgs::Empty& msg) {
    is_start_ = true;
    ROS_INFO_THROTTLE(1.0, "[MPD] start cmd received");
}

void MPPIControllerROS::stop_cmd_callback([[maybe_unused]] const std_msgs::Empty& msg) {
    is_start_ = false;
    ROS_INFO_THROTTLE(1.0, "[MPD] stop cmd received");
}

// Control loop
void MPPIControllerROS::timer_callback([[maybe_unused]] const ros::TimerEvent& te) {
    // ROS_INFO_THROTTLE(1.0, "Timer callback started");

    // Check readiness
    if (!is_robot_state_ok_ || !is_costmap_ok_) {
        ROS_WARN_THROTTLE(2.0, "[MPPIControllerROS] Not ready: state=%d, costmap=%d", is_robot_state_ok_, is_costmap_ok_);
        return;
    }

    // 활성화 상태 체크
    if (!is_activate_ad_ && !is_simulation_) {
        // 조작 불가 → zero velocity
        geometry_msgs::Twist zero_cmd;
        pub_cmd_vel_.publish(zero_cmd);
        ROS_WARN_THROTTLE(2.0, "[MPPIControllerROS] Not activated");
        return;
    }

    // 스타트 신호 체크
    if (!is_start_) {
        // 아직 스타트 명령 안 받음
        geometry_msgs::Twist zero_cmd;
        pub_cmd_vel_.publish(zero_cmd);
        ROS_WARN_THROTTLE(2.0, "[MPPIControllerROS] Waiting for start signal");
        return;
    }

    
    mtx_.lock();

    stop_watch_.lap();

    mpc_solver_ptr_->set_obstacle_map(obstacle_map_);

    // // Solve MPC
    // mppi::cpu::State initial_state = mppi::cpu::State::Zero();
    // initial_state[STATE_SPACE::x] = robot_state_.x;
    // initial_state[STATE_SPACE::y] = robot_state_.y;
    // initial_state[STATE_SPACE::yaw] = robot_state_.yaw;
    // // initial_state[STATE_SPACE::vel] = robot_state_.vel;
    // // initial_state[STATE_SPACE::steer] = robot_state_.steer;
    // const auto [updated_control_seq, collision_rate] = mpc_solver_ptr_->solve(initial_state);

    // mtx_.unlock();

    // 현재 상태
    mppi::cpu::State init_state;
    init_state << robot_state_.x, robot_state_.y, robot_state_.yaw;

    // Solve -> best control sequence
    auto [best_control_seq, collision_rate] = mpc_solver_ptr_->solve(init_state);

    double calc_time = stop_watch_.lap();
    mtx_.unlock();

    // ROS_INFO_THROTTLE(1.0, "MPPI solver execution completed: Collision Rate=%.2f, Calculation Time=%.2f", collision_rate, calc_time);

    // 제어 명령 퍼블리시
    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x  = best_control_seq(0, mppi::CONTROL_SPACE::Vx);
    cmd_vel.linear.y  = best_control_seq(0, mppi::CONTROL_SPACE::Vy);
    cmd_vel.angular.z = best_control_seq(0, mppi::CONTROL_SPACE::w);
    pub_cmd_vel_.publish(cmd_vel);
    ROS_INFO_THROTTLE(3.0, "Control command published: Vx=%.2f, Vy=%.2f, w=%.2f", cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.angular.z);
    

    // Predict state sequence
    const auto [best_state_seq, state_cost, collision_cost, input_error] = mpc_solver_ptr_->get_predictive_seq(init_state, best_control_seq);

    // First control input
    double vx_cmd = best_control_seq(0, mppi::CONTROL_SPACE::Vx);
    double vy_cmd = best_control_seq(0, mppi::CONTROL_SPACE::Vy);
    double w_cmd  = best_control_seq(0, mppi::CONTROL_SPACE::w);

    // Publish Twist message
    geometry_msgs::Twist cmd_vel_twist;
    cmd_vel_twist.linear.x  = vx_cmd;
    cmd_vel_twist.linear.y  = vy_cmd;
    cmd_vel_twist.angular.z = w_cmd;
    pub_cmd_vel_.publish(cmd_vel_twist);

    // control_msg_.header.stamp = ros::Time::now();

    const double calculation_time = stop_watch_.lap();

    // ==========  For debug ===============
    if (is_visualize_mppi_) {
        // publish predicted state
        const int num_visualized_samples = 100;
        const auto [predict_state_seq_batch, weights] = mpc_solver_ptr_->get_state_seq_candidates(num_visualized_samples);
        publish_candidate_paths(predict_state_seq_batch, weights, pub_candidate_paths_);

        // Get covariance of proposed distribution
        const auto cov_matrices = mpc_solver_ptr_->get_cov_matrices();

        // publish best state
        publish_traj(best_state_seq, "best_path", "red", pub_best_path_);

        // publish control covariance
        publish_control_covs(best_state_seq, cov_matrices, pub_control_covariances_);

        // publish proposed state distribution
        const auto [mean, xycov_mat] = mpc_solver_ptr_->get_proposed_state_distribution();
        publish_state_seq_dists(mean, xycov_mat, pub_proposal_state_distributions_);

        // publish nominal state
        const auto nominal_control_seq = mpc_solver_ptr_->get_control_seq();
        const auto nominal_state_seq = std::get<0>(mpc_solver_ptr_->get_predictive_seq(init_state, nominal_control_seq));
        publish_path(nominal_state_seq, "nominal_path", "g", pub_nominal_path_);
    } else {
        // publish best state
        publish_traj(best_state_seq, "best_path", "red", pub_best_path_);
    }

    // publish cost of the best state seq
    std_msgs::Float32 cost_msg;
    cost_msg.data = static_cast<float>(state_cost + collision_cost);
    pub_cost_.publish(cost_msg);

    // // publish speed
    // std_msgs::Float32 speed_msg;
    // speed_msg.data = robot_state_.vel;
    // pub_speed_.publish(speed_msg);

    // publish calculate time
    std_msgs::Float32 calculation_time_msg;
    calculation_time_msg.data = static_cast<float>(calculation_time);
    pub_calculation_time_.publish(calculation_time_msg);

    // publish collision rate
    std_msgs::Float32 collision_rate_msg;
    collision_rate_msg.data = static_cast<float>(collision_rate);
    pub_collision_rate_.publish(collision_rate_msg);

    // publish mppi metrics
    mppi_metrics_msgs::MPPIMetrics mppi_metrics_msg;
    mppi_metrics_msg.header.stamp = ros::Time::now();
    mppi_metrics_msg.calculation_time = calculation_time;
    mppi_metrics_msg.state_cost = state_cost;
    mppi_metrics_msg.collision_cost = collision_cost;
    mppi_metrics_msg.input_error = input_error;
    pub_mppi_metrics_.publish(mppi_metrics_msg);
}

void MPPIControllerROS::publish_traj(const mppi::cpu::StateSeq& state_seq,
                                     const std::string& name_space,
                                     const std::string& rgb,
                                     const ros::Publisher& publisher) const {
    visualization_msgs::MarkerArray marker_array;

    visualization_msgs::Marker arrow;
    arrow.header.frame_id = map_frame_id_; // or robot_frame_id_, depending on mode
    arrow.header.stamp = ros::Time::now();
    arrow.ns = name_space;
    arrow.type = visualization_msgs::Marker::ARROW;
    arrow.action = visualization_msgs::Marker::ADD;

    geometry_msgs::Vector3 arrow_scale;
    arrow_scale.x = 0.02;  // shaft diameter
    arrow_scale.y = 0.04;  // head diameter
    arrow_scale.z = 0.1;   // head length
    arrow.scale = arrow_scale;

    arrow.pose.orientation.w = 1.0;
    arrow.color.a = 1.0;
    arrow.color.r = ((rgb == "r" || rgb == "red") ? 1.0 : 0.0);
    arrow.color.g = ((rgb == "g" || rgb == "green") ? 1.0 : 0.0);
    arrow.color.b = ((rgb == "b" || rgb == "blue") ? 1.0 : 0.0);

    arrow.points.resize(2);

    for (int i = 0; i < state_seq.rows(); i++) {
        arrow.id = i;

        double x   = state_seq(i, STATE_SPACE::x);
        double y   = state_seq(i, STATE_SPACE::y);
        double yaw = state_seq(i, STATE_SPACE::yaw);

        // 화살표 길이(예: 0.3m)
        const double length = 0.3; 
        geometry_msgs::Point start, end;
        start.x = x;
        start.y = y;
        start.z = 0.1;

        end.x = x + length * cos(yaw);
        end.y = y + length * sin(yaw);
        end.z = 0.1;

        arrow.points[0] = start;
        arrow.points[1] = end;

        marker_array.markers.push_back(arrow);
    }
    publisher.publish(marker_array);
}

void MPPIControllerROS::publish_path(
    const mppi::cpu::StateSeq& state_seq,
    const std::string& name_space,
    const std::string& rgb,
    const ros::Publisher& publisher) const{
    visualization_msgs::MarkerArray marker_array;

    // LINE_STRIP
    visualization_msgs::Marker line;
    line.header.frame_id = map_frame_id_;
    line.header.stamp = ros::Time::now();
    line.ns = name_space + "_line";
    line.id = 0;
    line.type = visualization_msgs::Marker::LINE_STRIP;
    line.action = visualization_msgs::Marker::ADD;
    line.scale.x = 0.01; // thickness
    line.color.a = 1.0;
    line.color.r = ((rgb == "r" || rgb == "red") ? 1.0 : 0.0);
    line.color.g = ((rgb == "g" || rgb == "green") ? 1.0 : 0.0);
    line.color.b = ((rgb == "b" || rgb == "blue") ? 1.0 : 0.0);

    // node points (SPHERE_LIST)
    visualization_msgs::Marker nodes;
    nodes.header.frame_id = map_frame_id_;
    nodes.header.stamp = ros::Time::now();
    nodes.ns = name_space + "_nodes";
    nodes.id = 1;
    nodes.type = visualization_msgs::Marker::SPHERE_LIST;
    nodes.action = visualization_msgs::Marker::ADD;
    double node_size = 0.1;
    nodes.scale.x = node_size;
    nodes.scale.y = node_size;
    nodes.scale.z = node_size;
    nodes.color.a = 1.0;
    nodes.color.r = line.color.r;
    nodes.color.g = line.color.g;
    nodes.color.b = line.color.b;

    for (int i = 0; i < state_seq.rows(); i++) {
        geometry_msgs::Point pt;
        pt.x = state_seq(i, mppi::STATE_SPACE::x);
        pt.y = state_seq(i, mppi::STATE_SPACE::y);
        pt.z = 0.1;

        line.points.push_back(pt);
        nodes.points.push_back(pt);
    }
    marker_array.markers.push_back(line);
    marker_array.markers.push_back(nodes);

    publisher.publish(marker_array);
}


void MPPIControllerROS::publish_candidate_paths(
    const std::vector<mppi::cpu::StateSeq>& state_seq_batch,
    const std::vector<double>& weights,
    const ros::Publisher& publisher) const
{
    // sanity check
    assert(state_seq_batch.size() == weights.size());
    if (weights.empty()) {
        // nothing to visualize
        return;
    }

    visualization_msgs::MarkerArray marker_array;

    // find max_weight for scaling
    const double max_weight = weights[0];
    const double max_node_size = 0.05;

    for (size_t i = 0; i < state_seq_batch.size(); i++) {
        // 1) LINE_STRIP
        visualization_msgs::Marker line;
        // Always use map_frame_id_ (or robot_frame_id_, up to your usage):
        line.header.frame_id = map_frame_id_;
        line.header.stamp = ros::Time::now();
        line.ns = "candidate_path_line";
        line.id = i;
        line.type = visualization_msgs::Marker::LINE_STRIP;
        line.action = visualization_msgs::Marker::ADD;
        line.pose.orientation.w = 1.0;
        line.scale.x = 0.01;
        line.color.a = 0.3;
        line.color.r = 0.0;
        line.color.g = 0.5;
        line.color.b = 1.0;

        // 2) SPHERE_LIST for nodes
        visualization_msgs::Marker nodes;
        nodes.header.frame_id = map_frame_id_;
        nodes.header.stamp = ros::Time::now();
        nodes.ns = "candidate_path_nodes";
        nodes.id = line.id + 10000 + i;
        nodes.type = visualization_msgs::Marker::SPHERE_LIST;
        nodes.action = visualization_msgs::Marker::ADD;
        nodes.pose.orientation.w = 1.0;
        // scale
        nodes.scale.x = weights[i] * max_node_size / max_weight + 0.01;
        nodes.scale.y = 0.01;
        nodes.scale.z = 0.01;
        // color
        nodes.color.a = 0.6;
        nodes.color.r = 0.0;
        nodes.color.g = 0.5;
        nodes.color.b = 1.0;

        // Fill line.points / nodes.points
        // e.g. state_seq_batch.at(i)(j, STATE_SPACE::x) => x coord
        const int n_rows = state_seq_batch.at(i).rows();
        for (int j = 0; j < n_rows; j++) {
            geometry_msgs::Point pt;
            pt.x = state_seq_batch[i](j, mppi::STATE_SPACE::x);
            pt.y = state_seq_batch[i](j, mppi::STATE_SPACE::y);
            pt.z = 0.0;
            line.points.push_back(pt);
            nodes.points.push_back(pt);
        }

        marker_array.markers.push_back(line);
        marker_array.markers.push_back(nodes);
    }

    // publish
    publisher.publish(marker_array);
}


void MPPIControllerROS::publish_control_covs(
    const mppi::cpu::StateSeq& state_seq,
    const mppi::cpu::ControlSeqCovMatrices& cov_matrices,
    const ros::Publisher& publisher) const{
    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker ellipse;
    ellipse.header.frame_id = map_frame_id_;
    ellipse.header.stamp = ros::Time::now();
    ellipse.ns = "angular_covariance";
    ellipse.type = visualization_msgs::Marker::SPHERE;
    ellipse.action = visualization_msgs::Marker::ADD;
    ellipse.color.a = 0.4;
    ellipse.color.r = 1.0; // yellow-ish
    ellipse.color.g = 1.0;
    ellipse.color.b = 0.0;

    // for scaling
    double control_cov_scale = 1.5;

    // cov_matrices.size() == (prediction_step_size-1)
    // state_seq.size() == (prediction_step_size)
    // i < state_seq.rows()-1
    for (int i = 0; i < (int)cov_matrices.size(); i++) {
        ellipse.id = i;

        // w축 공분산
        double cov = cov_matrices[i](mppi::CONTROL_SPACE::w, mppi::CONTROL_SPACE::w);
        double std_dev = sqrt(cov) * control_cov_scale;

        // 크기 (2sigma)
        geometry_msgs::Vector3 scale;
        scale.x = 2.0 * std_dev + 0.01;
        scale.y = 2.0 * std_dev + 0.01;
        scale.z = 0.01;
        ellipse.scale = scale;

        // 이 지점의 (x,y)
        ellipse.pose.position.x = state_seq(i, mppi::STATE_SPACE::x);
        ellipse.pose.position.y = state_seq(i, mppi::STATE_SPACE::y);
        ellipse.pose.position.z = 0.1;

        marker_array.markers.push_back(ellipse);
    }
    publisher.publish(marker_array);
}


void MPPIControllerROS::publish_state_seq_dists(
    const mppi::cpu::StateSeq& state_seq,
    const mppi::cpu::XYCovMatrices& cov_matrices,
    const ros::Publisher& publisher) const{
    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker ellipse;
    ellipse.header.frame_id = map_frame_id_;
    ellipse.header.stamp = ros::Time::now();
    ellipse.ns = "proposal_state_distributions";
    ellipse.type = visualization_msgs::Marker::SPHERE;
    ellipse.action = visualization_msgs::Marker::ADD;
    ellipse.color.a = 0.4;
    // blue color
    ellipse.color.r = 0.0;
    ellipse.color.g = 0.0;
    ellipse.color.b = 1.0;

    for (int i = 0; i < state_seq.rows(); i++) {
        ellipse.id = i;
        ellipse.pose.position.x = state_seq(i, mppi::STATE_SPACE::x);
        ellipse.pose.position.y = state_seq(i, mppi::STATE_SPACE::y);
        ellipse.pose.position.z = 0.1;

        // 2x2 공분산에 대한 고유값/벡터
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(
            cov_matrices.at(i).block<2,2>(0,0));
        const auto eigen_val = es.eigenvalues();
        const auto eigen_vec = es.eigenvectors();

        double major_len = std::sqrt(eigen_val(0));
        double minor_len = std::sqrt(eigen_val(1));
        double yaw = std::atan2(eigen_vec(1,0), eigen_vec(0,0));

        // marker scale
        ellipse.scale.x = 2.0*major_len + 0.1;
        ellipse.scale.y = 2.0*minor_len + 0.1;
        ellipse.scale.z = 0.1;

        // orientation
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, yaw);
        ellipse.pose.orientation = tf2::toMsg(q);

        marker_array.markers.push_back(ellipse);
    }
    publisher.publish(marker_array);
}


}  // namespace mppi