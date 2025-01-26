#include "mppi_controller/mpc_base.hpp"

#include <omp.h>            // 병렬 처리용
#include <algorithm>        // std::sort
#include <cmath>            // std::sin, std::cos 등
#include <iostream>
#include <ros/ros.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

namespace mppi {
namespace cpu {

    //-------------------------
    // Helper: 쿼터니언 -> yaw
    //-------------------------
    static double getYawFromQuaternion(const geometry_msgs::Quaternion& q)
    {
        tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
        tf2::Matrix3x3 mat(tf_q);
        double roll, pitch, yaw;
        mat.getRPY(roll, pitch, yaw);
        return yaw;
    }

    //-------------------------
    // Constructor
    //-------------------------

    // #### Public functions ####
    MPCBase::MPCBase(const Params::Common& params, const size_t& sample_num)
        : thread_num_(params.thread_num),
          prediction_step_size_(static_cast<size_t>(params.prediction_step_size)),
          prediction_interval_(params.prediction_interval),
        //   reference_speed_(params.reference_speed),
          q_dist_(params.q_dist),
          q_angle_(params.q_angle),
          // q_speed_(params.q_speed),
          collision_weight_(params.collision_weight),
          q_terminal_dist_(params.q_terminal_dist),
          q_terminal_angle_(params.q_terminal_angle)
    // q_terminal_speed_(params.q_terminal_speed),
    {
        // (sample_num)개의 trajectory 후보를 저장할 공간
        state_seq_candidates_.resize(sample_num);
        for (size_t i = 0; i < sample_num; i++) {
            state_seq_candidates_[i] =
                Eigen::MatrixXd::Zero(prediction_step_size_, STATE_SPACE::dim);
        }
    }

    void MPCBase::set_obstacle_map(const grid_map::GridMap& obstacle_map) { obstacle_map_ = obstacle_map; } // local cost map

    void MPCBase::set_reference_path(const nav_msgs::Path& ref_path) { reference_path_ = ref_path; }

    std::pair<std::vector<double>, std::vector<double>> MPCBase::calc_sample_costs(const PriorSamplesWithCosts& sampler, const cpu::State& init_state){
                return calc_sample_costs(sampler, init_state, &state_seq_candidates_);
            }

    std::tuple<cpu::StateSeq, double, double> MPCBase::get_predictive_seq(const cpu::State& initial_state, const cpu::ControlSeq& control_input_seq) const{
                // 단일 control seq에 대해 예측
                cpu::StateSeq seq = predict_state_seq(control_input_seq, initial_state);
                // cost 계산
                auto [total_cost, collision_cost] = state_cost(seq);
                double ref_cost = total_cost - collision_cost;
                return std::make_tuple(seq, ref_cost, collision_cost);
            }

    std::pair<std::vector<cpu::StateSeq>, std::vector<double>> MPCBase::get_state_seq_candidates(const int& _num_samples, const std::vector<double>& weights) const{
                if (weights.empty()) {
                    std::cerr << "[MPCBase] weights is empty." << std::endl;
                    return std::make_pair(std::vector<cpu::StateSeq>(), std::vector<double>());
                }

                const int num_samples = std::min(static_cast<int>(weights.size()), _num_samples);

                // 상위 num_samples 개를 고르기 위해 weight를 정렬
                std::vector<double> sorted_weights = weights;
                std::sort(sorted_weights.begin(), sorted_weights.end());

                std::vector<int> indices;
                indices.reserve(num_samples);

                // 가장 큰 weight부터 차례대로 인덱스 찾기
                for (int i = 0; i < num_samples; i++) {
                    double w_val = sorted_weights[sorted_weights.size() - 1 - i];
                    auto it = std::find(weights.begin(), weights.end(), w_val);
                    int idx = std::distance(weights.begin(), it);
                    indices.push_back(idx);
                }

                // 결과 할당
                std::vector<double> selected_weights(num_samples);
                std::vector<cpu::StateSeq> selected_seq(num_samples);
                for (int i = 0; i < num_samples; i++) {
                    selected_weights[i] = weights[indices[i]];
                    selected_seq[i]     = state_seq_candidates_[indices[i]];
                }
                return std::make_pair(selected_seq, selected_weights);
            }

    std::pair<cpu::StateSeq, cpu::XYCovMatrices> MPCBase::get_proposed_distribution() const{
                return calc_state_distribution(state_seq_candidates_);
            }


    // #### Private functions ####
    std::pair<std::vector<double>, std::vector<double>> MPCBase::calc_sample_costs(const PriorSamplesWithCosts& sampler, const cpu::State& init_state, cpu::StateSeqBatch* state_seq_candidates) const{
        
        const size_t N = sampler.get_num_samples();
        std::vector<double> costs(N, 0.0);
        std::vector<double> collision_costs(N, 0.0);

    #pragma omp parallel for num_threads(thread_num_)
        for (size_t i = 0; i < N; i++) {
            // Predict state sequence
            (*state_seq_candidates)[i] = predict_state_seq(
                sampler.noised_control_seq_samples_[i],
                init_state
            );

            // Calculate costs
            auto [total_cost, coll_cost] = state_cost((*state_seq_candidates)[i]);
            costs[i] = total_cost;
            collision_costs[i] = coll_cost;

            // Individual sample logging
            ROS_INFO_THROTTLE(1.0, "Sample %zu: Total Cost=%.2f, Collision Cost=%.2f", i, total_cost, coll_cost);
        }
       
        return std::make_pair(costs, collision_costs);
    }

            /**
             * @brief  홀로노믹 모델 예시 (간단 버전)
             *         - State = [x, y, yaw, Vx, Vy, w] 
             *         - Control= [Vx_cmd, Vy_cmd, w_cmd]
             *
             *         여기서는 "명령 속도를 곧바로 state에 대입"하는 식으로 구현 (가속도/지연은 고려X).
             */

    cpu::StateSeq MPCBase::predict_state_seq(const cpu::ControlSeq& control_seq, const cpu::State& init_state) const{
       
        cpu::StateSeq seq = Eigen::MatrixXd::Zero(prediction_step_size_, STATE_SPACE::dim);
        // Initial state
        seq.row(0) = init_state.transpose();

        for (size_t i = 0; i < prediction_step_size_ - 1; i++) {
            double x   = seq(i, STATE_SPACE::x);
            double y   = seq(i, STATE_SPACE::y);
            double yaw = seq(i, STATE_SPACE::yaw);

            // Control inputs
            double Vx_cmd = control_seq(i, CONTROL_SPACE::Vx);
            double Vy_cmd = control_seq(i, CONTROL_SPACE::Vy);
            double w_cmd  = control_seq(i, CONTROL_SPACE::w);

            // State update (simple holonomic model)
            double x_new = x + Vx_cmd * prediction_interval_;
            double y_new = y + Vy_cmd * prediction_interval_;
            double yaw_new = yaw + w_cmd * prediction_interval_;

            seq(i+1, STATE_SPACE::x) = x_new;
            seq(i+1, STATE_SPACE::y) = y_new;
            seq(i+1, STATE_SPACE::yaw) = std::atan2(std::sin(yaw_new), std::cos(yaw_new));

            // Individual step logging
            ROS_INFO_THROTTLE(1.0, "Predict step %zu: x=%.2f, y=%.2f, yaw=%.2f", i, x_new, y_new, yaw_new);
        }
        
        return seq;
    }

            /**
             * @brief state_cost
             *  - reference_path_와의 거리/각도 오차 계산
             *  - obstacle_map_에서 충돌비용 계산
             */
        //-------------------------------------
        // state_cost : trajectory 전체 cost
        //   - (1) 경로와 거리/방향 오차
        //   - (2) 장애물 map 충돌
        //   - (3) 맨 끝 스텝에 대한 terminal cost
        //-------------------------------------

    std::pair<double, double> MPCBase::state_cost(const cpu::StateSeq& state_seq) const{
                double sum_cost = 0.0;
                double sum_collision_cost = 0.0;

                const size_t N = state_seq.rows();
                if (N == 0) {
                    ROS_WARN_THROTTLE(2.0, "State sequence is empty.");
                    return {0.0, 0.0};
                }

                //-------------------------------------
                // 1) 중간 step (0 ~ N-2)에 대한 일반 cost
                //-------------------------------------

                for (size_t i = 0; i < N - 1; i++) {
                double x   = state_seq(i, STATE_SPACE::x);
                double y   = state_seq(i, STATE_SPACE::y);
                double yaw = state_seq(i, STATE_SPACE::yaw);

                // (1) path distance & angle cost
                double min_dist = 1e9;
                double path_yaw = 0.0; // 실제로는 pose.orientation에서 yaw를 구해서 넣음
                if (!reference_path_.poses.empty()) {
                    for (auto &pose : reference_path_.poses) {
                        double dx = pose.pose.position.x - x;
                        double dy = pose.pose.position.y - y;
                        double dist = std::hypot(dx, dy);
                        // ROS_INFO_THROTTLE(1.0 ,"Check dist: dx=%.3f, dy=%.3f => dist=%.3f", dx,dy,dist);
                        if (dist < min_dist) {
                            min_dist = dist;
                            path_yaw = getYawFromQuaternion(pose.pose.orientation);
                        }
                    }
                }
                else {
                    ROS_WARN_THROTTLE(5.0, "Reference path is empty.");
                }
                double diff_yaw = yaw - path_yaw;
                diff_yaw = std::atan2(std::sin(diff_yaw), std::cos(diff_yaw));

                sum_cost += q_dist_  * (min_dist * min_dist);
                sum_cost += q_angle_ * (diff_yaw * diff_yaw);

                // (2) obstacle map 충돌 비용
                double collision_val = 10.0; // 임의 기본값
                if (obstacle_map_.exists("collision_layer")) {
                    grid_map::Position pos(x, y);
                    if (obstacle_map_.isInside(pos)) {
                        collision_val =
                            obstacle_map_.atPosition("collision_layer", pos);
                    }
                }
                sum_collision_cost += collision_val * collision_weight_;
                // 상세 로그 추가
                //  ROS_INFO_THROTTLE(2.0, "Step %zu: x=%.2f, y=%.2f, yaw=%.2f, min_dist=%.2f, diff_yaw=%.2f, collision_val=%.2f",
                //   i, x, y, yaw, min_dist, diff_yaw, collision_val);
            }

            //-------------------------------------
            // 2) 마지막 스텝(N-1)에 대한 terminal cost
            //-------------------------------------
            {
                double x_f   = state_seq(N-1, STATE_SPACE::x);
                double y_f   = state_seq(N-1, STATE_SPACE::y);
                double yaw_f = state_seq(N-1, STATE_SPACE::yaw);

                double min_dist_f = 1e9;
                double path_yaw_f = 0.0;
                if (!reference_path_.poses.empty()) {
                    for (auto &pose : reference_path_.poses) {
                        double dx = pose.pose.position.x - x_f;
                        double dy = pose.pose.position.y - y_f;
                        double dist = std::hypot(dx, dy);
                        if (dist < min_dist_f) {
                            min_dist_f = dist;
                            path_yaw_f = getYawFromQuaternion(pose.pose.orientation);
                        }
                    }
                }

                double diff_yaw_f = yaw_f - path_yaw_f;
                diff_yaw_f = std::atan2(std::sin(diff_yaw_f), std::cos(diff_yaw_f));

                // terminal cost 가중치 사용
                sum_cost += q_terminal_dist_  * (min_dist_f * min_dist_f);
                sum_cost += q_terminal_angle_ * (diff_yaw_f * diff_yaw_f);

                // 장애물 비용
                double collision_val_f = 10.0;
                if (obstacle_map_.exists("collision_layer")) {
                    grid_map::Position posf(x_f, y_f);
                    if (obstacle_map_.isInside(posf)) {
                        collision_val_f =
                            obstacle_map_.atPosition("collision_layer", posf);
                    }
                }
                sum_collision_cost += collision_val_f * collision_weight_;
                
            }

            double total_cost = sum_cost + sum_collision_cost;
            // 최종 비용 로그
            // ROS_INFO_THROTTLE(2.0, "Total Cost: %.2f, Collision Cost: %.2f", sum_cost, sum_collision_cost);
            return std::make_pair(total_cost, sum_collision_cost);
        }

        //-------------------------------------
        // calc_state_distribution
        //-------------------------------------

    std::pair<cpu::StateSeq, cpu::XYCovMatrices> MPCBase::calc_state_distribution(const cpu::StateSeqBatch& state_seq_candidates) const{

        const size_t n_candidates = state_seq_candidates.size();
        if (n_candidates == 0) {
            ROS_WARN_THROTTLE(1.0, "MPCBase::calc_state_distribution: state_seq_candidates is empty");
            return {cpu::StateSeq(), {}};
        }

                const size_t T = state_seq_candidates[0].rows();

        // Calculate mean
        cpu::StateSeq mean_seq = Eigen::MatrixXd::Zero(T, STATE_SPACE::dim);
        for (auto &seq : state_seq_candidates) {
            mean_seq += seq;
        }
        mean_seq /= double(n_candidates);
        

        // Calculate covariance
        cpu::XYCovMatrices xy_cov_mats(T, Eigen::Matrix2d::Zero());

    #pragma omp parallel for num_threads(thread_num_)
        for (size_t i = 0; i < n_candidates; i++) {
            for (size_t t = 0; t < T; t++) {
                double dx = state_seq_candidates[i](t, STATE_SPACE::x) - mean_seq(t, STATE_SPACE::x);
                double dy = state_seq_candidates[i](t, STATE_SPACE::y) - mean_seq(t, STATE_SPACE::y);

                Eigen::Vector2d diff(dx, dy);
                Eigen::Matrix2d cov = diff * diff.transpose();

    #pragma omp critical
                {
                    xy_cov_mats[t] += cov;
                }
            }
        }

        for (size_t t = 0; t < T; t++) {
            xy_cov_mats[t] /= double(n_candidates);
        }


        
        return std::make_pair(mean_seq, xy_cov_mats);
    }

}  // namespace cpu
}  // namespace mppi