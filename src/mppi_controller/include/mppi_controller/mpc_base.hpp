// Kohei Honda, 2023

#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <array>
#include <grid_map_core/GridMap.hpp> // for local cost map
#include <nav_msgs/Path.h> // for reference path
#include <nav_msgs/Odometry.h> // for robot state
#include <memory>
#include <utility>

#include "mppi_controller/common.hpp"
#include "mppi_controller/prior_samples_with_costs.hpp"

namespace mppi {
namespace cpu {
    class MPCBase {
    public:
        MPCBase(const Params::Common& params, const size_t& sample_num);
        ~MPCBase() = default;

        void set_obstacle_map(const grid_map::GridMap& obstacle_map); // local cost map 

        void set_reference_path(const nav_msgs::Path& ref_path); // for reference path

        std::pair<std::vector<double>, std::vector<double>> calc_sample_costs(const PriorSamplesWithCosts& sampler, const cpu::State& init_state);

        std::tuple<StateSeq, double, double> get_predictive_seq(const cpu::State& initial_state, const cpu::ControlSeq& control_input_seq) const;

        std::pair<std::vector<StateSeq>, std::vector<double>> get_state_seq_candidates(const int& num_samples,
                                                                                       const std::vector<double>& weights) const;

        std::pair<cpu::StateSeq, cpu::XYCovMatrices> get_proposed_distribution() const;

    private:
        std::pair<std::vector<double>, std::vector<double>> calc_sample_costs(const PriorSamplesWithCosts& sampler, const cpu::State& init_state, cpu::StateSeqBatch* state_seq_candidates) const;

        // 홀로노믹 모델을 이용해 미래 state를 예측
        cpu::StateSeq predict_state_seq(const cpu::ControlSeq& control_seq,
                                        const cpu::State& init_state) const;


        // 단일 trajectory의 cost를 계산
        //  - obstacle_map으로부터 충돌 비용
        //  - reference_path로부터 거리, 방향 cost
        std::pair<double, double> state_cost(const cpu::StateSeq& state_seq) const;

        std::pair<cpu::StateSeq, cpu::XYCovMatrices> calc_state_distribution(const cpu::StateSeqBatch& state_seq_candidates) const;

    private:
        // == Constant parameters ==
        // const std::string obstacle_layer_name_ = "collision_layer"; // local cost map
        // 파라미터
        const int thread_num_;  //!< @brief number of thread for parallel computation
        const size_t prediction_step_size_;
        const double prediction_interval_;  //!< @brief prediction interval [s]
        
        // cost 가중치
        const double q_dist_;
        const double q_angle_;
        const double collision_weight_;
        const double q_terminal_dist_;
        const double q_terminal_angle_;

        // == Inner-variables ==
        grid_map::GridMap obstacle_map_; // local cost map
        nav_msgs::Path reference_path_; // reference path

        // rollout 시퀀스 임시 저장
        cpu::StateSeqBatch state_seq_candidates_;
    };

}  // namespace cpu
}  // namespace mppi
