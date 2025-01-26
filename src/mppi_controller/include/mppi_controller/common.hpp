#pragma once

#include <Eigen/Core>
#include <string>

namespace mppi {
namespace STATE_SPACE {
    static constexpr int x = 0;
    static constexpr int y = 1;
    static constexpr int yaw = 2;
    static constexpr int dim = 3;
};  // namespace STATE_SPACE

namespace CONTROL_SPACE {
    static constexpr int Vx = 0;
    static constexpr int Vy = 1;
    static constexpr int w = 2;
    
    static constexpr int dim = 3;
};  // namespace CONTROL_SPACE

struct Params {
    struct Common {
        int thread_num;
        int prediction_step_size;
        double prediction_interval;
        double max_Vx;
        double min_Vx;
        double max_Vy;
        double min_Vy;
        double max_w;
        double min_w;
        // std::string speed_prediction_mode;
        double q_dist;
        double q_angle;
        // double q_speed;
        double collision_weight;
        double q_terminal_dist;
        double q_terminal_angle;
        // double q_terminal_speed;
    };
    Common common;

    struct ForwardMPPI {
        int sample_batch_num;
        double lambda;
        double alpha;
        double non_biased_sampling_rate;
        double Vx_cov;
        double Vy_cov;
        double w_cov;
        int sample_num_for_grad_estimation;
        double Vx_cov_for_grad_estimation;
        double Vy_cov_for_grad_estimation;
        double w_cov_for_grad_estimation;
        int num_itr_for_grad_estimation;
        double step_size_for_grad_estimation;
    };
    ForwardMPPI forward_mppi;

    struct ReverseMPPI {
        int sample_batch_num;
        double negative_ratio;
        bool is_sample_rejection;
        double sample_inflation_ratio;
        double warm_start_ratio;
        int iteration_num;
        double step_size;
        double lambda;
        double alpha;
        double non_biased_sampling_rate;
        double Vx_cov;
        double Vy_cov;
        double w_cov;
    };
    ReverseMPPI reverse_mppi;

    struct SteinVariationalMPC {
        int sample_batch_num;
        double lambda;
        double alpha;
        double non_biased_sampling_rate;
        double Vx_cov;
        double Vy_cov;
        double w_cov;
        int num_svgd_iteration;
        int sample_num_for_grad_estimation;
        double Vx_cov_for_grad_estimation;
        double Vy_cov_for_grad_estimation;
        double w_cov_for_grad_estimation;
        double svgd_step_size;
        bool is_max_posterior_estimation;
    };
    SteinVariationalMPC stein_variational_mpc;

    struct SVGuidedMPPI {
        int sample_batch_num;
        double lambda;
        double alpha;
        double non_biased_sampling_rate;
        double Vx_cov;
        double Vy_cov;
        double w_cov;
        int guide_sample_num;
        double grad_lambda;
        int sample_num_for_grad_estimation;
        double Vx_cov_for_grad_estimation;
        double Vy_cov_for_grad_estimation;
        double w_cov_for_grad_estimation;
        double svgd_step_size;
        int num_svgd_iteration;
        bool is_use_nominal_solution;
        bool is_covariance_adaptation;
        double gaussian_fitting_lambda;
        double min_Vx_cov;
        double max_Vx_cov;
        double min_Vy_cov;
        double max_Vy_cov;
        double min_w_cov;
        double max_w_cov;
    };
    SVGuidedMPPI svg_mppi;
};

namespace cpu {
    using State = Eigen::Matrix<double, STATE_SPACE::dim, 1>;
    using Control = Eigen::Matrix<double, CONTROL_SPACE::dim, 1>;
    using StateSeq = Eigen::MatrixXd;
    using ControlSeq = Eigen::MatrixXd;
    using StateSeqBatch = std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>;
    
    using ControlSeqBatch = std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>;
    using ControlSeqCovMatrices = std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>;
    // 공분산 행렬 저장용
    using XYCovMatrices = std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>;
}  // namespace cpu

}  // namespace mppi
