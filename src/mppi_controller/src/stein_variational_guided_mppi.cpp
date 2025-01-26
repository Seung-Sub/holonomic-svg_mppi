#include "mppi_controller/stein_variational_guided_mppi.hpp"
#include <ros/ros.h>

namespace mppi {
namespace cpu {

////////////////////////////////////////////////////////////
// Constructor
////////////////////////////////////////////////////////////
SVGuidedMPPI::SVGuidedMPPI(const Params::Common& common_params,
                           const Params::SVGuidedMPPI& svg_mppi_params)
    : prediction_step_size_(static_cast<size_t>(common_params.prediction_step_size)),
      thread_num_(common_params.thread_num),
      lambda_(svg_mppi_params.lambda),
      alpha_(svg_mppi_params.alpha),
      non_biased_sampling_rate_(svg_mppi_params.non_biased_sampling_rate),

      // 3D control cov
      Vx_cov_(svg_mppi_params.Vx_cov),
      Vy_cov_(svg_mppi_params.Vy_cov),
      w_cov_(svg_mppi_params.w_cov),

      sample_num_for_grad_estimation_(svg_mppi_params.sample_num_for_grad_estimation),
      grad_lambda_(svg_mppi_params.grad_lambda),
      Vx_cov_for_grad_estimation_(svg_mppi_params.Vx_cov_for_grad_estimation),
      Vy_cov_for_grad_estimation_(svg_mppi_params.Vy_cov_for_grad_estimation),
      w_cov_for_grad_estimation_(svg_mppi_params.w_cov_for_grad_estimation),
      svgd_step_size_(svg_mppi_params.svgd_step_size),
      num_svgd_iteration_(svg_mppi_params.num_svgd_iteration),
      is_use_nominal_solution_(svg_mppi_params.is_use_nominal_solution),
      is_covariance_adaptation_(svg_mppi_params.is_covariance_adaptation),
      gaussian_fitting_lambda_(svg_mppi_params.gaussian_fitting_lambda),

      min_Vx_cov_(svg_mppi_params.min_Vx_cov),
      max_Vx_cov_(svg_mppi_params.max_Vx_cov),
      min_Vy_cov_(svg_mppi_params.min_Vy_cov),
      max_Vy_cov_(svg_mppi_params.max_Vy_cov),
      min_w_cov_(svg_mppi_params.min_w_cov),
      max_w_cov_(svg_mppi_params.max_w_cov)
{
    const size_t sample_batch_num = static_cast<size_t>(svg_mppi_params.sample_batch_num);
    const size_t guide_sample_num = static_cast<size_t>(svg_mppi_params.guide_sample_num);
    const size_t sn_grad = sample_num_for_grad_estimation_;

    // For caching
    const size_t sample_num_for_cache =
        std::max({sample_batch_num, sn_grad, guide_sample_num});

    // create MPCBase
    mpc_base_ptr_ = std::make_unique<MPCBase>(common_params, sample_num_for_cache);

    // setup prior & guide
    std::array<double, CONTROL_SPACE::dim> max_ctrl = {
        common_params.max_Vx, common_params.max_Vy, common_params.max_w
    };
    std::array<double, CONTROL_SPACE::dim> min_ctrl = {
        common_params.min_Vx, common_params.min_Vy, common_params.min_w
    };

    prior_samples_ptr_ = std::make_unique<PriorSamplesWithCosts>(
        sample_batch_num,
        prediction_step_size_,
        max_ctrl,
        min_ctrl,
        non_biased_sampling_rate_,
        thread_num_
    );

    guide_samples_ptr_ = std::make_unique<PriorSamplesWithCosts>(
        guide_sample_num,
        prediction_step_size_,
        max_ctrl,
        min_ctrl,
        non_biased_sampling_rate_,
        thread_num_
    );

    // init seq
    prev_control_seq_    = prior_samples_ptr_->get_zero_control_seq();
    nominal_control_seq_ = prior_samples_ptr_->get_zero_control_seq();

    // guide init
    std::array<double, CONTROL_SPACE::dim> guide_cov_arr = {
        Vx_cov_, Vy_cov_, w_cov_
    };
    auto guide_cov = guide_samples_ptr_->get_constant_control_seq_cov_matrices(guide_cov_arr);
    guide_samples_ptr_->random_sampling(guide_samples_ptr_->get_zero_control_seq(), guide_cov);

    // grad samplers
    for (size_t i = 0; i < sample_batch_num; i++) {
        grad_sampler_ptrs_.emplace_back(
            std::make_unique<PriorSamplesWithCosts>(
                sn_grad,
                prediction_step_size_,
                max_ctrl,
                min_ctrl,
                non_biased_sampling_rate_,
                thread_num_,
                i
            )
        );
    }
}

////////////////////////////////////////////////////////////
// solve
////////////////////////////////////////////////////////////
std::pair<ControlSeq, double>
SVGuidedMPPI::solve(const State& initial_state)
{   
    ROS_INFO_THROTTLE(5.0, "Starting SVGuidedMPPI solve process");
    // 샘플 비용 계산
    auto func_calc_costs = [&](const PriorSamplesWithCosts& sampler){
        return mpc_base_ptr_->calc_sample_costs(sampler, initial_state).first;
    };

    // ==== SVGD on guide ====
    for (int iter = 0; iter < num_svgd_iteration_; iter++) {
        ROS_INFO_THROTTLE(2.0, "SVGD Iteration: %d", iter);
        ControlSeqBatch grad_batch = approx_grad_posterior_batch(*guide_samples_ptr_, func_calc_costs);

#pragma omp parallel for num_threads(thread_num_)
        for (size_t i = 0; i < guide_samples_ptr_->get_num_samples(); i++) {
            guide_samples_ptr_->noised_control_seq_samples_[i] +=
                svgd_step_size_ * grad_batch[i];
        }
        // 추가: 각 반복마다 샘플의 평균과 공분산 로그 출력
        if (iter % 10 == 0) {
            ROS_INFO_THROTTLE(3.0, "Iteration %d complete.", iter);
        }
    }

    // pick best from guide
    auto guide_costs = mpc_base_ptr_->calc_sample_costs(*guide_samples_ptr_, initial_state).first;
    size_t min_idx = std::distance(
        guide_costs.begin(),
        std::min_element(guide_costs.begin(), guide_costs.end())
    );
    ControlSeq best_particle = guide_samples_ptr_->noised_control_seq_samples_[min_idx];
    ROS_INFO_THROTTLE(5.0, "Best particle index: %zu with cost: %f", min_idx, guide_costs[min_idx]);

    // prior Cov
    std::array<double, CONTROL_SPACE::dim> init_cov_arr = { Vx_cov_, Vy_cov_, w_cov_ };
    auto prior_cov = prior_samples_ptr_->get_constant_control_seq_cov_matrices(init_cov_arr);

    // if adapt => do dimension-wise 1D fitting
    if (is_covariance_adaptation_) {
        auto guide_costs_copy = guide_costs; // or gather from multiple iteration if wanted
        auto softmax_costs = softmax(guide_costs_copy, gaussian_fitting_lambda_, thread_num_);

        // dimension wise
        auto& guide_samps = guide_samples_ptr_->noised_control_seq_samples_;
        for (size_t step_i = 0; step_i < prediction_step_size_ - 1; step_i++) {
            std::vector<double> vx_vals(guide_samps.size()),
                                vy_vals(guide_samps.size()),
                                w_vals(guide_samps.size());
            for (size_t k = 0; k < guide_samps.size(); k++) {
                vx_vals[k] = guide_samps[k](step_i, CONTROL_SPACE::Vx);
                vy_vals[k] = guide_samps[k](step_i, CONTROL_SPACE::Vy);
                w_vals[k]  = guide_samps[k](step_i, CONTROL_SPACE::w);
            }
            double sigma_vx = dimension_gaussian_fitting(vx_vals, softmax_costs,
                                                         min_Vx_cov_, max_Vx_cov_);
            double sigma_vy = dimension_gaussian_fitting(vy_vals, softmax_costs,
                                                         min_Vy_cov_, max_Vy_cov_);
            double sigma_w  = dimension_gaussian_fitting(w_vals,  softmax_costs,
                                                         min_w_cov_, max_w_cov_);

            Eigen::Matrix3d step_cov = Eigen::Matrix3d::Identity();
            step_cov(CONTROL_SPACE::Vx, CONTROL_SPACE::Vx) = sigma_vx;
            step_cov(CONTROL_SPACE::Vy, CONTROL_SPACE::Vy) = sigma_vy;
            step_cov(CONTROL_SPACE::w,  CONTROL_SPACE::w)  = sigma_w;

            prior_cov[step_i] = step_cov;
        }
    }

    // prior sampling
    prior_samples_ptr_->random_sampling(prev_control_seq_, prior_cov);

    // rollout
    auto [costs, collision_costs] = mpc_base_ptr_->calc_sample_costs(*prior_samples_ptr_, initial_state);
    prior_samples_ptr_->costs_ = costs;

    // nominal
    if (is_use_nominal_solution_) {
        nominal_control_seq_ = best_particle;
    } else {
        nominal_control_seq_ = prior_samples_ptr_->get_zero_control_seq();
    }

    // weights => softmax
    auto weights = calc_weights(*prior_samples_ptr_, nominal_control_seq_);
    weights_ = weights;

    // weighted average
    ControlSeq updated_control_seq = prior_samples_ptr_->get_zero_control_seq();
    for (size_t i = 0; i < prior_samples_ptr_->get_num_samples(); i++) {
        updated_control_seq += weights[i] * prior_samples_ptr_->noised_control_seq_samples_[i];
    }

    // collision
    int coll_count = 0;
    for (auto c : collision_costs) {
        if (c > 0.0) coll_count++;
    }
    double collision_rate = double(coll_count) / prior_samples_ptr_->get_num_samples();

    // update
    prev_control_seq_ = updated_control_seq;

    // 반환 전 로그 추가
    ROS_INFO_THROTTLE(5.0, "Solver finished with collision rate: %.2f", collision_rate);
    return {updated_control_seq, collision_rate};
}

////////////////////////////////////////////////////////////
// set maps & path
////////////////////////////////////////////////////////////
void SVGuidedMPPI::set_obstacle_map(const grid_map::GridMap& obstacle_map)
{
    mpc_base_ptr_->set_obstacle_map(obstacle_map);
}

void SVGuidedMPPI::set_reference_path(const nav_msgs::Path& reference_path)
{
    mpc_base_ptr_->set_reference_path(reference_path);
}

////////////////////////////////////////////////////////////
// For visualization
////////////////////////////////////////////////////////////
std::pair<std::vector<StateSeq>, std::vector<double>>
SVGuidedMPPI::get_state_seq_candidates(const int& num_samples) const
{
    return mpc_base_ptr_->get_state_seq_candidates(num_samples, weights_);
}

std::tuple<StateSeq, double, double, double>
SVGuidedMPPI::get_predictive_seq(const State& initial_state,
                                 const ControlSeq& control_input_seq) const
{
    auto [pred_state, cost, coll] =
        mpc_base_ptr_->get_predictive_seq(initial_state, control_input_seq);

    double input_error = 0.0;
    for (size_t i = 0; i < prediction_step_size_ - 1; i++) {
        input_error += control_input_seq.row(i).norm();
    }
    return { pred_state, cost, coll, input_error };
}

ControlSeqCovMatrices SVGuidedMPPI::get_cov_matrices() const
{
    return prior_samples_ptr_->get_cov_matrices();
}

ControlSeq SVGuidedMPPI::get_control_seq() const
{
    return nominal_control_seq_;
}

std::pair<StateSeq, XYCovMatrices>
SVGuidedMPPI::get_proposed_state_distribution() const
{
    return mpc_base_ptr_->get_proposed_distribution();
}

////////////////////////////////////////////////////////////
// Private: approx_grad_log_likelihood
////////////////////////////////////////////////////////////
ControlSeq
SVGuidedMPPI::approx_grad_log_likelihood(
    const ControlSeq& mean_seq,
    const ControlSeq& noised_seq,
    const ControlSeqCovMatrices& inv_covs,
    const std::function<std::vector<double>(const PriorSamplesWithCosts&)>& calc_costs,
    PriorSamplesWithCosts* sampler
) const
{
    // use 3D for grad estimation
    std::array<double, CONTROL_SPACE::dim> grad_cov_arr = {
        Vx_cov_for_grad_estimation_,
        Vy_cov_for_grad_estimation_,
        w_cov_for_grad_estimation_
    };
    auto grad_cov = sampler->get_constant_control_seq_cov_matrices(grad_cov_arr);

    // random sampling
    sampler->random_sampling(noised_seq, grad_cov);
    sampler->costs_ = calc_costs(*sampler);

    std::vector<double> exp_costs(sampler->get_num_samples());
    ControlSeq sum_of_grads = mean_seq*0.0;

#pragma omp parallel for num_threads(thread_num_)
    for (size_t i = 0; i < sampler->get_num_samples(); i++) {
        double c_val = sampler->costs_[i];
        double exp_val = std::exp(-c_val / grad_lambda_);
        exp_costs[i] = exp_val;

#pragma omp critical
        {
            for (size_t j = 0; j < prediction_step_size_ - 1; j++) {
                Eigen::RowVectorXd diff =
                    sampler->noised_control_seq_samples_[i].row(j)
                    - noised_seq.row(j);
                // multiply by inv_covs[j]
                sum_of_grads.row(j) += exp_val * (inv_covs[j] * diff.transpose()).transpose();
            }
        }
    }

    double sum_exp = std::accumulate(exp_costs.begin(), exp_costs.end(), 0.0);
    return sum_of_grads / (sum_exp + 1e-10);
}

////////////////////////////////////////////////////////////
// Private: approx_grad_posterior_batch
////////////////////////////////////////////////////////////
ControlSeqBatch
SVGuidedMPPI::approx_grad_posterior_batch(
    const PriorSamplesWithCosts& samples,
    const std::function<std::vector<double>(const PriorSamplesWithCosts&)>& calc_costs
) const
{
    auto inv_covs = samples.get_inv_cov_matrices();
    auto mean_seq = samples.get_mean();
    ControlSeqBatch grad_batch = samples.get_zero_control_seq_batch();

#pragma omp parallel for num_threads(thread_num_)
    for (size_t i = 0; i < samples.get_num_samples(); i++) {
        grad_batch[i] =
            approx_grad_log_likelihood(mean_seq,
                                       samples.noised_control_seq_samples_[i],
                                       inv_covs,
                                       calc_costs,
                                       grad_sampler_ptrs_[i].get());
    }
    return grad_batch;
}

////////////////////////////////////////////////////////////
// 1D Gaussian fitting: Gao's
////////////////////////////////////////////////////////////
std::pair<double, double>
SVGuidedMPPI::gaussian_fitting(const std::vector<double>& x,
                               const std::vector<double>& y) const
{
    // same as original code
    if (x.size() != y.size()) {
        // error handle
        return {0.0, 1.0};
    }
    std::vector<double> y_hat(y.size());
    std::transform(y.begin(), y.end(), y_hat.begin(),
                   [](double val){ return std::max(val, 1e-10); });

    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();

    for (size_t i = 0; i < x.size(); i++) {
        double x_i = x[i];
        double yy  = y_hat[i];
        double yy2 = yy*yy;
        double logy= std::log(yy);

        A(0,0)+=yy2;
        A(0,1)+=yy2*x_i;
        A(0,2)+=yy2*x_i*x_i;

        A(1,0)+=yy2*x_i;
        A(1,1)+=yy2*x_i*x_i;
        A(1,2)+=yy2*x_i*x_i*x_i;

        A(2,0)+=yy2*x_i*x_i;
        A(2,1)+=yy2*x_i*x_i*x_i;
        A(2,2)+=yy2*x_i*x_i*x_i*x_i;

        b(0)+=yy2*logy;
        b(1)+=yy2*x_i*logy;
        b(2)+=yy2*x_i*x_i*logy;
    }

    auto u = A.colPivHouseholderQr().solve(b);

    double eps = 1e-5;
    double mean = -u(1)/(2.0*std::min(u(2), -eps));
    double var  = std::sqrt(1.0/(2.0*std::abs(u(2))));

    return {mean, var};
}

////////////////////////////////////////////////////////////
// dimension_gaussian_fitting
////////////////////////////////////////////////////////////
double SVGuidedMPPI::dimension_gaussian_fitting(const std::vector<double>& ctrl_vals,
                                                const std::vector<double>& weights,
                                                double min_cov,
                                                double max_cov) const
{
    // normalize weights if needed, or pass directly
    auto [m, var] = gaussian_fitting(ctrl_vals, weights);
    double var_clamp = std::clamp(var, min_cov, max_cov);
    return var_clamp;
}

////////////////////////////////////////////////////////////
// Weighted mean & sigma (if needed)
////////////////////////////////////////////////////////////
std::pair<ControlSeq, ControlSeqCovMatrices>
SVGuidedMPPI::weighted_mean_and_sigma(const PriorSamplesWithCosts& samples,
                                      const std::vector<double>& weights) const
{
    // (기존 Ackermann 방식 유지, or partial implementation)
    ControlSeq mean = samples.get_zero_control_seq();
    auto sigma = samples.get_zero_control_seq_cov_matrices();

    // ...
    return {mean, sigma};
}

std::pair<ControlSeq, ControlSeqCovMatrices>
SVGuidedMPPI::estimate_mu_and_sigma(const PriorSamplesWithCosts& samples) const
{
    // ...
    return {ControlSeq(), ControlSeqCovMatrices()};
}

////////////////////////////////////////////////////////////
// calc_weights
////////////////////////////////////////////////////////////
std::vector<double>
SVGuidedMPPI::calc_weights(const PriorSamplesWithCosts& prior_samples_with_costs,
                           const ControlSeq& nominal_control_seq) const
{
    auto cost_with_ctrl_term =
        prior_samples_with_costs.get_costs_with_control_term(lambda_, alpha_, nominal_control_seq);

    return softmax(cost_with_ctrl_term, lambda_, thread_num_);
}

} // namespace cpu
} // namespace mppi

