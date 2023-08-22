#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "Miller.h"

template <typename Scalar> struct MeasurementModel {
  MeasurementModel(float dx, float dy) {
    H.setZero();
    H(0, 0) = 1;
    H(1, 2) = 1;

    R.setZero();
    R(0, 0) = dx;
    R(1, 1) = dy;
  }

  Eigen::Matrix<Scalar, 2, 4> H;
  Eigen::Matrix<Scalar, 2, 2> R;
};

template <typename Scalar> struct TransitionModel {
  Scalar noise_diff_coeff_x;
  Scalar noise_diff_coeff_y;

  Eigen::Matrix<Scalar, 4, 4> F(const float time_interval) {
    Eigen::Matrix<Scalar, 4, 4> F;
    F.setZero();

    F(0, 0) = 1;
    F(0, 1) = time_interval;
    F(1, 1) = 1;

    F(2, 2) = 1;
    F(2, 3) = time_interval;
    F(3, 3) = 1;

    return F;
  }

  Eigen::Matrix<Scalar, 4, 4> Q(const float time_interval) {
    Eigen::Matrix<Scalar, 4, 4> Q;
    Q.setZero();

    auto const time_interval_pow_3 = std::pow(time_interval, 3.f);
    auto const time_interval_pow_2 = std::pow(time_interval, 2.f);

    auto const time_interval_pow_3_div_3 = time_interval_pow_3 / 3.f;
    auto const time_interval_pow_2_div_2 = time_interval_pow_2 / 2.f;

    Q(0, 0) = time_interval_pow_3_div_3 * noise_diff_coeff_x;
    Q(0, 1) = time_interval_pow_2_div_2 * noise_diff_coeff_x;
    Q(1, 0) = time_interval_pow_2_div_2 * noise_diff_coeff_x;
    Q(1, 1) = time_interval * noise_diff_coeff_x;

    Q(2, 2) = time_interval_pow_3_div_3 * noise_diff_coeff_y;
    Q(2, 3) = time_interval_pow_2_div_2 * noise_diff_coeff_y;

    Q(3, 2) = time_interval_pow_2_div_2 * noise_diff_coeff_y;
    Q(3, 3) = time_interval * noise_diff_coeff_y;

    return Q;
  }
};

struct Track {
  uint32_t id;
  uint64_t start_time;
  std::vector<std::tuple<int, uint64_t>> zs;

  Track Append(int index, uint64_t time) const {
    auto track = *this;
    track.zs.emplace_back(index, time);
    return track;
  }
};

template <typename Scalar, size_t Size> struct MultivariateGaussian {
  MultivariateGaussian() = default;

  template<typename Mean, typename Covariance>
  MultivariateGaussian(Mean&& mean, Covariance&& covariance):
    mean{std::forward<Mean>(mean)},
    covariance{std::forward<Covariance>(covariance)} { }


  Eigen::Matrix<Scalar, Size, 1> mean;
  Eigen::Matrix<Scalar, Size, Size> covariance;
};

template <typename Scalar, size_t Size> struct Bernoulli {
  Bernoulli() = default;

  template <typename Prob, typename Distribution, typename Track>
  Bernoulli(Prob &&prob, Distribution &&distribution, Track &&track):
      prob{std::forward<Prob>(prob)},
      distribution{std::forward<Distribution>(distribution)},
      track{std::forward<Track>(track)} {}

  Scalar prob;
  MultivariateGaussian<Scalar, Size> distribution;
  Track track;
};

template <typename Scalar, size_t Size>
using MultiBernoulli = std::vector<Bernoulli<Scalar, Size>>;


template <typename Scalar, size_t Size> struct MultiBernoulliMixture {
  std::vector<Scalar> probs;
  std::vector<MultiBernoulli<Scalar, Size>> multi_bernoullis;
};

template <typename Scalar, size_t Size> struct PoissonPointProcess {
  std::vector<Scalar> weights;
  std::vector<MultivariateGaussian<Scalar, Size>> distributions;
};

template <typename Scalar, size_t Size> struct State {
  MultiBernoulliMixture<Scalar, Size> multi_bernoulli_mixture;
  PoissonPointProcess<Scalar, Size> poisson_point_process;
};

template<typename Scalar, size_t MeasurementSize>
using Measurement = Eigen::Matrix<Scalar, MeasurementSize, 1>;

template<typename Scalar, size_t MeasurementSize>
using Measurements = std::vector<Measurement<Scalar, MeasurementSize>>;

template <typename Scalar, size_t StateSize, size_t MeasurementSize>
struct PoissonMultiBernoulliMixtureFilter {
  TransitionModel<Scalar> transition_model_;
  MeasurementModel<Scalar> measurement_model_;

  PoissonPointProcess<Scalar, StateSize> poisson_point_process_;

  Scalar survival_probability_;
  Scalar detection_probability_;
  Scalar clutter_intensity_;

  Scalar gating_threshold_;
  Scalar poisson_pruning_threshold_;
  Scalar existence_pruning_threshold_;
  Scalar hypothesis_pruning_threshold_;

  size_t num_association_ = 1;
  size_t num_hypothesis_ = 10;

  uint32_t id_ = 0;

  auto PredictionAndUpdateOfMultiBernoulliMixture(
      const MultiBernoulliMixture<Scalar, StateSize> &multi_bernoulli_mixture,
      const Measurements<Scalar, MeasurementSize> &measurements,
      const Eigen::Matrix<Scalar, StateSize, StateSize> &F,
      const Eigen::Matrix<Scalar, StateSize, StateSize> &Q,
      const Eigen::Matrix<Scalar, MeasurementSize, StateSize> &H,
      const Eigen::Matrix<Scalar, MeasurementSize, MeasurementSize> &R,
      const Eigen::Matrix<Scalar, StateSize, StateSize> &I,
      uint64_t current_time) {

    std::vector<MultiBernoulli<Scalar, StateSize>> pred_upd_multi_bernoullis;
    std::vector<std::vector<Scalar>> pred_upd_multi_bernoullis_log_weights;

    pred_upd_multi_bernoullis.resize(multi_bernoulli_mixture.probs.size());
    pred_upd_multi_bernoullis_log_weights.resize(multi_bernoulli_mixture.probs.size());

    for (size_t i = 0; i < multi_bernoulli_mixture.probs.size(); ++i) {
      const auto &multi_bernoulli =
          multi_bernoulli_mixture.multi_bernoullis[i];
      const auto pred_upd_multi_bernoulli_size =
          multi_bernoulli.size() * (1 + measurements.size());

      auto &pred_upd_multi_bernoulli = pred_upd_multi_bernoullis[i];
      auto &pred_upd_multi_bernoulli_log_weights =
          pred_upd_multi_bernoullis_log_weights[i];

      pred_upd_multi_bernoulli.reserve(pred_upd_multi_bernoulli_size);
      pred_upd_multi_bernoulli_log_weights.reserve(pred_upd_multi_bernoulli_size);

      for (size_t j = 0; j < multi_bernoulli.size(); ++j) {
        const auto &bernoulli = multi_bernoulli[j];

        // PREDICTION
        const auto pred_prob = bernoulli.prob * survival_probability_;
        const auto pred_mean = F * bernoulli.distribution.mean;
        const auto pred_covariance =
            F * bernoulli.distribution.covariance * F.transpose() + Q;

        // MISDETECTION BERNOULLI
        {
          const auto misdetection_probability = 1.f - detection_probability_;
          const auto prob_numerator = pred_prob * misdetection_probability;
          const auto prob_denominator = 1.f - pred_prob + prob_numerator;

          const auto prob = prob_numerator / prob_denominator;
          const auto log_weight = std::log(prob_denominator);

          pred_upd_multi_bernoulli.emplace_back(
              prob,
              MultivariateGaussian<Scalar, StateSize>{pred_mean,
                                                      pred_covariance},
              bernoulli.track.Append(-1, current_time));
          pred_upd_multi_bernoulli_log_weights.emplace_back(log_weight);
        }

        const auto S = H * pred_covariance * H.transpose();
        const auto S_inv = S.inverse();
        const auto S_log_det = std::log(S.determinant());

        const auto K = pred_covariance * H.transpose() * S_inv;

        const auto z_pred = H * pred_mean;
        for (size_t k = 0; k < measurements.size(); ++k) {
          const auto& z = measurements[k];

          const auto z_minus_z_pred = z - z_pred;
          const auto distance =
              (z_minus_z_pred.transpose() * S_inv * z_minus_z_pred)[0, 0];
          if (distance < gating_threshold_) {
            const auto u_covariance = (I - K * H) * pred_covariance;
            const auto covariance = (u_covariance + u_covariance.transpose()) * 0.5f;

            const auto mean = pred_mean + K * z_minus_z_pred;

            const auto log_weight =
                std::log(pred_prob * detection_probability_) -
                0.5f *
                    (distance + S_log_det +
                     static_cast<Scalar>(MeasurementSize) * 1.8378770664093453f);

            pred_upd_multi_bernoulli.emplace_back(
                1.f,
                MultivariateGaussian<Scalar, StateSize>{mean, covariance},
                bernoulli.track.Append(static_cast<int>(k), current_time)
            );
            pred_upd_multi_bernoulli_log_weights.emplace_back(log_weight);
          } else {
            pred_upd_multi_bernoulli.emplace_back();
            pred_upd_multi_bernoulli_log_weights.emplace_back(
                std::numeric_limits<float>::min());
          }
        }
      }
    }
    return std::make_tuple(pred_upd_multi_bernoullis,
                           pred_upd_multi_bernoullis_log_weights);
  }

  auto PredictionOfPoissonPointProcess(
      const PoissonPointProcess<Scalar, StateSize> &poisson_point_process,
      const Eigen::Matrix<Scalar, StateSize, StateSize> &F,
      const Eigen::Matrix<Scalar, StateSize, StateSize> &Q
  ) {

    PoissonPointProcess<Scalar, StateSize> pred_poisson_point_process;
    const auto num_poisson_point_process_distributions =
        poisson_point_process.weights.size() +
        poisson_point_process_.weights.size();

    pred_poisson_point_process.weights.reserve(num_poisson_point_process_distributions);
    pred_poisson_point_process.distributions.reserve(num_poisson_point_process_distributions);

    // ADD NEW POISSON POINT PROCESS DISTRIBUTIONS
    for (size_t i = 0; i < poisson_point_process_.weights.size(); ++i) {
      pred_poisson_point_process.distributions.emplace_back(
          poisson_point_process_.distributions[i]);
      pred_poisson_point_process.weights.emplace_back(poisson_point_process_.weights[i]);
    }

    // PRED PREV POISSON POINT PROCESS DISTRIBUTIONS                                            
    for (size_t i = 0; i < poisson_point_process.weights.size(); ++i) {
      pred_poisson_point_process.distributions.emplace_back(
          F * poisson_point_process.distributions[i].mean,
          F * poisson_point_process.distributions[i].covariance * F.transpose() + Q);
      pred_poisson_point_process.weights.emplace_back(poisson_point_process.weights[i]);
    }
    return pred_poisson_point_process;
  }

  auto AddNewMultiBernoulli(
      const PoissonPointProcess<Scalar, StateSize> &poisson_point_process,
      const Measurements<Scalar, MeasurementSize> &measurements,
      const Eigen::Matrix<Scalar, MeasurementSize, StateSize> &H,
      const Eigen::Matrix<Scalar, MeasurementSize, MeasurementSize> &R,
      const Eigen::Matrix<Scalar, StateSize, StateSize> &I,
      uint64_t current_time) {

    MultiBernoulli<Scalar, StateSize> multi_bernoulli;
    std::vector<Scalar> multi_bernoulli_log_weights;

    multi_bernoulli.reserve(measurements.size());
    multi_bernoulli_log_weights.reserve(measurements.size());
    for (size_t i = 0; i < measurements.size(); ++i) {
      const auto &z = measurements[i];

      Scalar prob = 0.f;
      Scalar weight = 0.f;
      Eigen::Matrix<Scalar, StateSize, 1> mean;
      Eigen::Matrix<Scalar, StateSize, StateSize> covariance;

      mean.setZero();
      covariance.setZero();

      bool assigned = false;
      for (size_t j = 0; j < poisson_point_process.weights.size(); ++j) {
        const auto& distribution = poisson_point_process.distributions[j];
        const auto& distribution_weight = poisson_point_process.weights[j];

        const auto S = H * distribution.covariance * H.transpose() + R;
        const auto S_inv = S.inverse();

        const auto z_pred = H * distribution.mean;
        const auto z_minus_z_pred = z - z_pred;

        const auto distance =
            (z_minus_z_pred.transpose() * S_inv * z_minus_z_pred)[0, 0];
        if (distance < gating_threshold_) {
          const auto S_log_det = std::log(S.determinant());

          const auto K = distribution.covariance * H.transpose() * S_inv;

          const auto p_u_covariance = (I - K * H) * distribution.covariance;
          const auto p_covariance =
              (p_u_covariance + p_u_covariance.transpose()) * 0.5f;

          const auto p_mean = distribution.mean + K * z_minus_z_pred;

          const auto p_weight = (detection_probability_ * distribution_weight) *
                                std::exp(-0.5 * (distance + S_log_det +
                                         static_cast<Scalar>(MeasurementSize) *
                                             1.8378770664093453f));

          weight += p_weight;
          mean += (p_weight * p_mean);
          covariance += (p_weight * (p_covariance + p_mean * p_mean.transpose()));

          assigned = true;
        }
      }

      if (assigned) {
        mean /= weight;
        covariance = covariance / weight - mean * mean.transpose();

        prob = weight;
        weight += clutter_intensity_;
        prob /= weight;

      } else {
        prob = 0.f;
        weight = clutter_intensity_;
      }

      multi_bernoulli.emplace_back(
          prob,
          MultivariateGaussian<Scalar, StateSize>{std::move(mean), std::move(covariance)},
          Track{id_++, current_time, {std::make_tuple(static_cast<int>(i), current_time)}}
      );
      multi_bernoulli_log_weights.emplace_back(std::log(weight));
    }

    return std::make_tuple(multi_bernoulli, multi_bernoulli_log_weights);
  }

  auto Associate(
      const MultiBernoulliMixture<Scalar, StateSize> &multi_bernoulli_mixture,

      const std::vector<MultiBernoulli<Scalar, StateSize>>
          &pred_upd_multi_bernoullis,
      const std::vector<std::vector<Scalar>>
          &pred_upd_multi_bernoullis_log_weights,

      const MultiBernoulli<Scalar, StateSize> &new_multi_bernoulli,
      const std::vector<Scalar> &new_multi_bernoulli_log_weights

  ) {

    std::vector<std::vector<const Bernoulli<Scalar, StateSize> *>>
        associated_multi_bernoullis;
    std::vector<Scalar> associated_weights;

    const auto max_num_hypothesis =
        ComputeMaxNumHypothesis(multi_bernoulli_mixture.probs);

    associated_multi_bernoullis.reserve(max_num_hypothesis);
    associated_weights.reserve(max_num_hypothesis);

    for (size_t i = 0; i < multi_bernoulli_mixture.probs.size(); ++i) {
      const auto prob = multi_bernoulli_mixture.probs[i];

      const auto num_pred_upd_multi_bernoulli =
          multi_bernoulli_mixture.multi_bernoullis[i].size();
      const auto num_new_multi_bernoulli = new_multi_bernoulli.size();

      const auto &pred_upd_multi_bernoulli = pred_upd_multi_bernoullis[i];
      const auto &pred_upd_multi_bernoulli_log_weights =
          pred_upd_multi_bernoullis_log_weights[i];

      auto cost = Eigen::Matrix<double, -1, -1>(num_new_multi_bernoulli,
                                                num_pred_upd_multi_bernoulli +
                                                    num_new_multi_bernoulli);
      for (size_t a = 0; a < num_new_multi_bernoulli; ++a) {
        for (size_t b = 0; b < num_pred_upd_multi_bernoulli + num_new_multi_bernoulli; ++b) {
          cost(a, b) = std::numeric_limits<float>::lowest();
        }
      }

      std::vector<const Bernoulli<Scalar, StateSize>*> multi_bernoulli;
      multi_bernoulli.reserve(num_pred_upd_multi_bernoulli + num_new_multi_bernoulli);

      Scalar total_misdetection_weight = 0.f;
      auto ind = 0;
      for (size_t j = 0; j < num_pred_upd_multi_bernoulli; ++j) {
        const auto misdetection_weight =
            pred_upd_multi_bernoulli_log_weights[ind];
        const auto misdetection_bernoulli = &pred_upd_multi_bernoulli[ind];

        multi_bernoulli.push_back(misdetection_bernoulli);

        total_misdetection_weight += misdetection_weight;

        ind += 1;
        for (size_t k = 0; k < num_new_multi_bernoulli; ++k) {
          const auto detection_weight = pred_upd_multi_bernoulli_log_weights[ind];
          if (detection_weight != std::numeric_limits<Scalar>::min()) {
            cost(k, j) = detection_weight - misdetection_weight;
          }
          ind += 1;
        }
      }

      for (size_t j = 0; j < num_new_multi_bernoulli; ++j) {
        multi_bernoulli.push_back(&new_multi_bernoulli[j]);
        cost(j, j + num_pred_upd_multi_bernoulli) =
            new_multi_bernoulli_log_weights[j];
      }


      auto associations = MurtyMiller<double>::getMBestAssignments(
          cost, ComputeNumAssociation(prob));
      for (const auto &association : associations) {
        std::vector<const Bernoulli<Scalar, StateSize> *>
            associated_multi_bernoulli = multi_bernoulli;
        Scalar association_cost = 0.f;
        for (const auto &edge : association) {
          if (edge.y < num_pred_upd_multi_bernoulli) {
            associated_multi_bernoulli[edge.y] =
                &pred_upd_multi_bernoulli[edge.y *
                                              (1 + num_new_multi_bernoulli) +
                                          edge.x + 1];
          }
          association_cost += static_cast<Scalar>(edge.v);
        }

        const auto associated_weight =
            association_cost + total_misdetection_weight + std::log(prob);
        associated_multi_bernoullis.push_back(
            std::move(associated_multi_bernoulli));
        associated_weights.push_back(associated_weight);
      }
    }

    return std::make_tuple(associated_multi_bernoullis, associated_weights);
  }

  auto
  UpdateAndPrunePoissonPointProcess(const PoissonPointProcess<Scalar, StateSize>
                                        &pred_poisson_point_process) {

    const auto size = pred_poisson_point_process.weights.size();
    PoissonPointProcess<Scalar, StateSize> poisson_point_process;

    poisson_point_process.weights.reserve(size);
    poisson_point_process.distributions.reserve(size);

    const auto misdetection_probability = 1.f - detection_probability_;
    for (size_t i = 0; i < size; ++i) {
      const auto weight =
          pred_poisson_point_process.weights[i] * misdetection_probability;
      if (weight >= poisson_pruning_threshold_) {
        poisson_point_process.weights.push_back(weight);
        poisson_point_process.distributions.push_back(
            std::move(pred_poisson_point_process.distributions[i]));
      }
    }

    return poisson_point_process;
  }

  State<Scalar, StateSize>
  operator()(State<Scalar, StateSize> state,
             const Measurements<Scalar, MeasurementSize> measurements,
             uint64_t current_time, float time_interval) {

    const auto F = transition_model_.F(time_interval);
    const auto Q = transition_model_.Q(time_interval);

    const auto& H = measurement_model_.H;
    const auto& R = measurement_model_.R;

    const auto I = Eigen::Matrix<Scalar, StateSize, StateSize>::Identity();

    auto pred_poisson_point_process =
        PredictionOfPoissonPointProcess(state.poisson_point_process, F, Q);
    auto poisson_point_process =
        UpdateAndPrunePoissonPointProcess(pred_poisson_point_process);

    if (measurements.size() == 0) return state;

    MultiBernoulli<Scalar, StateSize> new_multi_bernoulli;
    std::vector<Scalar> new_multi_bernoulli_log_weights;
    std::tie(new_multi_bernoulli, new_multi_bernoulli_log_weights) =
        AddNewMultiBernoulli(pred_poisson_point_process, measurements, H, R, I,
                             current_time);

    if (state.multi_bernoulli_mixture.probs.size() == 0) {
      MultiBernoulli<Scalar, StateSize> multi_bernoulli;
      multi_bernoulli.reserve(new_multi_bernoulli.size());
      for (const auto& bernoulli: new_multi_bernoulli) {
        if (bernoulli.prob >= existence_pruning_threshold_) {
          multi_bernoulli.push_back(bernoulli);
        }
      }

      return State<Scalar, StateSize>{
        {
          {1.f},
          {std::move(multi_bernoulli)},
        },
        std::move(poisson_point_process)
      };
    }

    std::vector<MultiBernoulli<Scalar, StateSize>> pred_upd_multi_bernoullis;
    std::vector<std::vector<Scalar>> pred_upd_multi_bernoullis_log_weights;
    std::tie(pred_upd_multi_bernoullis, pred_upd_multi_bernoullis_log_weights) =
        PredictionAndUpdateOfMultiBernoulliMixture(
            state.multi_bernoulli_mixture, measurements, F, Q, H, R, I,
            current_time);

    // ASSOCIATE
    std::vector<std::vector<const Bernoulli<Scalar, StateSize> *>> associated_multi_bernoullis;
    std::vector<Scalar> associated_weights;

    std::tie(associated_multi_bernoullis, associated_weights) =
        Associate(state.multi_bernoulli_mixture, pred_upd_multi_bernoullis,
                  pred_upd_multi_bernoullis_log_weights, new_multi_bernoulli,
                  new_multi_bernoulli_log_weights);

    auto multi_bernoulli_mixture = CreateMultiBernoulliMixture(
        associated_multi_bernoullis, associated_weights);

    return State<Scalar, StateSize>{std::move(multi_bernoulli_mixture),
                                    std::move(poisson_point_process)};
  }

  auto CreateMultiBernoulliMixture(
      const std::vector<std::vector<const Bernoulli<Scalar, StateSize> *>>
          &associated_multi_bernoullis,
      const std::vector<Scalar> &associated_weights) {

    MultiBernoulliMixture<Scalar, StateSize> multi_bernoulli_mixture;

    auto probs_and_inds = SoftmaxAndSortAndPrune(associated_weights);
    const auto size = probs_and_inds.size();

    multi_bernoulli_mixture.probs.reserve(size);
    multi_bernoulli_mixture.multi_bernoullis.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      const auto prob = std::get<0>(probs_and_inds[i]);
      const auto ind = std::get<1>(probs_and_inds[i]);

      const auto &associated_multi_bernoulli = associated_multi_bernoullis[ind];

      MultiBernoulli<Scalar, StateSize> multi_bernoulli;
      multi_bernoulli.reserve(associated_multi_bernoulli.size());

      for (const auto &bernoulli : associated_multi_bernoulli) {
        if (bernoulli->prob >= existence_pruning_threshold_) {
          multi_bernoulli.push_back(*bernoulli);
        }
      }

      multi_bernoulli_mixture.probs.push_back(prob);
      multi_bernoulli_mixture.multi_bernoullis.push_back(
          std::move(multi_bernoulli));
    }

    return multi_bernoulli_mixture;
  }

  inline std::vector<std::tuple<Scalar, size_t>>
  SoftmaxAndSortAndPrune(std::vector<Scalar> weights) {
    Scalar max = std::numeric_limits<Scalar>::lowest();

    for (const auto &weight : weights) {
      max = std::max(max, weight);
    }

    Scalar sum = 0.f;
    for (auto &weight : weights) {
      weight = std::exp(weight - max);
      sum += weight;
    }

    std::vector<std::tuple<Scalar, size_t>> probs_and_inds;
    probs_and_inds.reserve(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
      probs_and_inds.emplace_back(weights[i] / sum, i);
    }

    std::sort(probs_and_inds.begin(), probs_and_inds.end(),
              [](const auto &x, const auto &y) {
                return std::get<0>(x) > std::get<0>(y);
              });

    std::vector<std::tuple<Scalar, size_t>> pruned_probs_and_inds;
    pruned_probs_and_inds.reserve(probs_and_inds.size());

    sum = 0.f;
    for (size_t i = 0; i < probs_and_inds.size(); ++i) {
      const auto &prob_and_ind = probs_and_inds[i];
      if (std::get<0>(prob_and_ind) >= hypothesis_pruning_threshold_ and
          i < num_hypothesis_) {
        sum += std::get<0>(prob_and_ind);
        pruned_probs_and_inds.push_back(prob_and_ind);
      } else {
        break;
      }
    }

    for (auto &prob_and_ind : pruned_probs_and_inds) {
      std::get<0>(prob_and_ind) /= sum;
    }
    return pruned_probs_and_inds;
  }

  inline size_t ComputeMaxNumHypothesis(const std::vector<Scalar> &probs) {
    size_t max_num_hypothesis = 0;
    for (const auto& prob: probs) {
      max_num_hypothesis += ComputeNumAssociation(prob);
    }
    return max_num_hypothesis;
  }

  inline size_t ComputeNumAssociation(const Scalar prob) {
    return std::max(num_association_,
                    static_cast<size_t>(std::ceil(
                        prob * static_cast<Scalar>(num_hypothesis_))));
  }
};
