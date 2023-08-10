#include <eigen/dense>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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
  uint32_t start_time;
  std::vector<std::tuple<int, uint32_t>> zs;

  Track Append(int index, uint32_t time) const {
    auto track = *this;
    track.zs.emplace_back(index, time);
    return track;
  }
};

template <typename Scalar, size_t Size> struct MultivariateGaussian {
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

  Scalar gating_threshold_;

  State<Scalar, StateSize>
  operator()(State<Scalar, StateSize> state,
             Measurements<Scalar, MeasurementSize> measurements,
             uint32_t current_time, float time_interval) {

    const auto F = transition_model_.F(time_interval);
    const auto Q = transition_model_.Q(time_interval);

    const auto& H = measurement_model_.H;
    const auto& R = measurement_model_.R;

    const auto I = Eigen::Matrix<Scalar, StateSize, StateSize>::Identity();


    // PREDICTION AND UPDATE OF MULTI-BERNOULLI MIXTURE
    std::vector<MultiBernoulli<Scalar, StateSize>> pred_upd_multi_bernoullis;
    std::vector<std::vector<Scalar>> pred_upd_multi_bernoullis_log_weights;

    pred_upd_multi_bernoullis.resize(state.multi_bernoulli_mixture.probs.size());
    pred_upd_multi_bernoullis_log_weights.resize(state.multi_bernoulli_mixture.probs.size());

    for (size_t i = 0; i < state.multi_bernoulli_mixture.probs.size(); ++i) {
      const auto &multi_bernoulli =
          state.multi_bernoulli_mixture.multi_bernoullis[i];
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

    return State<Scalar, StateSize>{};
  }
};

struct A {
  int a;
};

int main(int argc, char **argv) {
  TransitionModel<float> transition_model{120.f, 120.f};
  MeasurementModel<float> measurement_model{100.f, 100.f};

  PoissonMultiBernoulliMixtureFilter<float, 4, 2> f{
    transition_model,
    measurement_model,

  };
  State<float, 4> a{};
  Measurements<float, 2> m{};
  f(a, m, 4, 5.f);


}
