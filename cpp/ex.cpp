#include <Eigen/Dense>
#include <boost/math/special_functions/factorials.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <pthreadpool.h>

#include "json11.hpp"

using namespace json11;

struct Track {
  std::string name;
  uint64_t start_time;
};

struct Z {
  int32_t index;
  uint64_t time;
};

template <typename Scalar, size_t Dim> struct Gaussian {
  Eigen::Matrix<Scalar, Dim, 1> mu;
  Eigen::Matrix<Scalar, Dim, Dim> covar;
};

template <typename Scalar, size_t Dim> struct HypEntry {
  Gaussian<Scalar, Dim> state;
  Scalar prob;
  Scalar log_weight;
  Track track;
  std::vector<Z> zs;
};

template <typename Scalar, size_t Dim> struct Hyp {
  Scalar prob;
  std::vector<HypEntry<Scalar, Dim>> entries;
};

template <typename Scalar, size_t Dim> struct PPP {
  Scalar weight;
  Gaussian<Scalar, Dim> gauss;
};

template <typename Scalar, size_t Dim> struct State {
  std::vector<Hyp<Scalar, Dim>> hyps;
  std::vector<PPP<Scalar, Dim>> ppps;
};
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

template <typename Scalar, size_t StateDim, size_t MeasurementDim>
struct PMBMFilter {
  std::vector<PPP<float, StateDim>> ppps_;

  TransitionModel<Scalar> transition_model_;
  MeasurementModel<Scalar> measurement_model_;

  Scalar survival_probability_;
  Scalar detection_probability_;
  Scalar clutter_intensity_;

  Scalar gating_threshold_;
  Scalar poisson_pruning_threshold_;
  Scalar hyp_pruning_threshold_;
  Scalar existence_pruning_threshold_;

  struct PredictAndUpdateHypContext {
    const std::vector<Hyp<Scalar, StateDim>> &hyps;
    const std::vector<Eigen::Matrix<Scalar, MeasurementDim, 1>> &zs;

    uint64_t current_time;

    std::vector<HypEntry<Scalar, StateDim> *> &hyp_to_entries;

    Scalar detection_probability;
    Scalar survival_probability;
    Scalar gating_threshold;

    const Eigen::Matrix<Scalar, StateDim, StateDim> &F;
    const Eigen::Matrix<Scalar, StateDim, StateDim> &Q;

    const Eigen::Matrix<Scalar, MeasurementDim, StateDim> &H;
    const Eigen::Matrix<Scalar, MeasurementDim, MeasurementDim> &R;

    const Eigen::Matrix<Scalar, StateDim, StateDim> I;
  };

  static void PredictAndUpdateHyp(PredictAndUpdateHypContext *context, size_t i,
                                  size_t j) {
    if (j > context->hyps[i].entries.size())
      return;

    const auto &prev_entry = context->hyps[i].entries[j];

    auto current_entry =
        &context->hyp_to_entries[i][j * (1 + context->zs.size())];

    // PREDICTION
    const auto prob = prev_entry.prob * context->survival_probability;
    const auto mu = context->F * prev_entry.state.mu;
    const auto covar =
        context->F * prev_entry.state.covar * context->F.transpose();

    // MISDETECTIKN
    const auto numerator = prob * (1.f - context->detection_probability);
    const auto denominator = (1.f - prob + numerator);

    *current_entry = HypEntry<Scalar, StateDim>{
        Gaussian<Scalar, StateDim>{mu, covar},
        numerator / denominator,
        std::log(denominator),
        prev_entry.track,
        prev_entry.zs,
    };
    current_entry->zs.push_back(Z{-1, context->current_time});

    const auto S = context->H * covar * context->H.transpose();
    const auto z_pred = context->H * mu;

    const auto S_inv = S.inverse();
    const auto S_log_det = std::log(S.determinant());

    const auto K = covar * context->H.transpose() * S_inv;

    int z_index = -1;
    for (const auto z : context->zs) {
      z_index += 1;
      current_entry += 1;

      const auto z_minus_z_pred = z - z_pred;
      const auto distance =
          (z_minus_z_pred.transpose() * S_inv * z_minus_z_pred)[0, 0];

      if (distance < context->gating_threshold) {
        const auto U_P = (context->I - K * context->H) * covar;
        const auto P = (U_P + U_P.transpose()) * 0.5f;
        const auto x = mu + K * z_minus_z_pred;

        *current_entry = HypEntry<Scalar, StateDim>{
            Gaussian<Scalar, StateDim>{x, P},
            1.f,
            std::log(prob * context->detection_probability) -
                0.5f *
                    (distance + S_log_det +
                     static_cast<Scalar>(MeasurementDim) * 1.8378770664093453f),
            prev_entry.track,
            prev_entry.zs,
        };
        current_entry->zs.push_back(Z{z_index, context->current_time});
      } else {
        // TODO
      }
    }
  }

  void operator()(State<Scalar, StateDim> state,
                  std::vector<Eigen::Matrix<Scalar, MeasurementDim, 1>> zs,
                  uint64_t current_time, float time_interval) {

    pthreadpool_t threadpool_ = pthreadpool_create(0);

    const auto F = transition_model_.F(time_interval);
    const auto Q = transition_model_.Q(time_interval);


    // PREDICTION AND UPDATE STEP OF MBM
    size_t max_entries = 0;
    std::vector<HypEntry<Scalar, StateDim> *> hyp_to_entries;
    hyp_to_entries.reserve(state.hyps.size());

    for (auto const &hyp : state.hyps) {
      max_entries = std::max(max_entries, hyp.entries.size());
      hyp_to_entries.push_back(
          new HypEntry<Scalar, StateDim>[hyp.entries.size() * (1 + zs.size())]);
    }

    {
      PredictAndUpdateHypContext context{
          state.hyps,
          zs,
          current_time,
          hyp_to_entries,
          detection_probability_,
          survival_probability_,
          gating_threshold_,
          F,
          Q,
          measurement_model_.H,
          measurement_model_.R,
          Eigen::Matrix<Scalar, StateDim, StateDim>::Identity(),
      };

      const auto start = std::chrono::steady_clock::now();
      pthreadpool_parallelize_2d(
          threadpool_, (pthreadpool_task_2d_t)PredictAndUpdateHyp, &context,
          state.hyps.size(), max_entries, PTHREADPOOL_FLAG_DISABLE_DENORMALS);
      const auto end = std::chrono::steady_clock::now();
      const std::chrono::duration<double> elapsed_seconds = end - start;
      std::cout << elapsed_seconds.count() << '\n';
    }

    // PREDICTION STEP OF PPP'S
    std::vector<PPP<Scalar, StateDim>> pred_ppps = ppps_;
    pred_ppps.reserve(state.ppps.size() + ppps_.size());
    for (const auto &ppp : state.ppps) {
      pred_ppps.push_back(PPP<Scalar, StateDim>{
          ppp.weight * survival_probability_,
          Gaussian<Scalar, StateDim>{F * ppp.gauss.mu,
                                     F * ppp.gauss.covar * F.transpose() + Q}});
    }

    /// UPDATE AND PRUNER PPPS
    std::vector<PPP<Scalar, StateDim>> ppps;
    ppps.reserve(pred_ppps.size());
    for (const auto &ppp : ppps_) {
      const auto weight = ppp.weight * (1.f - detection_probability_);
      if (weight >= poisson_pruning_threshold_) {
      }
    }
  }
};

template <typename T> void GetType(T t);

Eigen::Matrix<float, 4, 1> MuFromJson(const Json &cfg) {
  Eigen::Matrix<float, 4, 1> x;
  for (size_t i = 0; i < 4; ++i) {
    x(i, 0) = cfg.array_items()[i].array_items()[0].number_value();
  }
  return x;
}

Eigen::Matrix<float, 4, 4> CovarFromJson(const Json &cfg) {
  Eigen::Matrix<float, 4, 4> x;
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      x(i, j) = cfg.array_items()[i].array_items()[j].number_value();
    }
  }
  return x;
}

State<float, 4> StateFromJson(const Json &cfg) {
  State<float, 4> state;

  for (const auto &hyp_cfg : cfg["hyps"].array_items()) {
    Hyp<float, 4> hyp{static_cast<float>(hyp_cfg["weight"].number_value())};
    for (const auto &entry : hyp_cfg["entries"].array_items()) {
      std::vector<Z> zs;
      for (const auto &z : entry["hist"].array_items()) {
        zs.push_back(
            Z{static_cast<int32_t>(z.array_items()[0].number_value()),
              static_cast<uint64_t>(z.array_items()[1].number_value())});
      }

      hyp.entries.push_back(HypEntry<float, 4>{
          Gaussian<float, 4>{MuFromJson(entry["gauss"]["mu"]),
                             CovarFromJson(entry["gauss"]["covar"])},
          static_cast<float>(entry["prob"].number_value()),
          0.f,
          Track{
              entry["name"].string_value(),
              static_cast<uint64_t>(entry["start_time"].number_value()),
          },
          zs,
      });
    }
    state.hyps.push_back(hyp);
  }

  for (const auto &ppp : cfg["ppps"].array_items()) {
    state.ppps.push_back(PPP<float, 4>{
        static_cast<float>(ppp["weight"].number_value()),
        Gaussian<float, 4>{MuFromJson(ppp["gauss"]["mu"]),
                           CovarFromJson(ppp["gauss"]["covar"])}});
  }

  return state;
}

std::vector<Eigen::Matrix<float, 2, 1>> ZsFromJSon(const Json &cfg) {
  std::vector<Eigen::Matrix<float, 2, 1>> zs;

  for (auto z_cfg : cfg["zs"].array_items()) {
    Eigen::Matrix<float, 2, 1> z;
    for (size_t i = 0; i < 2; ++i) {
      z(i, 0) = z_cfg.array_items()[i].array_items()[0].number_value();
    }
    zs.push_back(z);
  }

  return zs;
}

std::string read(const std::string &path) {
  std::ifstream fp(path);
  std::stringstream ss;
  ss << fp.rdbuf();
  return ss.str();
}

Json parse(const std::string &in) {
  std::string err;
  return Json::parse(in, err);
}

int main(int argc, char **argv) {
  auto cfg = parse(read(argv[1]));
  auto state = StateFromJson(cfg);
  auto zs = ZsFromJSon(cfg);

  size_t h = 600, w = 800;
  size_t nh = 8, nw = 8;
  std::vector<PPP<float, 4>> ppps;
  float ppp_weight = 0.001;
  for (size_t i = 0; i < nw; ++i) {
    for (size_t j = 0; j < nh; ++j) {
      Gaussian<float, 4> gauss;
      gauss.mu.setZero();
      gauss.covar.setZero();

      const auto ww = static_cast<float>(w) / static_cast<float>(nw);
      const auto hh = static_cast<float>(h) / static_cast<float>(nh);
      const auto fi = static_cast<float>(i);
      const auto fj = static_cast<float>(j);

      gauss.mu(0, 0) = (fi + 0.5f) * ww;
      gauss.mu(2, 0) = (fj + 0.5f) * hh;

      gauss.covar(0, 0) = ww;
      gauss.covar(1, 1) = 1;
      gauss.covar(2, 2) = hh;
      gauss.covar(3, 3) = 1;

      ppps.push_back(PPP<float, 4>{ppp_weight, gauss});
    }
  }

  TransitionModel<float> transition_model{120.f, 120.f};
  MeasurementModel<float> measurement_model{100.f, 100.f};

  PMBMFilter<float, 4, 2> filter{ppps, transition_model, measurement_model};

  for (int i = 0; i < 1000; ++i)
    state.hyps.push_back(state.hyps[0]);

  filter(state, zs, 74, 1);
}
