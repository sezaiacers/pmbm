#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/pytypes.h>

#include <iostream>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "pmbm.hpp"

PYBIND11_MODULE(pmbm, m) {
  m.doc() = "pmbm filter"; // optional module docstring
  pybind11::class_<TransitionModel<float>>(m, "TransitionModel")
    .def(pybind11::init<float, float>())
    .def("F", &TransitionModel<float>::F)
    .def("Q", &TransitionModel<float>::Q)
    .def_readonly("noise_diff_coeff_x", &TransitionModel<float>::noise_diff_coeff_x)
    .def_readonly("noise_diff_coeff_y", &TransitionModel<float>::noise_diff_coeff_y);

  pybind11::class_<MeasurementModel<float>>(m, "MeasurementModel")
    .def(pybind11::init<float, float>())
    .def_readonly("H", &MeasurementModel<float>::H)
    .def_readonly("R", &MeasurementModel<float>::R);

  pybind11::class_<Track>(m, "Track")
    .def(pybind11::init<uint32_t, uint32_t, std::vector<std::tuple<int, uint32_t>>>())
    .def_readonly("id", &Track::id)
    .def_readonly("start_time", &Track::start_time)
    .def_readonly("zs", &Track::zs);

  pybind11::class_<MultivariateGaussian<float, 4>>(m, "MultivariateGaussian")
    .def(pybind11::init<Eigen::Matrix<float, 4, 1>, Eigen::Matrix<float, 4, 4>>())
    .def_readonly("mean", &MultivariateGaussian<float, 4>::mean)
    .def_readonly("mean", &MultivariateGaussian<float, 4>::covariance);

  pybind11::class_<Bernoulli<float, 4>>(m, "Bernoulli")
    .def(pybind11::init<float, MultivariateGaussian<float, 4>, Track>())
    .def_readonly("prob", &Bernoulli<float, 4>::prob)
    .def_readonly("distribution", &Bernoulli<float, 4>::distribution)
    .def_readonly("track", &Bernoulli<float, 4>::track);

  pybind11::class_<MultiBernoulliMixture<float, 4>>(m, "MultiBernoulliMixture")
    .def(pybind11::init<>())
    .def_readonly("probs", &MultiBernoulliMixture<float, 4>::probs)
    .def_readonly("multi_bernoullis", &MultiBernoulliMixture<float, 4>::multi_bernoullis);

  pybind11::class_<PoissonPointProcess<float, 4>>(m, "PoissonPointProcess")
    .def(pybind11::init<std::vector<float>, std::vector<MultivariateGaussian<float, 4>>>())
    .def_readonly("weights", &PoissonPointProcess<float, 4>::weights)
    .def_readonly("distributions", &PoissonPointProcess<float, 4>::distributions);

  pybind11::class_<State<float, 4>>(m, "State")
    .def(pybind11::init<>())
    .def_readonly("multi_bernoulli_mixture", &State<float, 4>::multi_bernoulli_mixture)
    .def_readonly("poisson_point_process", &State<float, 4>::poisson_point_process);

  pybind11::class_<PoissonMultiBernoulliMixtureFilter<float, 4, 2>>(m, "PoissonMultiBernoulliMixtureFilter")
    .def(pybind11::init<
        TransitionModel<float>,
        MeasurementModel<float>,
        PoissonPointProcess<float, 4>,
        float,
        float,
        float,

        float,
        float,
        float,
        float,

        size_t,
        size_t>())
    .def("__call__", &PoissonMultiBernoulliMixtureFilter<float, 4, 2>::operator());
}
