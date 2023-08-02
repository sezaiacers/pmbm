#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Miller.h"
#include <iostream>
#include <tuple>
#include <vector>

std::tuple<std::vector<std::vector<int>>, std::vector<double>>
murty(std::vector<std::vector<double>> vs, int kbest) {
  MurtyMiller<double>::WeightMatrix cost(vs.size(), vs[0].size());
  for (int i = 0; i < vs.size(); ++i) {
    for (int j = 0; j < vs[i].size(); ++j) {
      cost(i, j) = vs[i][j];
    }
  }

  std::vector<std::vector<int>> assocs;
  std::vector<double> assoc_costs;
  auto solutions = MurtyMiller<double>::getMBestAssignments(cost, kbest);
  for (const auto &solution : solutions) {

    std::vector<int> assoc;
    double sum = 0;
    for (const auto &edge : solution) {
      sum += edge.v;
      assoc.push_back(edge.y);
    }
    assocs.push_back(std::move(assoc));
    assoc_costs.push_back(sum);
  }
  return std::make_tuple(std::move(assocs), std::move(assoc_costs));
}

/*
int main(int argc, char** argv) {
 std::vector<std::vector<double>> vs = {
   {5.9960632e+00, 6.5997372e+00, 1.0000000e+04, 1.0000000e+04, 1.0000000e+04},
   {7.1424918e+00, 1.0000000e+04, 6.2361584e+00, 1.0000000e+04, 1.0000000e+04},
   {1.0000000e+04, 1.0000000e+04, 1.0000000e+04, 6.9077554e+00, 1.0000000e+04},
   {1.0000000e+04, 1.0000000e+04, 1.0000000e+04, 1.0000000e+04, 6.9077554e+00},
 };

 std::vector<std::vector<int>> assocs;
 std::vector<double> costs;

 std::tie(assocs, costs) = murty(vs, 1);

 std::cout << costs[0] << std::endl;



}
 */

PYBIND11_MODULE(opt, m) {
  m.doc() = "pybind11 opt plugin"; // optional module docstring

  m.def("murty", &murty, "murty optimization algorithm");
}
