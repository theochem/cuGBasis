#ifndef CHEMTOOLS_CUDA_INCLUDE_UTILS_H_
#define CHEMTOOLS_CUDA_INCLUDE_UTILS_H_

#include <array>
#include <cmath>
#include <utility>
#include <pybind11/numpy.h>

#include "cuda_utils.cuh"

namespace chemtools {

/// Convert an array to array_t without copying.
inline pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> as_pyarray_from_vector(std::vector<double> &seq) {
  // Took this from github/pybind/pybind11/issues/1042 from user YannickJadoul
  auto size = seq.size();
  auto data = seq.data();
  std::unique_ptr<std::vector<double>> seq_ptr = std::make_unique<std::vector<double>>(std::move(seq));
  auto capsule = pybind11::capsule(
      seq_ptr.get(),
      [](void *p) { std::unique_ptr<std::vector<double>>(reinterpret_cast<std::vector<double>*>(p)); }
      );
  seq_ptr.release();
  return pybind11::array(size, data, capsule);
}
} // end chemtools
#endif //CHEMTOOLS_CUDA_INCLUDE_UTILS_H_
