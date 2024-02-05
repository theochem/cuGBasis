#ifndef CHEMTOOLS_CUDA_INCLUDE_BOYS_FUNCTIONS_CUH_
#define CHEMTOOLS_CUDA_INCLUDE_BOYS_FUNCTIONS_CUH_

#include <type_traits>

namespace chemtools {
/*
 * Boys Function computed with the "forward" recurrence relation.
 *    Note that it is numerically unstable for small t or large M.
 *    This is due to the closest in near-values that both of them are to each other.
 *
 */
template<int M>
__device__
__forceinline__
typename std::enable_if<(M == 0), double>::type
boys_function(const double &param) {
  double r = erf(sqrt(param)) * rsqrt(param);
  r *= 8.8622692545275805e-01;
  return r;
}

template<int M>
__device__
__forceinline__
typename std::enable_if<(M > 0), double>::type
boys_function(const double &param) {
  double r = (2.0 * __int2double_rn(M) - 1.0) * boys_function<M - 1>(param) - exp(-param);
  r /= (2.0 * param);
  return r;
}
}
#endif /* BOYS_FUNCTIONS_H_ */
