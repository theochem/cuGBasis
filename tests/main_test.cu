#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include <pybind11/embed.h>

namespace py = pybind11;
using namespace py::literals;

int main(int argc, char* argv[]) {
  // Setup up Python interpretor and pybind11 so that python can be used for all tests
  py::scoped_interpreter guard{};

  // Run the tests
  int result = Catch::Session().run(argc, argv);

  // clean -up

  return result;
}