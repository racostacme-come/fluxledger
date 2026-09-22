#include "transport.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
PYBIND11_MODULE(_core, m) {
    m.doc() = "Conservative periodic transport with MC reconstruction and SSPRK3";
    m.def(
        "solve",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> initial, double a,
           double nu, double length, double duration, double cfl, const std::string &scheme,
           std::size_t max_steps) {
            if (initial.ndim() != 1)
                throw std::invalid_argument("initial must be one-dimensional");
            const auto kind = fluxledger::parse_scheme(scheme);
            std::vector<double> input(initial.data(), initial.data() + initial.size());
            fluxledger::Result result;
            {
                py::gil_scoped_release release;
                result = fluxledger::solve(input, a, nu, length, duration, cfl, kind, max_steps);
            }
            py::array_t<double> output(result.values.size());
            std::copy(result.values.begin(), result.values.end(), output.mutable_data());
            return py::make_tuple(output, result.steps, result.dt);
        },
        py::arg("initial"), py::arg("velocity"), py::arg("diffusivity"), py::arg("length"),
        py::arg("duration"), py::arg("cfl"), py::arg("scheme"), py::arg("max_steps"));
}
