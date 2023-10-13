#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "fastnet.h"

namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;

PYBIND11_MODULE(fastnet, m) {
    py::class_<Graph>(m, "Graph")
    .def(py::init<int, std::vector<std::vector<float>>, std::vector<std::string>,  std::vector<float>, std::vector<float>, float>())
    .def("nodes", &Graph::get_nodes, py::call_guard<py::gil_scoped_release>())
    .def("check_visible_neigh", &Graph::check_visible_neigh, py::call_guard<py::gil_scoped_release>());

    py::class_<Node>(m, "Node")
    .def(py::init<int&, std::vector<float>, std::string>())
    .def("params", &Node::get_params, py::call_guard<py::gil_scoped_release>())
    .def("color", &Node::get_color, py::call_guard<py::gil_scoped_release>())
    .def("neighbours", &Node::get_neighs, py::call_guard<py::gil_scoped_release>());
}