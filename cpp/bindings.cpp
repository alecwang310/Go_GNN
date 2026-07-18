#include <pybind11/pybind11.h>
#include <string>

#include "goboard.hpp"
#include "search.hpp"

namespace py = pybind11;

PYBIND11_MODULE(gomcts, m) {
    m.doc() = "C++ MCTS search for Go GNN engine";

    m.attr("BLACK") = (int)BLACK;
    m.attr("WHITE") = (int)WHITE;
    m.attr("EMPTY") = (int)EMPTY;

    py::class_<GoBoard>(m, "GoBoard")
        .def(py::init<>())
        .def("reset", &GoBoard::reset)
        .def("get_index", &GoBoard::get_index)
        .def("is_legal", &GoBoard::is_legal)
        .def("play_move", &GoBoard::play_move)
        .def("calculate_area_score", &GoBoard::calculate_area_score)
        .def_readwrite("komi", &GoBoard::komi)
        .def_readonly_static("PADDED_SIZE", &GoBoard::PADDED_SIZE);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<const std::string&, const std::string&, float, int, float, float, int>(),
             py::arg("model_path"),
             py::arg("device") = "cpu",
             py::arg("komi") = 7.5f,
             py::arg("num_simulations") = 200,
             py::arg("cpuct") = 2.0f,
             py::arg("fpu_value") = 1.25f,
             py::arg("num_threads") = 16)
        .def("search", &MCTS::search)
        .def("set_num_simulations", &MCTS::set_num_simulations)
        .def("get_num_simulations", &MCTS::get_num_simulations)
        .def("set_num_threads", &MCTS::set_num_threads)
        .def("get_num_threads", &MCTS::get_num_threads);
}
