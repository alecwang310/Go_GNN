cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(gognn_test)

# Point this to your libtorch folder if not in environment variables
# set(CMAKE_PREFIX_PATH "C:/path/to/libtorch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(gognn_test main.cpp)
target_link_libraries(gognn_test "${TORCH_LIBRARIES}")
set_property(TARGET gognn_test PROPERTY CXX_STANDARD 17)

# pybind11 module for exposing C++ search to Python
# CMake 4.x removed python_add_library, so we build manually instead of using pybind11_add_module
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

add_library(gomcts MODULE bindings.cpp)
target_link_libraries(gomcts PRIVATE "${TORCH_LIBRARIES}" pybind11::headers pybind11::module)
target_link_libraries(gomcts PRIVATE Python3::Module)
set_property(TARGET gomcts PROPERTY CXX_STANDARD 17)
set_property(TARGET gomcts PROPERTY PREFIX "")

set_property(TARGET gomcts PROPERTY SUFFIX ".pyd")