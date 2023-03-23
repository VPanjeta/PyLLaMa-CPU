cmake_minimum_required(VERSION 3.13)
project(pyllama_cpu)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE Release)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I. -O3 -DNDEBUG -std=c11")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I. -O3 -DNDEBUG -std=c++14")

include(FetchContent)
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG        v2.6.2
	GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(pyllama_cpu bridge.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pyllama_cpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/libllama.a")
set_target_properties(pyllama_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(pyllama_cpu PUBLIC cxx_std_14)
set_target_properties(pyllama_cpu PROPERTIES SUFFIX ".so")