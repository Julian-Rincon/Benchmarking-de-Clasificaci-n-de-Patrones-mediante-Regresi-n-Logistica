cmake_minimum_required(VERSION 3.16)

project(logistic_regression_eigen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(logistic_regression_eigen src/main.cpp)

target_include_directories(logistic_regression_eigen PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen
)

if (MSVC)
    target_compile_options(logistic_regression_eigen PRIVATE /O2)
else()
    target_compile_options(logistic_regression_eigen PRIVATE -O2)
    target_link_options(logistic_regression_eigen PRIVATE -static -static-libgcc -static-libstdc++)
endif()
