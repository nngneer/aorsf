cmake_minimum_required(VERSION 3.15)
project(pyaorsf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and nanobind
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# Find Armadillo
find_package(Armadillo REQUIRED)

# Find OpenMP (optional)
find_package(OpenMP)

# Path to aorsf core
set(AORSF_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/core")

# Core library sources (compile directly, don't require pre-built library)
set(AORSF_CORE_SOURCES
    ${AORSF_CORE_DIR}/Coxph.cpp
    ${AORSF_CORE_DIR}/Forest.cpp
    ${AORSF_CORE_DIR}/ForestClassification.cpp
    ${AORSF_CORE_DIR}/ForestRegression.cpp
    ${AORSF_CORE_DIR}/ForestSurvival.cpp
    ${AORSF_CORE_DIR}/Tree.cpp
    ${AORSF_CORE_DIR}/TreeClassification.cpp
    ${AORSF_CORE_DIR}/TreeRegression.cpp
    ${AORSF_CORE_DIR}/TreeSurvival.cpp
    ${AORSF_CORE_DIR}/utility.cpp
)

# Python binding sources
set(PYAORSF_SOURCES
    src/_pyaorsf.cpp
)

# Create the Python module
nanobind_add_module(_pyaorsf
    ${AORSF_CORE_SOURCES}
    ${PYAORSF_SOURCES}
)

target_include_directories(_pyaorsf PRIVATE
    ${AORSF_CORE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(_pyaorsf PRIVATE
    ${ARMADILLO_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_pyaorsf PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install the module
install(TARGETS _pyaorsf LIBRARY DESTINATION pyaorsf)
