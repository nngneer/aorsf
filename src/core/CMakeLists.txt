cmake_minimum_required(VERSION 3.15)
project(aorsf_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Armadillo
find_package(Armadillo REQUIRED)

# Find OpenMP (optional)
find_package(OpenMP)

# Core library sources
set(AORSF_CORE_SOURCES
    Coxph.cpp
    Forest.cpp
    ForestClassification.cpp
    ForestRegression.cpp
    ForestSurvival.cpp
    Tree.cpp
    TreeClassification.cpp
    TreeRegression.cpp
    TreeSurvival.cpp
    utility.cpp
)

# Core library headers
set(AORSF_CORE_HEADERS
    arma_config.h
    Callbacks.h
    Coxph.h
    Data.h
    Exceptions.h
    Forest.h
    ForestClassification.h
    ForestRegression.h
    ForestSurvival.h
    globals.h
    Interrupts.h
    Output.h
    RMath.h
    Tree.h
    TreeClassification.h
    TreeRegression.h
    TreeSurvival.h
    utility.h
)

# Create static library
add_library(aorsf_core STATIC ${AORSF_CORE_SOURCES})

target_include_directories(aorsf_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/aorsf>
        ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(aorsf_core
    PUBLIC
        ${ARMADILLO_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(aorsf_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Install rules
install(TARGETS aorsf_core
    EXPORT aorsf_coreTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${AORSF_CORE_HEADERS}
    DESTINATION include/aorsf
)

# Export for find_package
install(EXPORT aorsf_coreTargets
    FILE aorsf_coreTargets.cmake
    NAMESPACE aorsf::
    DESTINATION lib/cmake/aorsf_core
)

# Optional: Build tests
option(BUILD_TESTS "Build standalone tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
