cmake_minimum_required(VERSION 3.15)
project(aorsf_c VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Armadillo
find_package(Armadillo REQUIRED)

# Find OpenMP (optional)
find_package(OpenMP)

# Option to build shared or static library
option(BUILD_SHARED "Build shared library" ON)

# C API sources
set(AORSF_C_API_SOURCES
    aorsf_c.cpp
)

set(AORSF_C_API_HEADERS
    aorsf_c.h
    aorsf_export.h
)

# Core library sources (include directly to avoid dependency issues)
set(AORSF_CORE_SOURCES
    ../core/Coxph.cpp
    ../core/Forest.cpp
    ../core/ForestClassification.cpp
    ../core/ForestRegression.cpp
    ../core/ForestSurvival.cpp
    ../core/Tree.cpp
    ../core/TreeClassification.cpp
    ../core/TreeRegression.cpp
    ../core/TreeSurvival.cpp
    ../core/utility.cpp
)

# Combine all sources
set(ALL_SOURCES ${AORSF_C_API_SOURCES} ${AORSF_CORE_SOURCES})

# Create library
if(BUILD_SHARED)
    add_library(aorsf_c SHARED ${ALL_SOURCES})
    target_compile_definitions(aorsf_c PRIVATE AORSF_BUILD_SHARED)
    # Set RPATH for macOS
    if(APPLE)
        set_target_properties(aorsf_c PROPERTIES
            INSTALL_RPATH "@loader_path"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    elseif(UNIX)
        set_target_properties(aorsf_c PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
else()
    add_library(aorsf_c STATIC ${ALL_SOURCES})
endif()

# Include directories
target_include_directories(aorsf_c
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/aorsf>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../core
        ${ARMADILLO_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(aorsf_c
    PUBLIC
        ${ARMADILLO_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(aorsf_c PUBLIC OpenMP::OpenMP_CXX)
endif()

# Set visibility for symbols (hide all except exported)
if(BUILD_SHARED)
    set_target_properties(aorsf_c PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()

# Platform-specific settings
if(WIN32)
    # Windows: no prefix for DLL
    set_target_properties(aorsf_c PROPERTIES PREFIX "")
    # Output to bin directory on Windows
    set_target_properties(aorsf_c PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
elseif(APPLE)
    # macOS: .dylib extension
    set_target_properties(aorsf_c PROPERTIES SUFFIX ".dylib")
endif()

# Install rules
install(TARGETS aorsf_c
    EXPORT aorsf_cTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${AORSF_C_API_HEADERS}
    DESTINATION include/aorsf
)

# Export for find_package
install(EXPORT aorsf_cTargets
    FILE aorsf_cTargets.cmake
    NAMESPACE aorsf::
    DESTINATION lib/cmake/aorsf_c
)

# Optional: Build tests
option(BUILD_C_API_TESTS "Build C API tests" OFF)
if(BUILD_C_API_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
