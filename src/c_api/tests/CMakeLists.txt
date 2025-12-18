cmake_minimum_required(VERSION 3.15)

# C API test (pure C to verify C compatibility)
add_executable(test_c_api test_c_api.c)
target_link_libraries(test_c_api PRIVATE aorsf_c)
target_include_directories(test_c_api PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Need to link math library on some platforms
if(UNIX AND NOT APPLE)
    target_link_libraries(test_c_api PRIVATE m)
endif()

# Add to ctest
add_test(NAME test_c_api COMMAND test_c_api)
