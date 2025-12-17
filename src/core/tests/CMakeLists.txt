# Standalone test executable
add_executable(test_standalone test_standalone.cpp)
target_link_libraries(test_standalone PRIVATE aorsf_core)

# Survival test executable
add_executable(test_survival test_survival.cpp)
target_link_libraries(test_survival PRIVATE aorsf_core)

# Enable testing
enable_testing()
add_test(NAME standalone_test COMMAND test_standalone)
add_test(NAME survival_test COMMAND test_survival)
