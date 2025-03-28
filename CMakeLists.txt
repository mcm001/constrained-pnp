cmake_minimum_required(VERSION 3.20)

project(constrained_pnp)

enable_testing()

# only build wpimath
set(WITH_JAVA OFF CACHE BOOL "" FORCE)
set(WITH_JAVA_SOURCE OFF CACHE BOOL "" FORCE)
set(WITH_CSCORE OFF CACHE BOOL "" FORCE)
set(WITH_WPIMATH ON CACHE BOOL "" FORCE)
set(WITH_WPILIB OFF CACHE BOOL "" FORCE)
set(WITH_TESTS OFF CACHE BOOL "" FORCE)
set(WITH_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WITH_GUI OFF CACHE BOOL "" FORCE)
set(WITH_NTCORE OFF CACHE BOOL "" FORCE)
set(WITH_SIMULATION_MODULES OFF CACHE BOOL "" FORCE)

add_subdirectory(allwpilib)

# and add our library

add_library(constrained_pnp
    src/constrained_pnp.cpp
)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(
  localizer_test
  test/Test_Pose.cpp
)

target_link_libraries(
  localizer_test
  GTest::gtest_main constrained_pnp
)

include(GoogleTest)
gtest_discover_tests(localizer_test)
