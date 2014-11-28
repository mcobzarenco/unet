include(ExternalProject)

ExternalProject_Add(
    GTest
    URL http://googletest.googlecode.com/files/gtest-1.7.0.zip
    TIMEOUT 30
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(GTest binary_dir)
ExternalProject_Get_Property(GTest source_dir)
include_directories(SYSTEM ${source_dir}/include)
link_directories(${binary_dir})
enable_testing(true)

add_executable(
  test_all
    test_all.cpp
    test_utilities.cpp
    test_objectives.cpp
    test_mlp.cpp
    ${source_dir}
)
target_link_libraries(
  test_all
    pthread
    gtest
    ${CMAKE_SOURCE_DIR}/lib/libstan.a
    ${Boost_LIBRARIES}
    ${GLOG_LIBRARY}
)

add_custom_target(
   test test_all 1>&2
   DEPENDS test_all GTest
)
