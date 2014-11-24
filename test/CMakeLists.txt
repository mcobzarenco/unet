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
  test_unet
    test_utilities.cpp

    ${source_dir}
)
target_link_libraries(
  test_unet
    pthread
    gtest
    gtest_main
    ${CMAKE_SOURCE_DIR}/lib/libstan.a
    ${Boost_LIBRARIES}
    ${GLOG_LIBRARY}
)

add_custom_target(
   test test_unet --gtest_catch_exceptions=1 --gtest_color=yes 1>&2
   DEPENDS test_unet
)
