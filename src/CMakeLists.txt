add_executable(
  unet
    unet.cpp
)
target_link_libraries(
  unet
    ${CMAKE_SOURCE_DIR}/lib/libstan.a
    ${GLOG_LIBRARY}
    ${Boost_LIBRARIES}
)

if(MKL_ROOT)
    target_link_libraries(unet pthread m)
endif()
