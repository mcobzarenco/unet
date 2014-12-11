add_executable(
  unet
    unet.cpp
)
target_link_libraries(
  unet
    ${GLOG_LIBRARY}
    ${Boost_LIBRARIES}
)

if(MKL_ROOT)
    target_link_libraries(unet pthread m)
endif()
