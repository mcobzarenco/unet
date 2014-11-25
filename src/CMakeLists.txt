add_executable(
  unet
    unet.cpp
    mlp.hpp
    mlp.cpp
    utilities.hpp
    init.hpp
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
