cmake_minimum_required(VERSION 3.10)

project(Conv)

set(MKLDNN_ENABLE_CONCURRENT_EXEC ON)

add_subdirectory(mkl-dnn)
IF(NOT TARGET dnnl)
  MESSAGE("dnnl target not found!!")
  RETURN(1)
ENDIF(NOT TARGET dnnl)
# This is only used by the bernoulli_generate util which is used by dropout
# add_definitions(-DIDEEP_USE_MKL)
target_compile_definitions(dnnl PRIVATE) # -DUSE_MKL -DUSE_CBLAS)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/mkl-dnn/include
    # Needed to get the configured (configure_file()) version of some DNNL .h files
    ${CMAKE_CURRENT_BINARY_DIR}/mkl-dnn/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /opt/intel/oneapi/mkl/latest/include
)

add_library(ideep ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(ideep PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(ideep PUBLIC dnnl) # mkl-dnn doesn't work here

add_executable(conv conv.cpp)
target_link_libraries(conv PUBLIC ideep)
