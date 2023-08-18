file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(src_all ${src_all} ${_tmp})

if(USE_${ENGINE_UPPER})
    file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/${ENGINE_LOWER}/*.cpp)
    set(src_all ${src_all} ${_tmp})
#    file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/${ENGINE_LOWER}/*.cu)
#    set(src_all ${src_all} ${_tmp})
endif()

if(USE_TENSORRT)
    set(src_all ${src_all} ${TRT_SAMPLES_COMMON_HOME}/logger.cpp)
endif()

if (BUILD_SHARED_LIBS)
    add_library(${CLS_LIB_NAME} SHARED ${src_all})
    set_target_properties(${CLS_LIB_NAME} PROPERTIES VERSION ${LIB_MAJOR_VERSION}.${LIB_MINOR_VERSION}.${GIT_HASH} SOVERSION ${LIB_MAJOR_VERSION})
else()
    set(src_all ${src_all})
    add_library(${CLS_LIB_NAME} STATIC ${src_all})
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")
set_target_properties(${CLS_LIB_NAME} PROPERTIES  CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${CLS_LIB_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        $<$<BOOL:${USE_${ENGINE_UPPER}}>:${CMAKE_CURRENT_SOURCE_DIR}/${ENGINE_LOWER}>
        )

target_compile_definitions(${CLS_LIB_NAME} PRIVATE
        $<$<BOOL:${USE_${ENGINE_UPPER}}>:USE_${ENGINE_UPPER}>
        $<$<BOOL:${USE_${DEVICE_UPPER}}>:USE_${DEVICE_UPPER}>
        )

target_link_options(${CLS_LIB_NAME} PRIVATE
        -Wl,--no-undefined
        -Wl,--no-allow-shlib-undefined
        )

if(USE_CUDA)
    target_link_options(
            ${CLS_LIB_NAME} PUBLIC
            "-Wl,-rpath-link,${CUDNN_HOME}/lib64"
            "-Wl,-rpath-link,${TRT_HOME}/lib64"
            "-Wl,-rpath-link,${CUDA_HOME}/lib64"
    )

    target_link_libraries(${CLS_LIB_NAME} PUBLIC
            cudart
            nvinfer
            nvparsers
            nvonnxparser
            nvinfer_plugin
            pthread
            dl
            nppig
            nppidei
            nppial
            )
endif()
