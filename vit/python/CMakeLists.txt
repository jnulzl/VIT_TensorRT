
if(PYTHON_API)
        add_subdirectory(${PROJECT_SOURCE_DIR}/../third_party/pybind11-2.12.0 build)

        file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
        set(src_for_python ${src_for_python} ${_tmp})
        pybind11_add_module(py${VIT_LIB_NAME} ${src_for_python})
        target_link_libraries(py${VIT_LIB_NAME} PRIVATE
                ${VIT_LIB_NAME})
        target_include_directories(py${VIT_LIB_NAME} PRIVATE
                ${PROJECT_SOURCE_DIR}/common
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${CMAKE_CURRENT_SOURCE_DIR}/..
        )
        target_compile_definitions(py${VIT_LIB_NAME} PRIVATE
                $<$<BOOL:${USE_${ENGINE_UPPER}}>:USE_${ENGINE_UPPER}>
                $<$<BOOL:${USE_${DEVICE_UPPER}}>:USE_${DEVICE_UPPER}>
                TORCH_API_INCLUDE_EXTENSION_H
        )

        target_compile_options(py${VIT_LIB_NAME} PRIVATE -DMODEL_NAME=py${VIT_LIB_NAME})

        ##############libtorch##############
        if(NOT LIBTORCH_DIR)
                message(FATAL_ERROR "LIBTORCH_DIR is not set")
        endif()
        message(STATUS "LIBTORCH_DIR :  " ${LIBTORCH_DIR})
        set(LIBTORCH_INCLUDE
                ${LIBTORCH_DIR}/include
                ${LIBTORCH_DIR}/include/torch
                ${LIBTORCH_DIR}/include/torch/csrc/api/include
        )
        set(LIBTORCH_LIB_PATH
                ${LIBTORCH_DIR}/lib
                ${LIBTORCH_DIR}/lib64
                ${LIBTORCH_DIR}/lib/x86_64-linux-gnu
        )
        target_include_directories(py${VIT_LIB_NAME} PRIVATE ${LIBTORCH_INCLUDE})
        target_link_directories(py${VIT_LIB_NAME} PRIVATE ${LIBTORCH_LIB_PATH})
        target_link_libraries(py${VIT_LIB_NAME} PRIVATE
                torch
                torch_cpu
                torch_cuda
                torch_python
        )
endif()
