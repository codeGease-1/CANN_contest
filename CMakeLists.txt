cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(rotary_pos_emb_custom LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 14)

option(ROPE_USE_CAST_ROUND "Use CAST_ROUND for FP16 output conversion" OFF)
option(ROPE_USE_POWER_INV_FREQ "Use Power for inverse-frequency generation" OFF)
option(ROPE_DEBUG_INTERMEDIATE "Overwrite FP32 output with intermediate diagnostics" OFF)

if(DEFINED NPU_ARCH)
    set(SOC_ARCH ${NPU_ARCH})
else()
    set(SOC_ARCH "dav-2201")
endif()

add_executable(rotary_pos_emb_custom main.asc)
add_executable(rotary_pos_emb_eval eval_main.asc)

target_link_libraries(rotary_pos_emb_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_link_libraries(rotary_pos_emb_eval PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(rotary_pos_emb_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(rotary_pos_emb_eval PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(rotary_pos_emb_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SOC_ARCH}>
)

target_compile_options(rotary_pos_emb_eval PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SOC_ARCH}>
)

if(ROPE_USE_CAST_ROUND)
    target_compile_definitions(rotary_pos_emb_eval PRIVATE ROPE_USE_CAST_ROUND=1)
endif()

if(ROPE_USE_POWER_INV_FREQ)
    target_compile_definitions(rotary_pos_emb_eval PRIVATE ROPE_USE_POWER_INV_FREQ=1)
endif()

if(ROPE_DEBUG_INTERMEDIATE)
    target_compile_definitions(rotary_pos_emb_eval PRIVATE ROPE_DEBUG_INTERMEDIATE=1)
endif()
