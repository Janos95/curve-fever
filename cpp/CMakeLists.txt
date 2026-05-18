cmake_minimum_required(VERSION 3.24)
project(curve_fever_webgpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(DAWN_ROOT "/Users/janosmeny/Projects/dawn-7069" CACHE PATH "Path to the local Dawn checkout")
set(DAWN_BUILD_DIR "${DAWN_ROOT}/out/Release" CACHE PATH "Path to the local Dawn build directory")

add_library(curve_engine INTERFACE)
target_include_directories(curve_engine INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_sources(curve_engine INTERFACE
  curve_engine.hpp
  curve_shaders.hpp
)

if(EMSCRIPTEN)
add_executable(curve_web
  curve_webgpu_web.cpp
)

target_link_libraries(curve_web PRIVATE
  curve_engine
)

set_target_properties(curve_web PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../web/dist"
  OUTPUT_NAME "curve_web"
  SUFFIX ".js"
)

target_link_options(curve_web PRIVATE
  "--use-port=emdawnwebgpu"
  "-sALLOW_MEMORY_GROWTH=1"
  "-sASSERTIONS=1"
  "-sENVIRONMENT=web"
  "-sEXIT_RUNTIME=0"
  "-sEXPORTED_FUNCTIONS=['_main','_curve_set_turn','_curve_reset_game']"
  "-sEXPORTED_RUNTIME_METHODS=['ccall']"
)

target_compile_options(curve_web PRIVATE
  "--use-port=emdawnwebgpu"
)

add_custom_command(TARGET curve_web POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/../web/index.html"
    "${CMAKE_SOURCE_DIR}/../web/dist/index.html"
  COMMAND ${CMAKE_COMMAND} -E touch
    "${CMAKE_SOURCE_DIR}/../web/dist/.nojekyll"
)
else()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['include'])"
  OUTPUT_VARIABLE Python3_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
  OUTPUT_VARIABLE Python3_EXTENSION_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_executable(curve_webgpu_demo
  curve_webgpu_demo.cpp
)

add_executable(curve_webgpu_benchmark
  curve_webgpu_benchmark.cpp
)

add_library(_curve_webgpu_native MODULE
  curve_webgpu_python.cpp
)

target_include_directories(curve_webgpu_demo PRIVATE
  "${DAWN_ROOT}/include"
  "${DAWN_BUILD_DIR}/gen/include"
)

target_include_directories(curve_webgpu_benchmark PRIVATE
  "${DAWN_ROOT}/include"
  "${DAWN_BUILD_DIR}/gen/include"
)

target_include_directories(_curve_webgpu_native PRIVATE
  "${DAWN_ROOT}/include"
  "${DAWN_BUILD_DIR}/gen/include"
  "${Python3_INCLUDE_DIR}"
)

target_link_libraries(curve_webgpu_demo PRIVATE
  curve_engine
  "${DAWN_BUILD_DIR}/src/dawn/native/libwebgpu_dawn.dylib"
)

target_link_libraries(curve_webgpu_benchmark PRIVATE
  curve_engine
  "${DAWN_BUILD_DIR}/src/dawn/native/libwebgpu_dawn.dylib"
)

target_link_libraries(_curve_webgpu_native PRIVATE
  curve_engine
  "${DAWN_BUILD_DIR}/src/dawn/native/libwebgpu_dawn_tinygrad.dylib"
)

set_target_properties(_curve_webgpu_native PROPERTIES
  PREFIX ""
  SUFFIX "${Python3_EXTENSION_SUFFIX}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../src/curve_fever"
)

target_link_options(curve_webgpu_demo PRIVATE
  "-Wl,-rpath,${DAWN_BUILD_DIR}/src/dawn/native"
)

target_link_options(curve_webgpu_benchmark PRIVATE
  "-Wl,-rpath,${DAWN_BUILD_DIR}/src/dawn/native"
)

target_link_options(_curve_webgpu_native PRIVATE
  "-Wl,-rpath,${DAWN_BUILD_DIR}/src/dawn/native"
)

if(APPLE)
target_link_options(_curve_webgpu_native PRIVATE
  "-undefined"
  "dynamic_lookup"
)
add_custom_command(TARGET _curve_webgpu_native POST_BUILD
  COMMAND install_name_tool
    -change "out/Release/src/dawn/native/libwebgpu_dawn_tinygrad.dylib"
            "${DAWN_BUILD_DIR}/src/dawn/native/libwebgpu_dawn_tinygrad.dylib"
            "$<TARGET_FILE:_curve_webgpu_native>"
)
endif()
endif()
