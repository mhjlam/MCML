# MCML 3.0.0 CMake Build System
# Author: M.H.J. Lam
# Date: 2025

cmake_minimum_required(VERSION 3.20)
project(MCML VERSION 3.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Wconversion -Wsign-conversion")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Wconversion -Wsign-conversion")
    # Suppress specific warnings that are too noisy
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-invalid-utf8 -Wno-deprecated-declarations")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /EHsc")
    # Disable specific MSVC warnings that are too noisy
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996")  # Disable deprecation warnings
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od /DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
endif()

# Define source directories - adjust paths based on build location
# When using -S . -B output, CMAKE_SOURCE_DIR is build/ and CMAKE_BINARY_DIR is build/output/
set(MCML_SRC_DIR ${CMAKE_SOURCE_DIR}/../src/mcml)
set(CONV_SRC_DIR ${CMAKE_SOURCE_DIR}/../src/conv)
set(BIN_DIR ${CMAKE_SOURCE_DIR}/../bin)  # Always relative to the source dir (build/)
set(SAMPLE_DIR ${CMAKE_SOURCE_DIR}/../sample)

# Include directories
include_directories(${MCML_SRC_DIR})
include_directories(${CONV_SRC_DIR})

# Create output directories
file(MAKE_DIRECTORY ${BIN_DIR})

# Define source files
set(MCML_SOURCES
    ${MCML_SRC_DIR}/main.cpp
    ${MCML_SRC_DIR}/reader.cpp
    ${MCML_SRC_DIR}/reader_cin.cpp
    ${MCML_SRC_DIR}/reader_util.cpp
    ${MCML_SRC_DIR}/simulator.cpp
    ${MCML_SRC_DIR}/tracer.cpp
    ${MCML_SRC_DIR}/writer.cpp
)

# Define header files (for IDE integration)
set(MCML_HEADERS
    ${MCML_SRC_DIR}/mcml.hpp
    ${MCML_SRC_DIR}/reader.hpp
    ${MCML_SRC_DIR}/reader.tpp
    ${MCML_SRC_DIR}/reader_cin.hpp
    ${MCML_SRC_DIR}/reader_cin.tpp
    ${MCML_SRC_DIR}/simulator.hpp
    ${MCML_SRC_DIR}/tracer.hpp
    ${MCML_SRC_DIR}/writer.hpp
    ${MCML_SRC_DIR}/observer.hpp
    ${MCML_SRC_DIR}/random.hpp
    ${MCML_SRC_DIR}/timer.hpp
    ${MCML_SRC_DIR}/exception.hpp
)

# Define CONV source files
set(CONV_SOURCES
    ${CONV_SRC_DIR}/main.cpp
    ${CONV_SRC_DIR}/beam_profile.cpp
    ${CONV_SRC_DIR}/conv.cpp
    ${CONV_SRC_DIR}/convolution_engine.cpp
    ${CONV_SRC_DIR}/mcml_reader.cpp
    ${CONV_SRC_DIR}/math_utils.cpp
    ${CONV_SRC_DIR}/processor.cpp
)

# Define CONV header files
set(CONV_HEADERS
    ${CONV_SRC_DIR}/conv.hpp
    ${CONV_SRC_DIR}/matrix.hpp
    ${CONV_SRC_DIR}/math_utils.hpp
    ${CONV_SRC_DIR}/beam_profile.hpp
    ${CONV_SRC_DIR}/binary_tree.hpp
    ${CONV_SRC_DIR}/mcml_reader.hpp
    ${CONV_SRC_DIR}/convolution_engine.hpp
    ${CONV_SRC_DIR}/processor.hpp
)

# Create executables
add_executable(mcml ${MCML_SOURCES} ${MCML_HEADERS})
add_executable(conv ${CONV_SOURCES} ${CONV_HEADERS})

# Set output directories for both executables and intermediate files
set_target_properties(mcml PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BIN_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BIN_DIR}
    # Intermediate files (object files) go to output directory
    ARCHIVE_OUTPUT_DIRECTORY ${BUILD_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${BUILD_OUTPUT_DIR}
)

set_target_properties(conv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BIN_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BIN_DIR}
    # Intermediate files (object files) go to output directory
    ARCHIVE_OUTPUT_DIRECTORY ${BUILD_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${BUILD_OUTPUT_DIR}
)

# Copy sample files to binary directory (post-build step)
add_custom_command(TARGET mcml POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
    ${SAMPLE_DIR} ${BIN_DIR}/sample
    COMMENT "Copying sample files to binary directory"
)

# Print build information
message(STATUS "MCML 3.0.0 Build Configuration:")
message(STATUS "  Platform:     ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler:     ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build Type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Source Dir:   ${SRC_DIR}")
message(STATUS "  Binary Dir:   ${BIN_DIR}")
message(STATUS "  Build Dir:    ${CMAKE_BINARY_DIR}")
message(STATUS "  Flags:        ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")

# Optional: Create install targets
install(TARGETS mcml conv
    RUNTIME DESTINATION bin
    COMPONENT runtime
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/../sample/
    DESTINATION bin/sample
    COMPONENT samples
)

# Optional: Create package configuration
set(CPACK_PACKAGE_NAME "MCML")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Monte Carlo Multi-Layered photon transport simulator")
set(CPACK_PACKAGE_VENDOR "M.H.J. Lam")
include(CPack)
