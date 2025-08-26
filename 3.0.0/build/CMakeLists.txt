# MCML 3.0.0 CMake Build System
# Author: M.H.J. Lam
# Date: 2025

cmake_minimum_required(VERSION 3.20)
project(MCML VERSION 3.0.0 LANGUAGES CXX)

# Force CMake to put all its generated files in the output subdirectory
if(NOT DEFINED CMAKE_BINARY_DIR OR CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/output)
    set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_BINARY_DIR})
endif()

# Ensure output directory exists
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR})

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
set(SRC_DIR ${CMAKE_SOURCE_DIR}/../src/mcml)
set(BIN_DIR ${CMAKE_SOURCE_DIR}/../bin)  # Always relative to the source dir (build/)
set(SAMPLE_DIR ${CMAKE_SOURCE_DIR}/../sample)

# Include directories
include_directories(${SRC_DIR})

# Create output directories
file(MAKE_DIRECTORY ${BIN_DIR})

# Define source files
set(MCML_SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/reader.cpp
    ${SRC_DIR}/reader_cin.cpp
    ${SRC_DIR}/reader_util.cpp
    ${SRC_DIR}/simulator.cpp
    ${SRC_DIR}/tracer.cpp
    ${SRC_DIR}/writer.cpp
)

# Define header files (for IDE integration)
set(MCML_HEADERS
    ${SRC_DIR}/mcml.hpp
    ${SRC_DIR}/reader.hpp
    ${SRC_DIR}/reader.tpp
    ${SRC_DIR}/reader_cin.hpp
    ${SRC_DIR}/reader_cin.tpp
    ${SRC_DIR}/simulator.hpp
    ${SRC_DIR}/tracer.hpp
    ${SRC_DIR}/writer.hpp
    ${SRC_DIR}/observer.hpp
    ${SRC_DIR}/random.hpp
    ${SRC_DIR}/timer.hpp
    ${SRC_DIR}/exception.hpp
)

# Create executable
add_executable(mcml ${MCML_SOURCES} ${MCML_HEADERS})

# Set output directories for both executable and intermediate files
set_target_properties(mcml PROPERTIES
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
install(TARGETS mcml
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
