cmake_minimum_required(VERSION 3.10)
project(DDHDG)

set (DDHDG_VERSION_MAJOR 0)
set (DDHDG_VERSION_MINOR 1)

FIND_PACKAGE(deal.II 9.1.0 QUIET
    HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
    )
IF(NOT ${deal.II_FOUND})
    MESSAGE(FATAL_ERROR "\n"
        "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
        "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
        "or set an environment variable \"DEAL_II_DIR\" that contains this path."
        )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

option(WITH_PYTHON_INTERFACE "Compile also the python interface" OFF)

include_directories("include")
add_subdirectory("src")

add_executable(MAIN main.cpp)

target_link_libraries(MAIN DDHDG)

DEAL_II_SETUP_TARGET(MAIN)
