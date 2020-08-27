cmake_minimum_required(VERSION 3.10)
project(DDHDG)

set(DDHDG_VERSION_MAJOR 0)
set(DDHDG_VERSION_MINOR 1)

FIND_PACKAGE(deal.II QUIET
        HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
        )
IF (NOT ${deal.II_FOUND})
    MESSAGE(FATAL_ERROR "\n"
            "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
            "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
            "or set an environment variable \"DEAL_II_DIR\" that contains this path."
            )
ENDIF ()
IF (NOT ${DEAL_II_VERSION_MAJOR} EQUAL 9)
    MESSAGE(FATAL_ERROR "\n"
            "*** Your installation of deal.II is not compatible with this software. ***\n"
            "This software requires deal.II 9. The version of your current installation "
            "of deal.II is ${DEAL_II_VERSION} and is located in ${DEAL_II_PATH}. If this "
            "is not the version you intended to use, you may want to either pass a flag "
            "-DDEAL_II_DIR=/path/to/deal.II to cmake or set an environment variable "
            "\"DEAL_II_DIR\" that contains this path."
            )
ENDIF ()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

option(WITH_PYTHON_INTERFACE "Compile also the python interface" OFF)
option(WITH_TESTS "Compile the tests" ON)
option(SERIALIZE_OPENBLAS "Serialize matrix inversion (use with non-threadsafe openblas)" ON)

find_package(Boost REQUIRED COMPONENTS log log_setup)

# Set a flag that indicates that the code will be linked
# against a dynamically loaded library (this is the most
# common case for linux)
add_compile_definitions(BOOST_LOG_DYN_LINK)

if (SERIALIZE_OPENBLAS)
    add_compile_definitions(WITH_MUTEX)
endif ()

include_directories("include")
add_subdirectory("src")

if (WITH_TESTS)
    find_package(GTest)
    if (GTest_FOUND)
        add_subdirectory("tests")
    else ()
        MESSAGE(WARNING "GTest library not found. Tests will not be compiled!")
    endif ()
endif ()

add_executable(MAIN main.cpp)

target_link_libraries(MAIN DDHDG)

DEAL_II_SETUP_TARGET(MAIN)
