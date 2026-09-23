# Host overlay: build Logging product sources only (ThirdParty/Logging).
# Do not add_subdirectory Logging's CMakeLists — that tree also compiles
# Logging's private backends. XSigma links Logging::Logging.

function(xsigma_add_logging_product)
    if(TARGET Logging::Logging)
        return()
    endif()

    set(_logging_src "${XSIGMA_THIRDPARTY_DIR}/Logging")
    if(NOT EXISTS "${_logging_src}/logging/logging.h")
        message(FATAL_ERROR "ThirdParty/Logging is missing. Initialize the submodule:\n"
                            "  git submodule update --init ThirdParty/Logging")
    endif()

    if(NOT TARGET Fmt::fmt)
        set(FMT_TEST OFF CACHE BOOL "Disable fmt tests" FORCE)
        set(FMT_DOC OFF CACHE BOOL "Disable fmt documentation" FORCE)
        set(FMT_INSTALL OFF CACHE BOOL "Disable fmt install" FORCE)
        set(FMT_HEADER_ONLY ON CACHE BOOL "Use header-only fmt library" FORCE)
        add_third_party_library(
            fmt
            LIBRARY_TYPE STATIC
            SOURCE_DIR "${XSIGMA_THIRDPARTY_DIR}/fmt"
            ALIASES "fmt::fmt-header-only"
            XSIGMA_TARGETS "Fmt::fmt=fmt::fmt-header-only|fmt::fmt|fmt"
            INCLUDE_DIR "${XSIGMA_THIRDPARTY_DIR}/fmt/include"
            AUTO_ADD_TO_DEPENDENCIES OFF
        )
    endif()

    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles(
        "
        #include <cxxabi.h>
        #include <cstdlib>
        int main() {
          int status = 0;
          char* result = abi::__cxa_demangle(\"_Z1fv\", nullptr, nullptr, &status);
          std::free(result);
          return status;
        }
        "
        XSIGMA_LOGGING_HAS_CXA_DEMANGLE
    )
    mark_as_advanced(XSIGMA_LOGGING_HAS_CXA_DEMANGLE)

    file(GLOB_RECURSE _logging_sources LIST_DIRECTORIES false
         "${_logging_src}/logging/*.cpp")
    file(GLOB_RECURSE _logging_headers LIST_DIRECTORIES false
         "${_logging_src}/logging/*.h" "${_logging_src}/logging/*.hxx")
    list(FILTER _logging_sources EXCLUDE REGEX "/Testing/")
    list(FILTER _logging_headers EXCLUDE REGEX "/Testing/")
    list(FILTER _logging_sources EXCLUDE REGEX "/Logging/ThirdParty/")
    list(FILTER _logging_headers EXCLUDE REGEX "/Logging/ThirdParty/")

    add_library(Logging EXCLUDE_FROM_ALL ${_logging_sources} ${_logging_headers})
    add_library(Logging::Logging ALIAS Logging)
    set_target_properties(
        Logging PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "ThirdParty"
    )

    if(BUILD_SHARED_LIBS)
        target_compile_definitions(Logging PUBLIC LOGGING_SHARED_DEFINE PRIVATE LOGGING_BUILDING_DLL)
    else()
        target_compile_definitions(Logging PUBLIC LOGGING_STATIC_DEFINE)
    endif()

    target_compile_definitions(
        Logging
        PUBLIC
            LOGGING_HAS_NATIVE=1
            LOGGING_HAS_LOGURU=0
            LOGGING_HAS_GLOG=0
            LOGGING_HAS_SPDLOG=0
            LOGGING_HAS_MAGICENUM=0
            LOGGING_PORTABLE_FLOAT_FORMAT=0
            LOGGING_FORMAT_USE_STD=0
            LOGGING_DEFAULT_EXCEPTION_MODE_LOG_FATAL=0
    )
    if(XSIGMA_LOGGING_HAS_CXA_DEMANGLE)
        target_compile_definitions(Logging PUBLIC LOGGING_HAS_CXA_DEMANGLE=1)
    else()
        target_compile_definitions(Logging PUBLIC LOGGING_HAS_CXA_DEMANGLE=0)
    endif()

    target_include_directories(
        Logging
        PUBLIC
            $<BUILD_INTERFACE:${_logging_src}>
            $<BUILD_INTERFACE:${_logging_src}/logging/logger>
    )

    if(TARGET Fmt::fmt)
        target_link_libraries(Logging PUBLIC Fmt::fmt)
    endif()

    if(WIN32)
        target_link_libraries(Logging PRIVATE dbghelp)
    endif()

    if(NOT MSVC)
        target_compile_options(Logging PRIVATE -include cstdlib)
    endif()
endfunction()
