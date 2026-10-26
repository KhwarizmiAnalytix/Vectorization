# =============================================================================
# Vectorization — Validation and Checks Module
# =============================================================================
# Compiler version minimums run on every configure (before include_guard) because
# the floor depends on the selected C++ standard.  A compiler that clears only the
# C++17 floor fails later with an unrecognised flag rather than this FATAL_ERROR.
# All other checks are guarded by include_guard(DIRECTORY) so they run only once.

if(DEFINED VECTORIZATION_CXX_STANDARD AND NOT VECTORIZATION_CXX_STANDARD STREQUAL "")
    set(_vec_checks_std "${VECTORIZATION_CXX_STANDARD}")
elseif(DEFINED CMAKE_CXX_STANDARD AND NOT CMAKE_CXX_STANDARD STREQUAL "")
    set(_vec_checks_std "${CMAKE_CXX_STANDARD}")
else()
    set(_vec_checks_std 20)
endif()

if(_vec_checks_std STREQUAL "23")
    # https://gcc.gnu.org/gcc-11/changes.html
    set(_vec_checks_min_gcc        11.0)
    # https://releases.llvm.org/17.0.0/tools/clang/docs/ReleaseNotes.html
    set(_vec_checks_min_clang      17.0)
    # Apple has not published an explicit minimum for -std=c++23; best available estimate.
    set(_vec_checks_min_appleclang 15.0)
    # /std:c++23preview from VS 17.0 (19.30); stable /std:c++23 in very recent toolsets.
    # https://learn.microsoft.com/en-us/cpp/build/reference/std-specify-language-standard-version
    set(_vec_checks_min_msvc       19.34)
elseif(_vec_checks_std STREQUAL "20")
    # https://gcc.gnu.org/gcc-10/changes.html
    set(_vec_checks_min_gcc        10.0)
    # https://releases.llvm.org/11.0.0/tools/clang/docs/ReleaseNotes.html
    set(_vec_checks_min_clang      11.0)
    # Xcode 14 (AppleClang 14) accepts -std=c++20; 13 is a conservative floor.
    set(_vec_checks_min_appleclang 13.0)
    # https://devblogs.microsoft.com/cppblog/msvc-cpp20-and-the-std-cpp20-switch/
    set(_vec_checks_min_msvc       19.29)
else()
    # C++17 baseline (or earlier).  Floor is set by <filesystem> (used unconditionally by
    # Logging's string_util.h): GCC < 9.1 / Clang < 9.0 only have
    # <experimental/filesystem> and need an extra link flag this build does not add.
    # https://gcc.gnu.org/gcc-9/changes.html
    set(_vec_checks_min_gcc        9.1)
    # https://releases.llvm.org/9.0.0/projects/libcxx/docs/UsingLibcxx.html
    set(_vec_checks_min_clang      9.0)
    # AppleClang 11 / Xcode 11 is the first to ship the non-experimental <filesystem>.
    set(_vec_checks_min_appleclang 11.0)
    # MSVC 19.14 / VS 2017 15.7 also supports /Zc:__cplusplus used by platform.cmake.
    # https://learn.microsoft.com/en-us/cpp/overview/what-s-new-for-cpp-2017
    set(_vec_checks_min_msvc       19.14)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS _vec_checks_min_gcc)
    message(
        FATAL_ERROR
            "GCC ${_vec_checks_min_gcc} or later is required for C++${_vec_checks_std}. "
            "Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS _vec_checks_min_clang)
    message(
        FATAL_ERROR
            "LLVM Clang ${_vec_checks_min_clang} or later is required for C++${_vec_checks_std}. "
            "Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS _vec_checks_min_appleclang)
    message(
        FATAL_ERROR
            "Apple Clang ${_vec_checks_min_appleclang} or later is required for C++${_vec_checks_std}. "
            "Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS _vec_checks_min_msvc)
    message(
        FATAL_ERROR
            "MSVC ${_vec_checks_min_msvc} or later is required for C++${_vec_checks_std}. "
            "Found: ${CMAKE_CXX_COMPILER_VERSION}"
    )
endif()

unset(_vec_checks_std)
unset(_vec_checks_min_gcc)
unset(_vec_checks_min_clang)
unset(_vec_checks_min_appleclang)
unset(_vec_checks_min_msvc)

include_guard(DIRECTORY)
include(CheckIncludeFileCXX)
include(CheckCXXSourceCompiles)
find_package(Threads REQUIRED)
if(WIN32)
    check_include_file_cxx("windows.h" VECTORIZATION_HAVE_WINDOWS_H)
    if(NOT VECTORIZATION_HAVE_WINDOWS_H)
        message(FATAL_ERROR "Vectorization requires windows.h on Windows.")
    endif()
elseif(UNIX)
    check_include_file_cxx("unistd.h"  VECTORIZATION_HAVE_UNISTD_H)
    check_include_file_cxx("pthread.h" VECTORIZATION_HAVE_PTHREAD_H)
    if(NOT VECTORIZATION_HAVE_UNISTD_H OR NOT VECTORIZATION_HAVE_PTHREAD_H)
        message(FATAL_ERROR "Vectorization requires unistd.h and pthread.h.")
    endif()
endif()
