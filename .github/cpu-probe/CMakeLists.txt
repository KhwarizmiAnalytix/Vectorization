cmake_minimum_required(VERSION 3.22)
project(VectorizationCpuProbe LANGUAGES CXX)

# Compile the probe for the baseline ISA so it can safely inspect older hosts.
try_run(probe_result probe_compiled "${CMAKE_CURRENT_BINARY_DIR}/probe"
        SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/avx512.cpp")
if(NOT probe_compiled OR NOT "${probe_result}" MATCHES "^[01]$")
  message(FATAL_ERROR "Could not determine AVX-512 CPU/OS support")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/supported.txt" "${probe_result}")
