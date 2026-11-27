# =============================================================================
# C++ runtime selection (macOS)
# =============================================================================
# The top-level project chooses the C++ runtime for the whole process.
# add_link_options() is inherited by every subdirectory added afterwards, so
# this file is included before any dependency: Logging, Memory, Scalars,
# LinearAlgebra and Parallel then all link the same libc++. Two libc++ copies
# in one process give two std::exception typeinfos, and a
# catch (const std::exception &) in one library misses exceptions thrown in
# another. A nested Vectorization inherits its host's choice instead.
# Only upstream/Homebrew LLVM ("Clang") ships its own libc++; AppleClang uses
# the SDK headers and the system libc++.
# =============================================================================
# Same top-level test as CMakeLists.txt: a nested Vectorization calls no project().
if(APPLE
   AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
   AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
)
  message(STATUS "Applying macOS LLVM linker options")

  if(DEFINED PROJECT_LLVM_INSTALL_PREFIX AND PROJECT_LLVM_INSTALL_PREFIX)
    set(_xsigma_llvm_prefix "${PROJECT_LLVM_INSTALL_PREFIX}")
  elseif(EXISTS "/opt/homebrew/opt/llvm/lib")
    set(_xsigma_llvm_prefix "/opt/homebrew/opt/llvm")
  elseif(EXISTS "/usr/local/opt/llvm/lib")
    set(_xsigma_llvm_prefix "/usr/local/opt/llvm")
  else()
    set(_xsigma_llvm_prefix "/opt/homebrew/opt/llvm")
  endif()

  set(LLVM_LINK_FLAGS
      -L${_xsigma_llvm_prefix}/lib/c++ -L${_xsigma_llvm_prefix}/lib
      -Wl,-rpath,${_xsigma_llvm_prefix}/lib/c++ -Wl,-rpath,${_xsigma_llvm_prefix}/lib
  )

  # Add each flag only if not already included
  foreach(flag IN LISTS LLVM_LINK_FLAGS)
    string(FIND "${CMAKE_EXE_LINKER_FLAGS}" "${flag}" flag_found)
    if(flag_found EQUAL -1)
      add_link_options(${flag})
    endif()
  endforeach()
endif()
