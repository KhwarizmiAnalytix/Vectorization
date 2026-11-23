# CUDA rule overrides for Clang (clang++ or clang-cl) as the CUDA compiler on
# Windows. Registered as CMAKE_USER_MAKE_RULES_OVERRIDE_CUDA by
# Cmake/dependencies.cmake. CMake includes this file right after its CUDA
# platform rules, in every directory scope that runs enable_language(CUDA)
# (Memory, Scalars, LinearAlgebra and this project), so the fixes below are
# not undone by a later enable_language(CUDA) in another scope.

# Memory forces Clang as the CUDA compiler (CMAKE_CUDA_COMPILER_FORCED). The
# platform rules then never map MSVC_RUNTIME_LIBRARY to CUDA compile flags,
# and generation fails in any directory with CUDA sources: "MSVC_RUNTIME_LIBRARY
# value 'MultiThreadedDLL' not known for this CUDA compiler". These are the
# flags CMake uses for Clang's GNU driver; clang-cl accepts them too.
set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded
    -D_MT -Xclang --dependent-lib=libcmt
)
set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL
    -D_DLL -D_MT -Xclang --dependent-lib=msvcrt
)
set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug
    -D_DEBUG -D_MT -Xclang --dependent-lib=libcmtd
)
set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL
    -D_DEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrtd
)

# clang-cl also writes every per-architecture ptxas output to the /Fo path, so
# with more than one --cuda-gpu-arch each cubin overwrites the previous one and
# fatbinary fails ("elf arch does not match user-specified arch"). -o names
# only the final object; the driver keeps the per-architecture files apart.
string(REPLACE "/Fo<OBJECT>" "-o <OBJECT>" CMAKE_CUDA_COMPILE_OBJECT "${CMAKE_CUDA_COMPILE_OBJECT}")
