# CUDA rule override for clang-cl as the CUDA compiler. Registered as
# CMAKE_USER_MAKE_RULES_OVERRIDE_CUDA by Cmake/tools/cuda.cmake before CUDA is
# enabled, so CMake includes it right after its own CUDA platform rules.

# clang-cl writes every per-architecture ptxas output to the /Fo path, so with
# more than one --cuda-gpu-arch each cubin overwrites the previous one and
# fatbinary fails ("elf arch does not match user-specified arch"). -o names
# only the final object; the driver keeps the per-architecture files apart.
string(REPLACE "/Fo<OBJECT>" "-o <OBJECT>" CMAKE_CUDA_COMPILE_OBJECT "${CMAKE_CUDA_COMPILE_OBJECT}")
