// Empty stub for CUDA's legacy texture_fetch_functions.h (bound-texture-reference
// API), removed in CUDA 13 in favor of texture objects / texture_indirect_functions.h
// (which CUDA 13 still ships). Clang's own __clang_cuda_runtime_wrapper.h
// unconditionally #includes "texture_fetch_functions.h" for every CUDA translation
// unit regardless of toolkit version, so on CUDA 13 that include fails with
// "file not found" until a Clang release stops assuming the header exists. None of
// this project's CUDA sources call the legacy bound-texture API, so this directory is
// added to the include path (only when the real header is absent) purely to satisfy
// that #include with a no-op.
