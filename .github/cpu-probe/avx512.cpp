#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>

int main() {
  int info[4];
  __cpuidex(info, 1, 0);
  // OSXSAVE, AVX, FMA, F16C, and enabled XMM/YMM/opmask/ZMM state.
  constexpr unsigned required_ecx = (1u << 27) | (1u << 28) | (1u << 12) | (1u << 29);
  if ((static_cast<unsigned>(info[2]) & required_ecx) != required_ecx ||
      (_xgetbv(0) & 0xe6) != 0xe6) {
    return 1;
  }
  __cpuidex(info, 7, 0);
  constexpr unsigned required_ebx = (1u << 16) | (1u << 17) | (1u << 28) |
                                    (1u << 30) | (1u << 31);
  return (static_cast<unsigned>(info[1]) & required_ebx) == required_ebx ? 0 : 1;
}
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
int main() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
                 __builtin_cpu_supports("avx512cd") && __builtin_cpu_supports("avx512bw") &&
                 __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("f16c") &&
                 __builtin_cpu_supports("fma")
             ? 0 : 1;
}
#else
int main() { return 1; }
#endif
