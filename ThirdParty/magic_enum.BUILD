# =============================================================================
# magic_enum Library BUILD Configuration
# =============================================================================
# Static reflection for enums (header-only)
# Equivalent to ThirdParty/magic_enum
# =============================================================================

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "magic_enum",
    hdrs = glob(
        [
            "include/magic_enum/**/*.hpp",
            "include/magic_enum.hpp",
        ],
        allow_empty = True,
    ),
    includes = ["include"],
    strip_include_prefix = "include",
)
