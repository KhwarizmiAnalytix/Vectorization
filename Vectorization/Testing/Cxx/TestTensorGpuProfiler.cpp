/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

/*
 * Profiles the fused Metal expression `z = z + exp(z) + sin(z)` through
 * profiler::session.
 */

#include "VectorizationTest.h"

#if VECTORIZATION_HAS_METAL && VECTORIZATION_HAS_PROFILER

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <profiler.h>

#include "terminals/tensor.h"

extern "C" int xsigma_metal_device_count();

namespace
{

using namespace vectorization;

constexpr const char* kWorkloadScope = "z_plus_exp_sin";
constexpr const char* kAssignScope   = "vectorization::tensor::assign";
constexpr const char* kKernelName    = "fused_float";
constexpr const char* kTrace         = "/tmp/xsigma_metal_z_exp_sin.json";
constexpr std::size_t kN             = 1U << 20;
constexpr int         kIters         = 16;

}  // namespace

VECTORIZATIONTEST(TensorGpuProfiler, metal_z_plus_exp_sin)
{
    if (xsigma_metal_device_count() <= 0)
    {
        GTEST_SKIP() << "No Metal device";
    }

    std::vector<float> host(kN);
    for (std::size_t i = 0; i < kN; ++i)
    {
        host[i] = static_cast<float>(static_cast<double>(i) / static_cast<double>(kN) * 0.5);
    }

    tensor<float> z(kN, device_enum::METAL);
    z.copy_from_host(host);
    // Compile the fused MSL kernel once so the timed region is dispatch, not JIT.
    z = z + ::exp(z) + ::sin(z);

    profiler::session_options opts;
    opts.gpu_tracing = true;
    opts.activities  = {profiler::activity::cpu, profiler::activity::metal};
    profiler::session session(opts);
    if (!session.start())
    {
        GTEST_SKIP() << "profiler::session failed to start";
    }

    {
        PROFILER_SCOPE(kWorkloadScope);
        for (int i = 0; i < kIters; ++i)
        {
            z = z + ::exp(z) + ::sin(z);
        }
    }

    ASSERT_TRUE(session.stop());

    const auto& events = session.events();
    if (events.empty())
    {
        GTEST_SKIP() << "instrumentation produced no CPU events in this environment";
    }

    std::cout << "\n=== profiler::session events (z = z + exp(z) + sin(z), N=" << kN
              << ", iters=" << kIters << ") ===\n";
    uint64_t assign_count = 0;
    uint64_t assign_ns    = 0;
    bool     saw_workload = false;
    for (const auto& event : events)
    {
        std::cout << "  " << event.name << "  duration_ns=" << event.duration_ns << "\n";
        if (event.name == kWorkloadScope)
        {
            saw_workload = true;
            EXPECT_GT(event.duration_ns, 0U);
        }
        if (event.name == kAssignScope)
        {
            ++assign_count;
            assign_ns += event.duration_ns;
        }
    }
    std::cout << std::flush;

    EXPECT_TRUE(saw_workload);
    EXPECT_EQ(assign_count, static_cast<uint64_t>(kIters));
    EXPECT_GT(assign_ns, 0U);

    auto hotspot = session.generate_hotspot_report();
    if (hotspot != nullptr)
    {
        std::cout << "\n=== hotspot table ===\n" << hotspot->table() << std::flush;
    }

    EXPECT_TRUE(session.write_trace(kTrace));
    std::cout << "Chrome trace: " << kTrace << "\n";

    const std::string chrome = session.generate_chrome_trace_json();
    if (chrome.empty())
    {
        GTEST_SKIP() << "native collector produced no Chrome trace";
    }
    EXPECT_NE(chrome.find(kKernelName), std::string::npos)
        << "expected fused_float in native Chrome trace";
    std::cout << std::flush;
}

#else

VECTORIZATIONTEST(TensorGpuProfiler, metal_z_plus_exp_sin)
{
    GTEST_SKIP() << "requires VECTORIZATION_HAS_METAL + Profiler";
}

#endif
