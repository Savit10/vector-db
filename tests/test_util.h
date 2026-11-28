// Minimal dependency-free test runner shared by all test executables.
#pragma once

#include <cstdio>
#include <exception>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "vdb/flat_index.h"

inline int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

inline void run(const std::string& name, const std::function<void()>& fn) {
    int before = g_failures;
    try {
        fn();
    } catch (const std::exception& e) {
        std::printf("    FAIL exception: %s\n", e.what());
        ++g_failures;
    }
    std::printf("[%s] %s\n", g_failures == before ? " OK " : "FAIL", name.c_str());
}

inline int report() {
    std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL PASSED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}

inline std::vector<float> random_vectors(std::size_t n, std::size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd;
    std::vector<float> v(n * dim);
    for (auto& x : v) x = nd(rng);
    return v;
}

inline std::vector<vdb::VectorId> sequential_ids(std::size_t n, vdb::VectorId start = 0) {
    std::vector<vdb::VectorId> ids(n);
    for (std::size_t i = 0; i < n; ++i) ids[i] = start + static_cast<vdb::VectorId>(i);
    return ids;
}
