// Minimal dependency-free test runner. Swap for GoogleTest/Catch2 later if you want.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "vdb/flat_index.h"

static int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

static void run(const std::string& name, const std::function<void()>& fn) {
    int before = g_failures;
    try {
        fn();
    } catch (const std::exception& e) {
        std::printf("    FAIL exception: %s\n", e.what());
        ++g_failures;
    }
    std::printf("[%s] %s\n", g_failures == before ? " OK " : "FAIL", name.c_str());
}

using namespace vdb;

int main() {
    run("distance: l2 and inner product", [] {
        float a[] = {1, 2, 3}, b[] = {4, 6, 3};
        CHECK(l2_sq(a, b, 3) == 25.0f);
        CHECK(inner_product(a, b, 3) == 25.0f);
        CHECK(distance(Metric::InnerProduct, a, b, 3) == -25.0f);
    });

    run("search: exact nearest neighbors on a tiny 1-D set", [] {
        FlatIndex idx(1);
        float data[] = {0, 10, 3, 7, 1};
        idx.add(data, {100, 101, 102, 103, 104});
        float q[] = {2};
        auto res = idx.search(q, 3);
        CHECK(res.size() == 3);
        if (res.size() == 3) {
            // distances: 104 -> 1, 102 -> 1, 100 -> 4; ties may come in either order
            CHECK((res[0].id == 104 || res[0].id == 102));
            CHECK((res[1].id == 104 || res[1].id == 102));
            CHECK(res[2].id == 100);
            CHECK(res[2].distance == 4.0f);
        }
    });

    run("search: edge cases (k=0, k>n, empty)", [] {
        FlatIndex empty(4);
        float q[] = {0, 0, 0, 0};
        CHECK(empty.search(q, 5).empty());

        FlatIndex idx(4);
        float data[] = {1, 1, 1, 1, 2, 2, 2, 2};
        idx.add(data, {1, 2});
        CHECK(idx.search(q, 0).empty());
        CHECK(idx.search(q, 10).size() == 2);
    });

    run("search: matches full sort on random data", [] {
        const std::size_t n = 2000, dim = 32, k = 10;
        std::mt19937 rng(42);
        std::normal_distribution<float> nd;
        std::vector<float> data(n * dim);
        for (auto& x : data) x = nd(rng);
        std::vector<VectorId> ids(n);
        for (std::size_t i = 0; i < n; ++i) ids[i] = static_cast<VectorId>(i);

        FlatIndex idx(dim);
        idx.add(data.data(), ids);

        std::vector<float> q(dim);
        for (auto& x : q) x = nd(rng);

        std::vector<std::pair<float, VectorId>> all;
        for (std::size_t i = 0; i < n; ++i)
            all.push_back({l2_sq(q.data(), &data[i * dim], dim), ids[i]});
        std::sort(all.begin(), all.end());

        auto res = idx.search(q.data(), k);
        CHECK(res.size() == k);
        for (std::size_t i = 0; i < std::min(k, res.size()); ++i) {
            CHECK(res[i].id == all[i].second);
            CHECK(std::fabs(res[i].distance - all[i].first) < 1e-4f);
        }
    });

    run("search: inner product metric picks largest dot product", [] {
        FlatIndex idx(2, Metric::InnerProduct);
        float data[] = {1, 0, 0, 1, 5, 5};
        idx.add(data, {1, 2, 3});
        float q[] = {1, 1};
        auto res = idx.search(q, 1);
        CHECK(res.size() == 1 && res[0].id == 3);
    });

    std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL PASSED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
