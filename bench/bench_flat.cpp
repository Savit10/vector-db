// Usage: ./bench_flat [n=100000] [dim=128] [queries=100] [k=10]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "vdb/flat_index.h"

int main(int argc, char** argv) {
    std::size_t n = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 100000;
    std::size_t dim = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 128;
    std::size_t nq = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 100;
    std::size_t k = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 10;

    std::mt19937 rng(0);
    std::normal_distribution<float> nd;
    std::vector<float> data(n * dim), queries(nq * dim);
    for (auto& x : data) x = nd(rng);
    for (auto& x : queries) x = nd(rng);
    std::vector<vdb::VectorId> ids(n);
    for (std::size_t i = 0; i < n; ++i) ids[i] = static_cast<vdb::VectorId>(i);

    vdb::FlatIndex idx(dim);
    idx.add(data.data(), ids);

    auto t0 = std::chrono::steady_clock::now();
    std::size_t sink = 0;
    for (std::size_t q = 0; q < nq; ++q) sink += idx.search(&queries[q * dim], k).size();
    auto t1 = std::chrono::steady_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    double gflops = 3.0 * n * dim * nq / secs / 1e9; // sub, mul, add per element
    std::printf("n=%zu dim=%zu queries=%zu k=%zu\n", n, dim, nq, k);
    std::printf("%.1f QPS | %.3f ms/query | ~%.2f GFLOP/s (sink=%zu)\n", nq / secs,
                1e3 * secs / nq, gflops, sink);
}
