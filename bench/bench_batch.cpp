// Compares per-query search() against search_batch() at several batch sizes.
// Usage: ./bench_batch [n=100000] [dim=128] [queries=256] [k=10]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "vdb/flat_index.h"

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    std::size_t n = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 100000;
    std::size_t dim = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 128;
    std::size_t nq = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 256;
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
    std::printf("n=%zu dim=%zu queries=%zu k=%zu\n\n", n, dim, nq, k);

    // Baseline: one query at a time.
    std::vector<std::vector<vdb::SearchResult>> ref(nq);
    auto t0 = Clock::now();
    for (std::size_t q = 0; q < nq; ++q) ref[q] = idx.search(&queries[q * dim], k);
    double base = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("%-14s %10.1f QPS  %8.3f ms/query\n", "search()", nq / base, 1e3 * base / nq);

    for (std::size_t bs : {1, 8, 32, 64, 128, 256}) {
        if (bs > nq) break;
        std::size_t hits = 0;
        auto t1 = Clock::now();
        for (std::size_t start = 0; start < nq; start += bs) {
            std::size_t cnt = std::min(bs, nq - start);
            auto res = idx.search_batch(&queries[start * dim], cnt, k);
            for (std::size_t q = 0; q < cnt && q < res.size(); ++q)
                for (auto& r : res[q])
                    for (auto& e : ref[start + q]) hits += (r.id == e.id);
        }
        double secs = std::chrono::duration<double>(Clock::now() - t1).count();
        std::printf("batch=%-8zu %10.1f QPS  %8.3f ms/query  %5.2fx  recall=%.3f\n", bs,
                    nq / secs, 1e3 * secs / nq, base / secs,
                    static_cast<double>(hits) / static_cast<double>(nq * k));
    }
}
