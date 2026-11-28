// Tests for FlatIndex::search_batch. Expected signature:
//
//   std::vector<std::vector<SearchResult>>
//   search_batch(const float* queries, std::size_t nq, std::size_t k) const;
//
// `queries` is nq row-major vectors. result[q] must equal search(&queries[q * dim], k),
// up to small floating-point differences (the ||q||^2 - 2 q.x + ||x||^2 expansion
// rounds differently from computing (q - x)^2 directly).
#include <algorithm>
#include <cmath>
#include <set>

#include "test_util.h"

using namespace vdb;

// Compares batch results to per-query search(). Distances must match within a
// relative tolerance; IDs are compared as sets, because near-ties may swap order.
static void expect_matches_search(const FlatIndex& idx, const std::vector<float>& queries,
                                  std::size_t nq, std::size_t dim, std::size_t k) {
    auto batch = idx.search_batch(queries.data(), nq, k);
    CHECK(batch.size() == nq);
    if (batch.size() != nq) return;

    std::size_t id_matches = 0, id_total = 0;
    for (std::size_t q = 0; q < nq; ++q) {
        auto ref = idx.search(&queries[q * dim], k);
        CHECK(batch[q].size() == ref.size());
        if (batch[q].size() != ref.size()) continue;

        for (std::size_t i = 0; i < ref.size(); ++i) {
            float tol = 1e-3f * (1.0f + std::fabs(ref[i].distance));
            CHECK(std::fabs(batch[q][i].distance - ref[i].distance) <= tol);
            if (i > 0) CHECK(batch[q][i - 1].distance <= batch[q][i].distance);
        }
        std::set<VectorId> ref_ids;
        for (auto& r : ref) ref_ids.insert(r.id);
        for (auto& r : batch[q]) id_matches += ref_ids.count(r.id);
        id_total += ref.size();
    }
    // Allow a tiny number of swaps at near-ties.
    CHECK(id_total == 0 || id_matches >= id_total * 99 / 100);
}

int main() {
    run("L2: matches search() (dim not a multiple of 8, odd nq)", [] {
        const std::size_t n = 3000, dim = 67, nq = 37, k = 10;
        FlatIndex idx(dim);
        idx.add(random_vectors(n, dim, 1).data(), sequential_ids(n));
        expect_matches_search(idx, random_vectors(nq, dim, 2), nq, dim, k);
    });

    run("inner product: matches search()", [] {
        const std::size_t n = 3000, dim = 64, nq = 20, k = 5;
        FlatIndex idx(dim, Metric::InnerProduct);
        idx.add(random_vectors(n, dim, 3).data(), sequential_ids(n));
        expect_matches_search(idx, random_vectors(nq, dim, 4), nq, dim, k);
    });

    run("add() called twice, non-sequential ids (precomputed norms stay in sync)", [] {
        const std::size_t dim = 32, nq = 16, k = 8;
        FlatIndex idx(dim);
        idx.add(random_vectors(500, dim, 5).data(), sequential_ids(500, 1000));
        idx.add(random_vectors(700, dim, 6).data(), sequential_ids(700, 50000));
        expect_matches_search(idx, random_vectors(nq, dim, 7), nq, dim, k);
    });

    run("single query batch equals search()", [] {
        const std::size_t n = 1000, dim = 16;
        FlatIndex idx(dim);
        idx.add(random_vectors(n, dim, 8).data(), sequential_ids(n));
        expect_matches_search(idx, random_vectors(1, dim, 9), 1, dim, 10);
    });

    run("edge cases: nq=0, k=0, k>n, empty index", [] {
        const std::size_t dim = 8;
        auto queries = random_vectors(3, dim, 10);

        FlatIndex empty(dim);
        auto r_empty = empty.search_batch(queries.data(), 3, 5);
        CHECK(r_empty.size() == 3);
        for (auto& r : r_empty) CHECK(r.empty());

        FlatIndex idx(dim);
        idx.add(random_vectors(4, dim, 11).data(), sequential_ids(4));
        CHECK(idx.search_batch(queries.data(), 0, 5).empty());

        auto r_k0 = idx.search_batch(queries.data(), 3, 0);
        CHECK(r_k0.size() == 3);
        for (auto& r : r_k0) CHECK(r.empty());

        auto r_big = idx.search_batch(queries.data(), 3, 100);
        CHECK(r_big.size() == 3);
        for (auto& r : r_big) CHECK(r.size() == 4);
    });

    // Not a pass/fail test: shows the precision cost of the norm expansion.
    // Every vector is shifted far from the origin, so ||q||^2 and ||x||^2 are huge
    // and nearly cancel. Think about why recall drops here, and what you could do.
    run("[info] precision when data is far from the origin", [] {
        const std::size_t n = 2000, dim = 64, nq = 50, k = 10;
        auto data = random_vectors(n, dim, 12);
        auto queries = random_vectors(nq, dim, 13);
        for (auto& x : data) x += 1000.0f;
        for (auto& x : queries) x += 1000.0f;

        FlatIndex idx(dim);
        idx.add(data.data(), sequential_ids(n));
        auto batch = idx.search_batch(queries.data(), nq, k);

        std::size_t hits = 0;
        for (std::size_t q = 0; q < nq && q < batch.size(); ++q) {
            std::set<VectorId> ref_ids;
            for (auto& r : idx.search(&queries[q * dim], k)) ref_ids.insert(r.id);
            for (auto& r : batch[q]) hits += ref_ids.count(r.id);
        }
        std::printf("    recall@%zu vs search(): %.3f\n", k,
                    static_cast<double>(hits) / static_cast<double>(nq * k));
    });

    return report();
}
