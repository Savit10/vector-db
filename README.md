# vector-db

A vector database written from scratch in C++17, built to learn the internals.
It uses no FAISS or hnswlib; those may only be used as reference points for recall and speed.

## Build & run

```bash
cmake -S . -B build && cmake --build build -j
./build/test_flat_index
./build/bench_flat 100000 128 100 10
```

## Layout

```
include/vdb/   headers (yours)
src/           implementations (yours; CMake builds src/flat_index.cpp)
tests/         dependency-free tests (auxiliary)
bench/         throughput benchmarks (auxiliary)
```

## API the tests and benchmark expect

The tests and benchmark include `vdb/flat_index.h` and use the names below.
If you design it differently, update `tests/` and `bench/` to match.

- `vdb::VectorId`: integer ID type. Avoid the name `id_t`, which clashes with POSIX `id_t` on macOS.
- `vdb::Metric::L2` and `vdb::Metric::InnerProduct`
- `vdb::l2_sq(a, b, dim)` and `vdb::inner_product(a, b, dim)`
- `vdb::distance(metric, a, b, dim)`: lower means closer, so inner product comes back negated.
- `vdb::SearchResult { id, distance }`
- `vdb::FlatIndex(dim, metric = L2)`, with:
  - `add(const float* data, const std::vector<VectorId>& ids)`: data is row-major
  - `search(const float* query, size_t k)`: returns `std::vector<SearchResult>`, closest first, at most k results
  - `size()`
  - `search_batch(const float* queries, size_t nq, size_t k) const`: returns `std::vector<std::vector<SearchResult>>`, where `result[q]` matches `search(&queries[q * dim], k)` (step 2, tested in `tests/test_search_batch.cpp`)

## Roadmap

1. **Flat index**: exact top-k with a bounded heap. *(in progress)*
2. **Faster distance**: SIMD (NEON on Apple Silicon), and batching queries so the work becomes a matrix multiply.
3. **Recall harness**: SIFT1M / random data; measure recall@k against Flat.
4. **IVF**: k-means coarse quantizer, inverted lists, `nprobe` tuning.
5. **HNSW**: layered navigable small-world graph (the core of most production vector DBs).
6. **Quantization**: scalar (int8) and product quantization (PQ), IVF-PQ.
7. **Persistence**: on-disk format, mmap, write-ahead log, deletes and tombstones.
8. **Metadata filtering** and a small query API or server.
