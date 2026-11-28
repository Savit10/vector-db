#include "vdb/flat_index.h"
#include <algorithm>
#include <queue>
#include <limits>
namespace vdb {
void FlatIndex::add(const float* data, const std::vector<VectorId>& ids) {
    data_.insert(data_.end(), data, data + (dim_ * ids.size()));
    ids_.insert(ids_.end(), ids.begin(), ids.end());
}

std::vector<SearchResult> FlatIndex::search (const float*query, std::size_t k) const {
    if (k == 0) {
        return {};
    }
    k = std::min(k, size());
    std::priority_queue<SearchResult> results;
    for (std::size_t i = 0; i < size(); i++) {
        const float* row = row_ptr(i);
        float dist = distance(metric_, query, row, dim_);
        SearchResult result = {ids_[i], dist};
        if (results.size() >= k) {
            const SearchResult& top = results.top();
            if (top.distance > dist) {
                results.pop();
                results.push(result);
            }
        }
        else {
            results.push(result);
        }
    }
    std::vector<SearchResult> out(k);
    for (std::size_t i = out.size(); i-- > 0; ) {   // i = size-1 down to 0
        out[i] = results.top();
        results.pop();
    }
    return out;
}

std::vector<std::vector<SearchResult>> FlatIndex::search_batch(const float* queries, std::size_t nq, std::size_t k) const{
    k = std::min(k, size());
    std::vector<std::vector<SearchResult>> results(nq);
    if (k == 0) { return results; }
    std::vector<std::priority_queue<SearchResult>> heaps(nq);
    std::size_t tile_size = 128; // L1 Cache = 128KB, half = 64, assuming float = 4 bytes and dim = 128
    for (std::size_t t0 = 0; t0 < size(); t0 += tile_size) { // t0 = tile start
        std::size_t t1 = std::min(t0 + tile_size, size());   // t1 = tile end; last tile may be partial
        for (std::size_t q = 0; q < nq; q++) {
            const float* query = queries + q * dim_;
            auto& curHeap = heaps[q];
            float threshold = (curHeap.size() == k)? curHeap.top().distance : std::numeric_limits<float>::infinity();
            for (std::size_t vector = t0; vector < t1; vector++) {
                float dist = distance(metric_, query, row_ptr(vector), dim_);
                if (dist < threshold) {
                    SearchResult result = {ids_[vector], dist};
                    if (curHeap.size() >= k) {
                        curHeap.pop();
                    }
                    curHeap.push(result);
                    threshold = (curHeap.size() == k)? curHeap.top().distance : std::numeric_limits<float>::infinity();
                }
                
            }
        }
    }
    for (std::size_t q = 0; q < nq; q++) {
        auto& curHeap = heaps[q];
        results[q].resize(curHeap.size());
        for (std::size_t i = curHeap.size(); i-- > 0; ) {   // i = size-1 down to 0
            results[q][i] = curHeap.top();
            curHeap.pop();
        }
    }
    return results;
}
}