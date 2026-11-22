#include "vdb/flat_index.h"
#include <algorithm>
namespace vdb {
void FlatIndex::add(const float* data, const std::vector<VectorId>& ids) {
    data_.insert(data_.end(), data, data + (dim_ * ids.size()));
    ids_.insert(ids_.end(), ids.begin(), ids.end());
}


std::vector<SearchResult> FlatIndex::search (const float*query, std::size_t k) const {
    std::vector<SearchResult> results;
    results.reserve(size());
    for (std::size_t i = 0; i < size(); i++) {
        const float* row = row_ptr(i);
        float dist = distance(metric_, query, row, dim_);
        SearchResult result = {ids_[i], dist};
        results.push_back(result);
    }
    k = std::min(k, size());
    std::partial_sort(results.begin(), results.begin() + k, results.end());
    results.resize(k);
    return results;
}
}