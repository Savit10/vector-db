#include "vdb/flat_index.h"
#include <algorithm>
#include <queue>
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
}