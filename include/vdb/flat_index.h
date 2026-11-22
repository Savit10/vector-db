#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <array>


namespace vdb {

using VectorId = std::int64_t;
enum class Metric {
    L2, InnerProduct
};

struct SearchResult {
    VectorId id;
    float distance;

    bool operator<(const SearchResult& other) const {
        return distance < other.distance;
    }
};

// Number of independent partial sums. 8 floats = two 4-wide NEON registers,
// which gives the CPU 8 separate dependency chains to overlap.
constexpr std::size_t kLanes = 8;

inline float l2_sq(const float* a, const float* b, std::size_t dim) {
    std::array<float, kLanes> acc{};
    std::size_t i = 0;

    // Main loop: process 8 floats per iteration. Inside, `j` is a compile-time
    // pattern (0..7), so acc[j] maps to a fixed register lane, not stack memory.
    for (; i + kLanes <= dim; i += kLanes) {
        for (std::size_t j = 0; j < kLanes; ++j) {
            float diff = a[i + j] - b[i + j];
            acc[j] += diff * diff;
        }
    }

    // Combine the 8 partial sums once, at the end.
    float sum = 0.0f;
    for (std::size_t j = 0; j < kLanes; ++j) sum += acc[j];

    // Tail: the last dim % 8 elements, done one at a time.
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

inline float inner_product(const float* a, const float* b, std::size_t dim) {
    std::array<float, kLanes> acc{};
    std::size_t i = 0;

    for (; i + kLanes <= dim; i += kLanes) {
        for (std::size_t j = 0; j < kLanes; ++j) {
            acc[j] += a[i + j] * b[i + j];
        }
    }

    float sum = 0.0f;
    for (std::size_t j = 0; j < kLanes; ++j) sum += acc[j];

    for (; i < dim; ++i) sum += a[i] * b[i];
    return sum;
}

inline float distance(Metric m, const float*a, const float*b, std::size_t dim) {
    if (m == Metric::L2) { return l2_sq(a, b, dim);}
    return -inner_product(a, b, dim);
}


class FlatIndex {
    private:
        std::vector<float> data_;
        std::vector<VectorId> ids_;
        std::size_t dim_;
        Metric metric_;
        const float* row_ptr(std::size_t i) const { return data_.data() + i * dim_; }

    public:
        FlatIndex(std::size_t dim, Metric metric = Metric::L2): dim_(dim), metric_(metric){}

        void add(const float* data, const std::vector<VectorId>& ids);
        std::vector<SearchResult> search (const float*query, std::size_t k) const;
        std::size_t size() const { return ids_.size();};
        

};

}


