#include <metric/fisher_rao.hpp>
#include <metric/simd_ops.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

namespace slm::metric {

float FisherRaoMetric::distance(const GaussianNode& p, const GaussianNode& q) const {
    assert(p.mu.size() == q.mu.size());
    assert(p.sigma.size() == q.sigma.size());
    assert(p.mu.size() == p.sigma.size());

    uint32_t dim = static_cast<uint32_t>(p.mu.size());

    float var_div = simd_variance_divergence(
        p.sigma.data(), q.sigma.data(), dim
    );

    float weighted_diff = simd_weighted_sq_diff(
        p.mu.data(), q.mu.data(),
        p.sigma.data(), q.sigma.data(), dim
    );

    return std::sqrt(var_div + weighted_diff);
}

std::vector<uint32_t> FisherRaoMetric::top_k(
    const GaussianNode& query,
    std::span<const GaussianNode> candidates,
    uint32_t k
) const {
    // Placeholder — implemented in Task 4
    return {};
}

} // namespace slm::metric
