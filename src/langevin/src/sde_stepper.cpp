#include <langevin/sde_stepper.hpp>
#include <cmath>

namespace slm::langevin {

LangevinStepper::LangevinStepper(Config config)
    : config_(config) {}

void LangevinStepper::activate(NodeState& node, double current_time) {
    node.pos = {0.0f, 0.0f};
    node.last_access_time = current_time;
    node.access_count += 1;
}

std::vector<uint32_t> LangevinStepper::step(
    std::span<NodeState> nodes,
    double current_time,
    std::mt19937& rng
) const {
    // Placeholder — implemented in Task 3
    return {};
}

} // namespace slm::langevin
