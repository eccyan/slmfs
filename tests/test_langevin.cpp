#include <gtest/gtest.h>
#include <langevin/poincare_disk.hpp>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace slm::langevin;

TEST(DiskPosition, RadiusAtOrigin) {
    DiskPosition p{0.0f, 0.0f};
    EXPECT_FLOAT_EQ(p.radius(), 0.0f);
}

TEST(DiskPosition, RadiusOnAxis) {
    DiskPosition p{0.6f, 0.0f};
    EXPECT_FLOAT_EQ(p.radius(), 0.6f);
}

TEST(DiskPosition, RadiusDiagonal) {
    DiskPosition p{0.3f, 0.4f};
    EXPECT_FLOAT_EQ(p.radius(), 0.5f);
}

TEST(DiskPosition, InverseMetricAtOrigin) {
    DiskPosition p{0.0f, 0.0f};
    EXPECT_FLOAT_EQ(inverse_metric(p), 0.25f);
}

TEST(DiskPosition, InverseMetricAtMidRadius) {
    DiskPosition p{0.5f, 0.0f};
    EXPECT_NEAR(inverse_metric(p), 0.140625f, 1e-6f);
}

TEST(DiskPosition, InverseMetricNearBoundary) {
    DiskPosition p{0.95f, 0.0f};
    float expected = (1.0f - 0.95f * 0.95f) * (1.0f - 0.95f * 0.95f) / 4.0f;
    EXPECT_NEAR(inverse_metric(p), expected, 1e-6f);
}

TEST(DiskPosition, ProjectInsideDiskUnchanged) {
    DiskPosition p{0.3f, 0.4f};
    auto projected = project_to_disk(p);
    EXPECT_FLOAT_EQ(projected.x, 0.3f);
    EXPECT_FLOAT_EQ(projected.y, 0.4f);
}

TEST(DiskPosition, ProjectOnBoundaryClamps) {
    DiskPosition p{1.0f, 0.0f};
    auto projected = project_to_disk(p);
    EXPECT_LT(projected.radius(), 1.0f);
    EXPECT_NEAR(projected.radius(), 0.999f, 1e-3f);
}

TEST(DiskPosition, ProjectOutsideDiskClamps) {
    DiskPosition p{2.0f, 0.0f};
    auto projected = project_to_disk(p);
    EXPECT_LT(projected.radius(), 1.0f);
    EXPECT_NEAR(projected.radius(), 0.999f, 1e-3f);
    EXPECT_GT(projected.x, 0.0f);
    EXPECT_FLOAT_EQ(projected.y, 0.0f);
}

TEST(DiskPosition, ProjectDiagonalOvershoot) {
    DiskPosition p{1.0f, 1.0f};
    auto projected = project_to_disk(p);
    EXPECT_LT(projected.radius(), 1.0f);
    EXPECT_NEAR(projected.x, projected.y, 1e-5f);
}

TEST(DiskPosition, ProjectZeroVectorUnchanged) {
    DiskPosition p{0.0f, 0.0f};
    auto projected = project_to_disk(p);
    EXPECT_FLOAT_EQ(projected.x, 0.0f);
    EXPECT_FLOAT_EQ(projected.y, 0.0f);
}

TEST(NodeState, DefaultConstruction) {
    NodeState state{};
    EXPECT_FLOAT_EQ(state.pos.x, 0.0f);
    EXPECT_FLOAT_EQ(state.pos.y, 0.0f);
    EXPECT_EQ(state.last_access_tick, 0u);
    EXPECT_EQ(state.access_count, 0u);
}

#include <langevin/sde_stepper.hpp>

// --- LangevinStepper activate tests ---

static const LangevinStepper DEFAULT_STEPPER({
    .dt = 1.0f, .lambda_decay = 5.0e-6f, .noise_scale = 0.0f,
    .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});

TEST(LangevinActivate, ResetsWithThermalKick) {
    NodeState node{};
    node.pos = {0.5f, 0.3f};
    node.last_access_tick = 100;
    node.access_count = 3;
    std::mt19937 rng(42);

    DEFAULT_STEPPER.activate(node, 200, rng);

    EXPECT_NEAR(node.pos.radius(), 0.01f, 1e-6f)
        << "Activated node should have thermal kick radius of ~0.01";
    EXPECT_EQ(node.last_access_tick, 200u);
    EXPECT_EQ(node.access_count, 4u);
}

TEST(LangevinActivate, IncrementsAccessCount) {
    NodeState node{};
    node.access_count = 0;
    std::mt19937 rng(42);

    DEFAULT_STEPPER.activate(node, 1, rng);
    EXPECT_EQ(node.access_count, 1u);

    DEFAULT_STEPPER.activate(node, 2, rng);
    EXPECT_EQ(node.access_count, 2u);
}

TEST(LangevinActivate, UpdatesTick) {
    NodeState node{};
    node.last_access_tick = 50;
    std::mt19937 rng(42);

    DEFAULT_STEPPER.activate(node, 999, rng);
    EXPECT_EQ(node.last_access_tick, 999u);
}

TEST(LangevinActivate, RandomAngleVaries) {
    NodeState node_a{}, node_b{};
    std::mt19937 rng_a(42), rng_b(99);

    DEFAULT_STEPPER.activate(node_a, 0, rng_a);
    DEFAULT_STEPPER.activate(node_b, 0, rng_b);

    EXPECT_NEAR(node_a.pos.radius(), node_b.pos.radius(), 1e-6f);
    bool angles_differ = (node_a.pos.x != node_b.pos.x)
                      || (node_a.pos.y != node_b.pos.y);
    EXPECT_TRUE(angles_differ)
        << "Different RNG seeds should produce different kick angles";
}

TEST(LangevinActivate, CustomKickRadius) {
    LangevinStepper big_kick({.dt = 1.0f, .lambda_decay = 0.0f, .noise_scale = 0.0f,
                               .archive_threshold = 0.95f, .thermal_kick_radius = 0.05f});
    NodeState node{};
    std::mt19937 rng(42);

    big_kick.activate(node, 0, rng);
    EXPECT_NEAR(node.pos.radius(), 0.05f, 1e-6f)
        << "Kick radius should match config value";
}

TEST(LangevinConfig, DefaultValues) {
    LangevinStepper::Config config{};
    config.dt = 1.0f;
    config.lambda_decay = 5.0e-6f;
    config.noise_scale = 2.0e-4f;
    config.archive_threshold = 0.95f;
    config.thermal_kick_radius = 0.01f;

    EXPECT_FLOAT_EQ(config.dt, 1.0f);
    EXPECT_FLOAT_EQ(config.lambda_decay, 5.0e-6f);
    EXPECT_FLOAT_EQ(config.noise_scale, 2.0e-4f);
    EXPECT_FLOAT_EQ(config.archive_threshold, 0.95f);
    EXPECT_FLOAT_EQ(config.thermal_kick_radius, 0.01f);
}

// --- LangevinStepper step tests ---
// Time is now measured in cognitive ticks, not wall-clock seconds.

TEST(LangevinStep, NoNodesReturnsEmpty) {
    LangevinStepper stepper({.dt = 1.0f, .lambda_decay = 5.0e-6f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);
    std::span<NodeState> empty;
    auto archived = stepper.step(empty, 100, 1, rng);
    EXPECT_TRUE(archived.empty());
}

TEST(LangevinStep, DeterministicDriftWithoutNoise) {
    LangevinStepper stepper({.dt = 1.0f, .lambda_decay = 5.0e-6f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.3f, 0.0f};
    node.last_access_tick = 0;
    std::vector<NodeState> nodes = {node};

    // Step at tick 200 with delta_ticks=1
    stepper.step(nodes, 200, 1, rng);

    EXPECT_GT(nodes[0].pos.radius(), 0.3f)
        << "Node should drift outward when unaccessed";
    EXPECT_LT(nodes[0].pos.radius(), 1.0f)
        << "Node should remain inside the disk";
}

TEST(LangevinStep, RecentlyAccessedDriftsLess) {
    LangevinStepper stepper({.dt = 1.0f, .lambda_decay = 5.0e-6f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});

    NodeState node_a{};
    node_a.pos = {0.3f, 0.0f};
    node_a.last_access_tick = 0;  // 2000 ticks old

    NodeState node_b{};
    node_b.pos = {0.3f, 0.0f};
    node_b.last_access_tick = 1800;  // 200 ticks old

    std::vector<NodeState> nodes_a = {node_a};
    std::vector<NodeState> nodes_b = {node_b};
    std::mt19937 rng_a(42), rng_b(42);

    stepper.step(nodes_a, 2000, 1, rng_a);
    stepper.step(nodes_b, 2000, 1, rng_b);

    EXPECT_GT(nodes_a[0].pos.radius(), nodes_b[0].pos.radius())
        << "Node accessed longer ago should drift more";
}

TEST(LangevinStep, NodeAtOriginDriftsNowhere) {
    LangevinStepper stepper({.dt = 1.0f, .lambda_decay = 5.0e-6f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.0f, 0.0f};
    node.last_access_tick = 0;
    std::vector<NodeState> nodes = {node};

    stepper.step(nodes, 200, 1, rng);
    EXPECT_FLOAT_EQ(nodes[0].pos.radius(), 0.0f);
}

TEST(LangevinStep, ArchivesNodesBeyondThreshold) {
    // Very large lambda to force archival in one step
    LangevinStepper stepper({.dt = 1.0f, .lambda_decay = 1.0f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.94f, 0.0f};
    node.last_access_tick = 0;
    std::vector<NodeState> nodes = {node};

    // 2000 ticks old, delta_ticks=1
    auto archived = stepper.step(nodes, 2000, 1, rng);
    EXPECT_EQ(archived.size(), 1u);
    EXPECT_EQ(archived[0], 0u);
}

TEST(LangevinStep, StaysInsideDisk) {
    LangevinStepper stepper({.dt = 1.0f, .lambda_decay = 1.0f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.9f, 0.0f};
    node.last_access_tick = 0;
    std::vector<NodeState> nodes = {node};

    stepper.step(nodes, 5000, 10, rng);
    EXPECT_LT(nodes[0].pos.radius(), 1.0f)
        << "project_to_disk must prevent escape from the disk";
}

TEST(LangevinStep, NoiseAddsRandomness) {
    LangevinStepper stepper({.dt = 1.0f, .lambda_decay = 0.0f,
                              .noise_scale = 0.1f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});

    NodeState node{};
    node.pos = {0.3f, 0.0f};
    node.last_access_tick = 0;

    std::vector<NodeState> nodes_a = {node};
    std::vector<NodeState> nodes_b = {node};
    std::mt19937 rng_a(42), rng_b(99);

    stepper.step(nodes_a, 200, 1, rng_a);
    stepper.step(nodes_b, 200, 1, rng_b);

    bool positions_differ = (nodes_a[0].pos.x != nodes_b[0].pos.x)
                         || (nodes_a[0].pos.y != nodes_b[0].pos.y);
    EXPECT_TRUE(positions_differ)
        << "Different RNG seeds should produce different positions";
}

TEST(LangevinStep, MultipleNodesIndependent) {
    LangevinStepper stepper({.dt = 1.0f, .lambda_decay = 5.0e-6f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    std::vector<NodeState> nodes(3);
    nodes[0].pos = {0.2f, 0.0f};
    nodes[0].last_access_tick = 0;
    nodes[1].pos = {0.5f, 0.0f};
    nodes[1].last_access_tick = 0;
    nodes[2].pos = {0.8f, 0.0f};
    nodes[2].last_access_tick = 0;

    // Step at tick 1000 with delta_ticks=1
    stepper.step(nodes, 1000, 1, rng);

    EXPECT_GT(nodes[0].pos.radius(), 0.2f);
    EXPECT_GT(nodes[1].pos.radius(), 0.5f);
    EXPECT_GT(nodes[2].pos.radius(), 0.8f);

    EXPECT_GT(nodes[2].pos.radius(), nodes[1].pos.radius());
    EXPECT_GT(nodes[1].pos.radius(), nodes[0].pos.radius());
}

TEST(LangevinStep, DeltaTicksScalesDrift) {
    // A burst of 10 ticks should produce more drift than a single tick.
    LangevinStepper stepper({.dt = 1.0f, .lambda_decay = 5.0e-6f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});

    NodeState node{};
    node.pos = {0.3f, 0.0f};
    node.last_access_tick = 0;

    std::vector<NodeState> nodes_1 = {node};
    std::vector<NodeState> nodes_10 = {node};
    std::mt19937 rng_1(42), rng_10(42);

    stepper.step(nodes_1, 500, 1, rng_1);
    stepper.step(nodes_10, 500, 10, rng_10);

    EXPECT_GT(nodes_10[0].pos.radius(), nodes_1[0].pos.radius())
        << "10-tick burst should produce more drift than 1 tick";
}

TEST(LangevinStep, ZeroDeltaTicksNoOp) {
    // delta_ticks=0 means dt_eff=0: no drift, no noise
    LangevinStepper stepper({.dt = 1.0f, .lambda_decay = 5.0e-6f,
                              .noise_scale = 0.1f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.3f, 0.0f};
    node.last_access_tick = 0;
    std::vector<NodeState> nodes = {node};

    stepper.step(nodes, 500, 0, rng);

    EXPECT_FLOAT_EQ(nodes[0].pos.x, 0.3f);
    EXPECT_FLOAT_EQ(nodes[0].pos.y, 0.0f);
}

// --- Integration: full lifecycle ---

TEST(LangevinIntegration, FullLifecycle) {
    // Simulate tick-by-tick to find archival window.
    // With lambda_decay=1e-3, dt=5.0, starting at r=0.01:
    // Target: archived within 1000-5000 ticks.
    LangevinStepper stepper({.dt = 1.0f, .lambda_decay = 5.0e-6f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.01f, 0.0f};
    node.last_access_tick = 0;
    node.access_count = 0;
    std::vector<NodeState> nodes = {node};

    uint64_t tick = 0;
    std::vector<uint32_t> archived;
    std::vector<float> radius_history;

    const uint64_t max_ticks = 10000;
    while (tick < max_ticks && archived.empty()) {
        tick += 1;
        archived = stepper.step(nodes, tick, 1, rng);
        radius_history.push_back(nodes[0].pos.radius());
    }

    // Radius should monotonically increase (no noise)
    for (size_t j = 1; j < radius_history.size(); ++j) {
        EXPECT_GE(radius_history[j], radius_history[j - 1] - 1e-6f)
            << "Radius should monotonically increase at tick " << j;
    }

    // Should be archived within reasonable tick count
    EXPECT_FALSE(archived.empty())
        << "Node should be archived after sufficient ticks without access";

    EXPECT_GE(tick, 500u)
        << "Node should survive at least 500 ticks";
    EXPECT_LE(tick, 5000u)
        << "Node should be archived within 5000 ticks";
}

TEST(LangevinIntegration, ActivationResetsLifecycle) {
    LangevinStepper stepper({.dt = 1.0f, .lambda_decay = 5.0e-6f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.01f, 0.0f};
    node.last_access_tick = 0;
    std::vector<NodeState> nodes = {node};

    // Drift for 500 ticks
    uint64_t tick = 0;
    while (tick < 500) {
        tick += 1;
        stepper.step(nodes, tick, 1, rng);
    }
    float radius_before_activation = nodes[0].pos.radius();
    EXPECT_GT(radius_before_activation, 0.01f)
        << "Should have drifted outward";

    // Activate (simulate agent reading the memory)
    stepper.activate(nodes[0], tick, rng);
    EXPECT_NEAR(nodes[0].pos.radius(), 0.01f, 1e-6f)
        << "Activation should reset to thermal kick radius";

    // Drift again for only 250 ticks (half the original duration)
    uint64_t drift_end = tick + 250;
    while (tick < drift_end) {
        tick += 1;
        stepper.step(nodes, tick, 1, rng);
    }
    float radius_after_reactivation = nodes[0].pos.radius();

    // Should have drifted less than before (250 vs 500 ticks)
    EXPECT_LT(radius_after_reactivation, radius_before_activation)
        << "Reactivated node should drift less (shorter time since access)";
}

TEST(LangevinIntegration, CohomologyDriftPenalty) {
    // Superseded node at r=0.93 should archive quickly with large lambda
    LangevinStepper stepper({.dt = 1.0f, .lambda_decay = 1.0f,
                              .noise_scale = 0.0f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.0f, 0.93f};
    node.last_access_tick = 0;
    std::vector<NodeState> nodes = {node};

    // 200 ticks old with large lambda should push it past
    auto archived = stepper.step(nodes, 200, 1, rng);
    EXPECT_EQ(archived.size(), 1u)
        << "Node placed at r=0.93 with old access should archive in one tick";
}

TEST(LangevinIntegration, NoiseTooWeakToArchiveAlone) {
    // Pure Brownian motion should not push a node to archive boundary
    LangevinStepper stepper({.dt = 1.0f, .lambda_decay = 0.0f,
                              .noise_scale = 2.0e-4f, .archive_threshold = 0.95f, .thermal_kick_radius = 0.01f});
    std::mt19937 rng(42);

    NodeState node{};
    node.pos = {0.5f, 0.0f};
    node.last_access_tick = 0;
    std::vector<NodeState> nodes = {node};

    // Simulate 3000 ticks with occasional bursts
    uint64_t tick = 0;
    while (tick < 3000) {
        tick += 1;
        auto archived = stepper.step(nodes, tick, 1, rng);
        EXPECT_TRUE(archived.empty())
            << "Pure noise should not archive a node at r=0.5 within 3000 ticks";
        if (!archived.empty()) break;
    }
}
