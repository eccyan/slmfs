#include <gtest/gtest.h>
#include <langevin/poincare_disk.hpp>
#include <cmath>

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
    EXPECT_DOUBLE_EQ(state.last_access_time, 0.0);
    EXPECT_EQ(state.access_count, 0u);
}
