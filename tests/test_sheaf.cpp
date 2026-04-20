#include <gtest/gtest.h>
#include <sheaf/neighborhood.hpp>
#include <vector>

using namespace slm::sheaf;

TEST(Edge, Construction) {
    std::vector<float> rel = {0.1f, 0.2f, 0.3f};
    Edge edge{42, EdgeType::Structural, rel};

    EXPECT_EQ(edge.neighbor_index, 42u);
    EXPECT_EQ(edge.type, EdgeType::Structural);
    EXPECT_EQ(edge.relation.size(), 3u);
    EXPECT_FLOAT_EQ(edge.relation[0], 0.1f);
}

TEST(Edge, KnnEdgeType) {
    Edge edge{10, EdgeType::KNN, {}};
    EXPECT_EQ(edge.type, EdgeType::KNN);
}

TEST(Neighborhood, Construction) {
    std::vector<float> new_mu = {1.0f, 2.0f};
    std::vector<float> neighbor_mu = {1.1f, 2.1f};
    std::vector<float> rel = {0.1f, 0.1f};

    Neighborhood hood;
    hood.new_node_mu = new_mu;
    hood.new_node_text = "new memory";
    hood.neighbor_mus.push_back(neighbor_mu);
    hood.neighbor_texts.push_back("old memory");
    hood.edges.push_back({0, EdgeType::Structural, rel});

    EXPECT_EQ(hood.new_node_mu.size(), 2u);
    EXPECT_EQ(hood.neighbor_mus.size(), 1u);
    EXPECT_EQ(hood.edges.size(), 1u);
    EXPECT_EQ(hood.new_node_text, "new memory");
    EXPECT_EQ(hood.neighbor_texts[0], "old memory");
}

TEST(Neighborhood, EmptyNeighborhood) {
    Neighborhood hood;
    hood.new_node_mu = std::vector<float>{1.0f};
    hood.new_node_text = "alone";
    EXPECT_TRUE(hood.edges.empty());
    EXPECT_TRUE(hood.neighbor_mus.empty());
}
