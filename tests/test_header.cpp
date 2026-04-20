#include <gtest/gtest.h>
#include <slab/header.hpp>
#include <cstring>

using namespace slm::slab;

TEST(MemoryFSHeader, SizeIs64Bytes) {
    EXPECT_EQ(sizeof(MemoryFSHeader), 64u);
}

TEST(MemoryFSHeader, AlignmentIs64Bytes) {
    EXPECT_EQ(alignof(MemoryFSHeader), 64u);
}

TEST(MemoryFSHeader, MagicConstant) {
    EXPECT_EQ(MEMFS_MAGIC, 0x4D454D46u);
}

TEST(MemoryFSHeader, CommandConstants) {
    EXPECT_EQ(CMD_READ, 0x01u);
    EXPECT_EQ(CMD_WRITE_COMMIT, 0x02u);
}

TEST(MemoryFSHeader, DoneResponseMagic) {
    EXPECT_EQ(MEMFS_DONE, 0x444F4E45u);
}

TEST(MemoryFSHeader, FieldOffsets) {
    MemoryFSHeader h{};
    auto base = reinterpret_cast<const char*>(&h);

    EXPECT_EQ(reinterpret_cast<const char*>(&h.magic) - base, 0);
    EXPECT_EQ(reinterpret_cast<const char*>(&h.command) - base, 4);
    EXPECT_EQ(reinterpret_cast<const char*>(&h.total_size) - base, 8);
    EXPECT_EQ(reinterpret_cast<const char*>(&h.text_offset) - base, 16);
    EXPECT_EQ(reinterpret_cast<const char*>(&h.text_length) - base, 20);
    EXPECT_EQ(reinterpret_cast<const char*>(&h.vector_offset) - base, 24);
    EXPECT_EQ(reinterpret_cast<const char*>(&h.vector_dim) - base, 28);
    EXPECT_EQ(reinterpret_cast<const char*>(&h.parent_id) - base, 32);
    EXPECT_EQ(reinterpret_cast<const char*>(&h.depth) - base, 36);
}

TEST(MemoryFSHeader, DefaultInitialization) {
    MemoryFSHeader h{};
    EXPECT_EQ(h.magic, 0u);
    EXPECT_EQ(h.command, 0u);
    EXPECT_EQ(h.total_size, 0u);
    EXPECT_EQ(h.text_offset, 0u);
    EXPECT_EQ(h.text_length, 0u);
    EXPECT_EQ(h.vector_offset, 0u);
    EXPECT_EQ(h.vector_dim, 0u);
    EXPECT_EQ(h.parent_id, 0u);
    EXPECT_EQ(h.depth, 0u);
}

TEST(MemoryFSHeader, AlignUp) {
    EXPECT_EQ(align_up(0, 64), 0u);
    EXPECT_EQ(align_up(1, 64), 64u);
    EXPECT_EQ(align_up(63, 64), 64u);
    EXPECT_EQ(align_up(64, 64), 64u);
    EXPECT_EQ(align_up(65, 64), 128u);
    EXPECT_EQ(align_up(100, 64), 128u);
}

TEST(MemoryFSHeader, VectorOffsetAlignment) {
    uint32_t text_end = 64 + 50;
    uint32_t vec_off = align_up(text_end, 64);
    EXPECT_EQ(vec_off, 128u);
    EXPECT_EQ(vec_off % 64, 0u);
}
