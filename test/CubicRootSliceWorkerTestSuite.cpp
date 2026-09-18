#include <gtest/gtest.h>
#include "../sliceTiling.hpp"

#include <utility>
#include <vector>

namespace {

void expectNoOverlaps(const std::vector<TileRect>& tiles) {
    for (size_t i = 0; i < tiles.size(); ++i) {
        for (size_t j = i + 1; j < tiles.size(); ++j) {
            EXPECT_FALSE(tilesOverlap(tiles[i], tiles[j]))
                << "tiles " << i << " and " << j << " overlap: "
                << "q[" << tiles[i].qLo << "," << tiles[i].qHi << "] r[" << tiles[i].rLo << "," << tiles[i].rHi
                << "] vs q[" << tiles[j].qLo << "," << tiles[j].qHi << "] r[" << tiles[j].rLo << "," << tiles[j].rHi << "]";
        }
    }
}

void expectPartition(const LutBounds& bounds, const std::vector<TileRect>& tiles) {
    ASSERT_FALSE(tiles.empty());
    long long area = 0;
    for (size_t i = 0; i < tiles.size(); ++i) {
        EXPECT_TRUE(tileWithinBounds(tiles[i], bounds))
            << "tile " << i << " q[" << tiles[i].qLo << "," << tiles[i].qHi << "] r["
            << tiles[i].rLo << "," << tiles[i].rHi << "] outside bounds";
        area += tileCells(tiles[i]);
    }
    expectNoOverlaps(tiles);
    EXPECT_EQ(area, boundsCells(bounds))
        << "union area " << area << " != bounds area " << boundsCells(bounds);
}

void expectEveryCellCoveredExactlyOnce(const LutBounds& bounds, const std::vector<TileRect>& tiles) {
    expectPartition(bounds, tiles);
    const int qSpan = bounds.quintMax - bounds.quintMin + 1;
    const int rSpan = bounds.quartMax - bounds.quartMin + 1;
    std::vector<int> hits(static_cast<size_t>(qSpan) * static_cast<size_t>(rSpan), 0);
    for (const TileRect& t : tiles) {
        for (int q = t.qLo; q <= t.qHi; ++q) {
            for (int r = t.rLo; r <= t.rHi; ++r) {
                const size_t idx = static_cast<size_t>(q - bounds.quintMin) * static_cast<size_t>(rSpan)
                    + static_cast<size_t>(r - bounds.quartMin);
                hits[idx] += 1;
            }
        }
    }
    for (int q = bounds.quintMin; q <= bounds.quintMax; ++q) {
        for (int r = bounds.quartMin; r <= bounds.quartMax; ++r) {
            const size_t idx = static_cast<size_t>(q - bounds.quintMin) * static_cast<size_t>(rSpan)
                + static_cast<size_t>(r - bounds.quartMin);
            EXPECT_EQ(1, hits[idx]) << "cell (" << q << "," << r << ") covered " << hits[idx] << " time(s)";
        }
    }
}

bool tilesEqual(const std::vector<TileRect>& a, const std::vector<TileRect>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (!sameTile(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

} // namespace

class CubicRootSliceWorkerTestSuite : public ::testing::Test {};

TEST_F(CubicRootSliceWorkerTestSuite, SignedLutBoundsCoverFullQuintAndQuart) {
    const LutBounds b = lutBoundsForChecker(false);
    EXPECT_EQ(SLICE_QUINT_MIN, b.quintMin);
    EXPECT_EQ(SLICE_QUINT_MAX, b.quintMax);
    EXPECT_EQ(SLICE_QUART_MIN, b.quartMin);
    EXPECT_EQ(SLICE_QUART_MAX, b.quartMax);
}

TEST_F(CubicRootSliceWorkerTestSuite, PositiveOnlyLutBoundsStartQuintAtZero) {
    const LutBounds b = lutBoundsForChecker(true);
    EXPECT_EQ(0, b.quintMin);
    EXPECT_EQ(SLICE_QUINT_MAX, b.quintMax);
    EXPECT_EQ(SLICE_QUART_MIN, b.quartMin);
    EXPECT_EQ(SLICE_QUART_MAX, b.quartMax);
    EXPECT_LT(boundsCells(b), boundsCells(lutBoundsForChecker(false)));
}

TEST_F(CubicRootSliceWorkerTestSuite, ProbeSitsAtLutCornerAndIsClamped) {
    const LutBounds b{-5, 20, -10, 80};
    const TileRect probe = makeProbeTile(b);
    EXPECT_EQ(b.quintMin, probe.qLo);
    EXPECT_EQ(b.quartMin, probe.rLo);
    EXPECT_EQ(b.quintMin, probe.qHi);
    EXPECT_EQ(b.quartMin + DEFAULT_SLICE_QUART_CHUNK - 1, probe.rHi);

    const LutBounds tiny{0, 0, 3, 8};
    const TileRect clamped = makeProbeTile(tiny);
    EXPECT_EQ(0, clamped.qLo);
    EXPECT_EQ(0, clamped.qHi);
    EXPECT_EQ(3, clamped.rLo);
    EXPECT_EQ(8, clamped.rHi);
}

TEST_F(CubicRootSliceWorkerTestSuite, OneByThirtyOnSmallGridCoversEveryCell) {
    const LutBounds b{-4, 6, -40, 41};
    const SlotPlan plan = makeSlotPlan(b, DEFAULT_SLICE_QUINT_CHUNK, DEFAULT_SLICE_QUART_CHUNK);
    const std::vector<TileRect> tiles = enumerateAllTiles(plan, SLICE_ENQUEUE_BATCH);
    expectEveryCellCoveredExactlyOnce(b, tiles);
    EXPECT_TRUE(sameTile(tiles.front(), plan.probe));
}

TEST_F(CubicRootSliceWorkerTestSuite, UnevenChunksStillPartitionThePlane) {
    const LutBounds b{0, 17, -25, 50};
    const SlotPlan plan = makeSlotPlan(b, 3, 7);
    const std::vector<TileRect> tiles = enumerateAllTiles(plan, 1);
    expectEveryCellCoveredExactlyOnce(b, tiles);
}

TEST_F(CubicRootSliceWorkerTestSuite, ChunksLargerThanRemainingStillCover) {
    const LutBounds b{-2, 8, -5, 20};
    const SlotPlan plan = makeSlotPlan(b, 100, 1000);
    const std::vector<TileRect> tiles = enumerateAllTiles(plan, 8);
    expectEveryCellCoveredExactlyOnce(b, tiles);
    EXPECT_LE(tiles.size(), 3u);
}

TEST_F(CubicRootSliceWorkerTestSuite, DrainBatchesMatchSingleStepEnumeration) {
    const LutBounds b{-3, 12, -60, 59};
    const SlotPlan plan = makeSlotPlan(b, 1, 30);
    const std::vector<TileRect> byOne = enumerateAllTiles(plan, 1);
    const std::vector<TileRect> byEight = enumerateAllTiles(plan, 8);
    const std::vector<TileRect> byBatchConst = enumerateAllTiles(plan, SLICE_ENQUEUE_BATCH);
    EXPECT_TRUE(tilesEqual(byOne, byEight));
    EXPECT_TRUE(tilesEqual(byOne, byBatchConst));
    expectEveryCellCoveredExactlyOnce(b, byEight);
}

TEST_F(CubicRootSliceWorkerTestSuite, NothingRemainsAfterLastTile) {
    const LutBounds b{0, 9, 0, 99};
    const SlotPlan plan = makeSlotPlan(b, 2, 30);
    const std::vector<TileRect> tiles = enumerateAllTiles(plan, 8);
    ASSERT_GE(tiles.size(), 2u);
    std::vector<TileRect> extra;
    EXPECT_FALSE(nextRemainingTiles(plan, tiles.back(), false, 8, extra));
}

TEST_F(CubicRootSliceWorkerTestSuite, ProbeOnlyWhenBoundsEqualProbe) {
    const LutBounds b{5, 5, 10, 10 + DEFAULT_SLICE_QUART_CHUNK - 1};
    const SlotPlan plan = makeSlotPlan(b, 1, 30);
    const std::vector<TileRect> tiles = enumerateAllTiles(plan, 8);
    ASSERT_EQ(1u, tiles.size());
    EXPECT_TRUE(sameTile(tiles[0], plan.probe));
    expectEveryCellCoveredExactlyOnce(b, tiles);
}

TEST_F(CubicRootSliceWorkerTestSuite, FirstRemainingTileAbutsTheProbe) {
    const LutBounds b{-608383, -608370, -152231, -152100};
    const SlotPlan plan = makeSlotPlan(b, 1, 30);
    std::vector<TileRect> first;
    ASSERT_TRUE(nextRemainingTiles(plan, plan.probe, true, 1, first));
    ASSERT_EQ(1u, first.size());
    EXPECT_EQ(plan.probe.qLo, first[0].qLo);
    EXPECT_EQ(plan.probe.rHi + 1, first[0].rLo);
    EXPECT_FALSE(tilesOverlap(plan.probe, first[0]));
}

TEST_F(CubicRootSliceWorkerTestSuite, ProductionSignedBoundsWithCoarseChunksPartitionLut) {
    const LutBounds b = lutBoundsForChecker(false);
    const SlotPlan plan = makeSlotPlan(b, SLICE_MAX_QUINT_RANGE, SLICE_MAX_QUART_RANGE);
    const std::vector<TileRect> tiles = enumerateAllTiles(plan, SLICE_ENQUEUE_BATCH);
    expectPartition(b, tiles);

    int cornerHits = 0;
    const std::pair<int, int> corners[] = {
        {b.quintMin, b.quartMin},
        {b.quintMin, b.quartMax},
        {b.quintMax, b.quartMin},
        {b.quintMax, b.quartMax},
        {0, 0},
        {-1, -1},
    };
    for (const auto& cell : corners) {
        int covers = 0;
        for (const TileRect& t : tiles) {
            if (tileContainsCell(t, cell.first, cell.second)) {
                ++covers;
            }
        }
        EXPECT_EQ(1, covers) << "corner (" << cell.first << "," << cell.second << ")";
        cornerHits += covers;
    }
    EXPECT_EQ(6, cornerHits);
}

TEST_F(CubicRootSliceWorkerTestSuite, ProductionPositiveOnlyBoundsPartitionLut) {
    const LutBounds b = lutBoundsForChecker(true);
    const SlotPlan plan = makeSlotPlan(b, SLICE_MAX_QUINT_RANGE, SLICE_MAX_QUART_RANGE);
    const std::vector<TileRect> tiles = enumerateAllTiles(plan, 8);
    expectPartition(b, tiles);

    int coversNegativeQuint = 0;
    for (const TileRect& t : tiles) {
        if (tileContainsCell(t, -1, 0)) {
            ++coversNegativeQuint;
        }
    }
    EXPECT_EQ(0, coversNegativeQuint);
    int coversZero = 0;
    for (const TileRect& t : tiles) {
        if (tileContainsCell(t, 0, SLICE_QUART_MIN) || tileContainsCell(t, 0, SLICE_QUART_MAX)) {
            ++coversZero;
        }
    }
    EXPECT_GT(coversZero, 0);
}

TEST_F(CubicRootSliceWorkerTestSuite, OneByThirtyProductionFirstAndLastTilesMeetTheLutEdge) {
    const LutBounds b = lutBoundsForChecker(false);
    const SlotPlan plan = makeSlotPlan(b, DEFAULT_SLICE_QUINT_CHUNK, DEFAULT_SLICE_QUART_CHUNK);
    EXPECT_EQ(b.quintMin, plan.probe.qLo);
    EXPECT_EQ(b.quintMin, plan.probe.qHi);
    EXPECT_EQ(b.quartMin, plan.probe.rLo);
    EXPECT_EQ(b.quartMin + DEFAULT_SLICE_QUART_CHUNK - 1, plan.probe.rHi);

    std::vector<TileRect> first;
    ASSERT_TRUE(nextRemainingTiles(plan, plan.probe, true, 1, first));
    EXPECT_EQ(b.quintMin, first[0].qLo);
    EXPECT_EQ(plan.probe.rHi + 1, first[0].rLo);

    const int quartSpan = b.quartMax - b.quartMin + 1;
    const int nQuartTiles = (quartSpan + DEFAULT_SLICE_QUART_CHUNK - 1) / DEFAULT_SLICE_QUART_CHUNK;
    TileRect expectedLast;
    expectedLast.qLo = b.quintMax;
    expectedLast.qHi = b.quintMax;
    expectedLast.rLo = b.quartMin + (nQuartTiles - 1) * DEFAULT_SLICE_QUART_CHUNK;
    expectedLast.rHi = b.quartMax;
    EXPECT_TRUE(tileWithinBounds(expectedLast, b));
    EXPECT_TRUE(tileContainsCell(expectedLast, b.quintMax, b.quartMax));
    EXPECT_TRUE(tileContainsCell(expectedLast, b.quintMax, b.quartMin)
        || expectedLast.rLo > b.quartMin);

    std::vector<TileRect> afterLast;
    EXPECT_FALSE(nextRemainingTiles(plan, expectedLast, false, 8, afterLast));
}

TEST_F(CubicRootSliceWorkerTestSuite, LoopRangesOnlySliceQuintAndQuart) {
    const std::vector<int> ranges = makeQuinticFirstSliceLoopRanges(-3, 5, 10, 39);
    ASSERT_GE(ranges.size(), 12u);
    EXPECT_EQ(-3, ranges[0]);
    EXPECT_EQ(5, ranges[1]);
    EXPECT_EQ(10, ranges[2]);
    EXPECT_EQ(39, ranges[3]);
    for (int i = 4; i < 12; ++i) {
        EXPECT_EQ(USE_DEFAULT, ranges[i]) << "inner loop index " << i << " was sliced";
    }
}

TEST_F(CubicRootSliceWorkerTestSuite, ChunksFromProbeNeverGoBelowOneByThirty) {
    const LutBounds b = lutBoundsForChecker(true);
    const auto slow = chunksFromProbe(1000.0, b, 300.0);
    EXPECT_EQ(DEFAULT_SLICE_QUINT_CHUNK, slow.first);
    EXPECT_EQ(DEFAULT_SLICE_QUART_CHUNK, slow.second);

    const auto fast = chunksFromProbe(1e-6, b, 300.0);
    EXPECT_GE(fast.first, DEFAULT_SLICE_QUINT_CHUNK);
    EXPECT_GE(fast.second, DEFAULT_SLICE_QUART_CHUNK);
    EXPECT_LE(fast.first, SLICE_MAX_QUINT_RANGE);
    EXPECT_LE(fast.second, SLICE_MAX_QUART_RANGE);
    EXPECT_LE(fast.first, b.quintMax - b.quintMin + 1);
    EXPECT_LE(fast.second, b.quartMax - b.quartMin + 1);
}

TEST_F(CubicRootSliceWorkerTestSuite, DrainLoopCoversEveryLutIndexLikeTheWorker) {
    const LutBounds b{-8, 9, -70, 71};
    const SlotPlan plan = makeSlotPlan(b, 1, 30);

    std::vector<TileRect> queued;
    queued.push_back(plan.probe);
    TileRect last = plan.probe;
    bool lastIsProbe = true;
    for (;;) {
        std::vector<TileRect> batch;
        if (!nextRemainingTiles(plan, last, lastIsProbe, SLICE_ENQUEUE_BATCH, batch)) {
            break;
        }
        queued.insert(queued.end(), batch.begin(), batch.end());
        last = queued.back();
        lastIsProbe = false;
    }

    expectEveryCellCoveredExactlyOnce(b, queued);
    EXPECT_TRUE(tileContainsCell(queued.front(), b.quintMin, b.quartMin));
    bool sawMax = false;
    for (const TileRect& t : queued) {
        if (tileContainsCell(t, b.quintMax, b.quartMax)) {
            sawMax = true;
        }
    }
    EXPECT_TRUE(sawMax);
}
