#include <gtest/gtest.h>
#include "../sliceTiling.hpp"

#include <algorithm>
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
        EXPECT_EQ(tiles[i].qLo, tiles[i].qHi) << "tile " << i << " is not a single quint";
        EXPECT_LE(tiles[i].rHi - tiles[i].rLo + 1, SLICE_QUART_CHUNK) << "tile " << i << " is wider than 100 quarts";
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

// One quint's quarts are tiled as adjacent 100-wide strips, last one clamped.
void expectQuartAxisPartition(const LutBounds& bounds) {
    long long covered = 0;
    int prevHi = bounds.quartMin - 1;
    for (int r = bounds.quartMin; r <= bounds.quartMax; r += SLICE_QUART_CHUNK) {
        const int rHi = std::min(r + SLICE_QUART_CHUNK - 1, bounds.quartMax);
        EXPECT_EQ(prevHi + 1, r);
        covered += rHi - r + 1;
        prevHi = rHi;
    }
    EXPECT_EQ(bounds.quartMax, prevHi);
    EXPECT_EQ(bounds.quartMax - bounds.quartMin + 1, covered);
}

} // namespace

TEST(CubicRootSliceWorkerTestSuite, SignedLutBoundsCoverFullQuintAndQuart) {
    const LutBounds b = lutBoundsForChecker(false);
    EXPECT_EQ(SLICE_QUINT_MIN, b.quintMin);
    EXPECT_EQ(SLICE_QUINT_MAX, b.quintMax);
    EXPECT_EQ(SLICE_QUART_MIN, b.quartMin);
    EXPECT_EQ(SLICE_QUART_MAX, b.quartMax);
}

TEST(CubicRootSliceWorkerTestSuite, PositiveOnlyLutBoundsStartQuintAtZero) {
    const LutBounds b = lutBoundsForChecker(true);
    EXPECT_EQ(0, b.quintMin);
    EXPECT_EQ(SLICE_QUINT_MAX, b.quintMax);
    EXPECT_EQ(SLICE_QUART_MIN, b.quartMin);
    EXPECT_EQ(SLICE_QUART_MAX, b.quartMax);
    EXPECT_LT(boundsCells(b), boundsCells(lutBoundsForChecker(false)));
}

TEST(CubicRootSliceWorkerTestSuite, FirstTileIsOneQuintByUpToOneHundredQuarts) {
    const LutBounds b{-5, 20, -10, 250};
    const TileRect t = firstTile(b);
    EXPECT_EQ(b.quintMin, t.qLo);
    EXPECT_EQ(b.quintMin, t.qHi);
    EXPECT_EQ(b.quartMin, t.rLo);
    EXPECT_EQ(b.quartMin + SLICE_QUART_CHUNK - 1, t.rHi);

    const LutBounds tiny{0, 0, 3, 8};
    const TileRect clamped = firstTile(tiny);
    EXPECT_EQ(0, clamped.qLo);
    EXPECT_EQ(0, clamped.qHi);
    EXPECT_EQ(3, clamped.rLo);
    EXPECT_EQ(8, clamped.rHi);
}

TEST(CubicRootSliceWorkerTestSuite, OneByOneHundredOnSmallGridCoversEveryCell) {
    const LutBounds b{-2, 3, -40, 161};
    const std::vector<TileRect> tiles = enumerateAllTiles(b, SLICE_ENQUEUE_BATCH);
    expectEveryCellCoveredExactlyOnce(b, tiles);
    EXPECT_TRUE(sameTile(tiles.front(), firstTile(b)));
    EXPECT_TRUE(sameTile(tiles.back(), lastTile(b)));
}

TEST(CubicRootSliceWorkerTestSuite, RaggedLastQuartStripStillCovers) {
    const LutBounds b{0, 4, 0, 249}; // 250 quarts: two full 100s and a leftover 50
    const std::vector<TileRect> tiles = enumerateAllTiles(b, 1);
    expectEveryCellCoveredExactlyOnce(b, tiles);
    EXPECT_EQ(5 * 3u, tiles.size());
    EXPECT_EQ(200, lastTile(b).rLo);
    EXPECT_EQ(249, lastTile(b).rHi);
}

TEST(CubicRootSliceWorkerTestSuite, DrainBatchesMatchSingleStepEnumeration) {
    const LutBounds b{-3, 4, -60, 159};
    const std::vector<TileRect> byOne = enumerateAllTiles(b, 1);
    const std::vector<TileRect> byEight = enumerateAllTiles(b, 8);
    const std::vector<TileRect> byBatchConst = enumerateAllTiles(b, SLICE_ENQUEUE_BATCH);
    EXPECT_TRUE(tilesEqual(byOne, byEight));
    EXPECT_TRUE(tilesEqual(byOne, byBatchConst));
    expectEveryCellCoveredExactlyOnce(b, byEight);
}

TEST(CubicRootSliceWorkerTestSuite, NothingRemainsAfterLastTile) {
    const LutBounds b{0, 9, 0, 249};
    const std::vector<TileRect> tiles = enumerateAllTiles(b, 8);
    ASSERT_GE(tiles.size(), 2u);
    std::vector<TileRect> extra;
    EXPECT_FALSE(nextTiles(b, tiles.back(), true, 8, extra));
    EXPECT_TRUE(sameTile(tiles.back(), lastTile(b)));
}

TEST(CubicRootSliceWorkerTestSuite, EmptySlotStartsAtFirstTile) {
    const LutBounds b{-4, 4, -20, 200};
    std::vector<TileRect> firstBatch;
    TileRect unused{};
    ASSERT_TRUE(nextTiles(b, unused, false, 1, firstBatch));
    ASSERT_EQ(1u, firstBatch.size());
    EXPECT_TRUE(sameTile(firstBatch[0], firstTile(b)));
}

TEST(CubicRootSliceWorkerTestSuite, NextTileAbutsPrevious) {
    const LutBounds b{-608383, -608370, -152231, -152000};
    const TileRect first = firstTile(b);
    TileRect second{};
    ASSERT_TRUE(nextTileAfter(b, first, second));
    EXPECT_EQ(first.qLo, second.qLo);
    EXPECT_EQ(first.rHi + 1, second.rLo);
    EXPECT_FALSE(tilesOverlap(first, second));
}

TEST(CubicRootSliceWorkerTestSuite, CrossingAQuintStartsANewQuartStrip) {
    const LutBounds b{0, 2, 0, 99}; // exactly one 1x100 tile per quint
    const TileRect first = firstTile(b);
    EXPECT_EQ(0, first.qLo);
    EXPECT_EQ(0, first.rLo);
    EXPECT_EQ(99, first.rHi);

    TileRect second{};
    ASSERT_TRUE(nextTileAfter(b, first, second));
    EXPECT_EQ(1, second.qLo);
    EXPECT_EQ(1, second.qHi);
    EXPECT_EQ(0, second.rLo);
    EXPECT_EQ(99, second.rHi);
}

TEST(CubicRootSliceWorkerTestSuite, ProductionQuartAxisIsCoveredWithoutHoles) {
    expectQuartAxisPartition(lutBoundsForChecker(false));
    expectQuartAxisPartition(lutBoundsForChecker(true));
}

TEST(CubicRootSliceWorkerTestSuite, ProductionFirstAndLastTilesMeetTheLutEdge) {
    const LutBounds b = lutBoundsForChecker(false);
    const TileRect first = firstTile(b);
    EXPECT_EQ(b.quintMin, first.qLo);
    EXPECT_EQ(b.quintMin, first.qHi);
    EXPECT_EQ(b.quartMin, first.rLo);
    EXPECT_EQ(b.quartMin + SLICE_QUART_CHUNK - 1, first.rHi);

    TileRect second{};
    ASSERT_TRUE(nextTileAfter(b, first, second));
    EXPECT_EQ(b.quintMin, second.qLo);
    EXPECT_EQ(first.rHi + 1, second.rLo);

    const TileRect last = lastTile(b);
    EXPECT_TRUE(tileWithinBounds(last, b));
    EXPECT_TRUE(tileContainsCell(last, b.quintMax, b.quartMax));
    EXPECT_EQ(b.quintMax, last.qLo);
    EXPECT_EQ(b.quintMax, last.qHi);

    std::vector<TileRect> afterLast;
    EXPECT_FALSE(nextTiles(b, last, true, 8, afterLast));
}

TEST(CubicRootSliceWorkerTestSuite, ProductionPositiveOnlyDoesNotCoverNegativeQuint) {
    const LutBounds b = lutBoundsForChecker(true);
    const TileRect first = firstTile(b);
    EXPECT_EQ(0, first.qLo);
    EXPECT_FALSE(tileContainsCell(first, -1, 0));

    const TileRect last = lastTile(b);
    EXPECT_EQ(SLICE_QUINT_MAX, last.qLo);
    EXPECT_TRUE(tileContainsCell(last, SLICE_QUINT_MAX, SLICE_QUART_MAX));
}

TEST(CubicRootSliceWorkerTestSuite, LoopRangesOnlySliceQuintAndQuart) {
    const std::vector<int> ranges = makeQuinticFirstSliceLoopRanges(-3, -3, 10, 109);
    ASSERT_GE(ranges.size(), 12u);
    EXPECT_EQ(-3, ranges[0]);
    EXPECT_EQ(-3, ranges[1]);
    EXPECT_EQ(10, ranges[2]);
    EXPECT_EQ(109, ranges[3]);
    for (int i = 4; i < 12; ++i) {
        EXPECT_EQ(USE_DEFAULT, ranges[i]) << "inner loop index " << i << " was sliced";
    }
}

TEST(CubicRootSliceWorkerTestSuite, DrainLoopCoversEveryLutIndexLikeTheWorker) {
    const LutBounds b{-8, 9, -70, 171};

    std::vector<TileRect> queued;
    TileRect last{};
    bool haveLast = false;
    for (;;) {
        std::vector<TileRect> batch;
        if (!nextTiles(b, last, haveLast, SLICE_ENQUEUE_BATCH, batch)) {
            break;
        }
        queued.insert(queued.end(), batch.begin(), batch.end());
        last = queued.back();
        haveLast = true;
    }

    expectEveryCellCoveredExactlyOnce(b, queued);
    EXPECT_TRUE(tileContainsCell(queued.front(), b.quintMin, b.quartMin));
    EXPECT_TRUE(tileContainsCell(queued.back(), b.quintMax, b.quartMax));
}
