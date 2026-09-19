#ifndef SLICE_TILING_HPP
#define SLICE_TILING_HPP

#include "math.hpp"

#include <algorithm>
#include <vector>

struct LutBounds {
    int quintMin;
    int quintMax;
    int quartMin;
    int quartMax;
};

struct TileRect {
    int qLo;
    int qHi;
    int rLo;
    int rHi;
};

inline LutBounds lutBoundsForChecker(bool positiveOnlyQuint) {
    return {
        positiveOnlyQuint ? 0 : SLICE_QUINT_MIN,
        SLICE_QUINT_MAX,
        SLICE_QUART_MIN,
        SLICE_QUART_MAX
    };
}

inline bool sameTile(const TileRect& a, const TileRect& b) {
    return a.qLo == b.qLo && a.qHi == b.qHi && a.rLo == b.rLo && a.rHi == b.rHi;
}

inline long long tileCells(const TileRect& t) {
    return static_cast<long long>(t.qHi - t.qLo + 1) * static_cast<long long>(t.rHi - t.rLo + 1);
}

inline long long boundsCells(const LutBounds& b) {
    return static_cast<long long>(b.quintMax - b.quintMin + 1)
        * static_cast<long long>(b.quartMax - b.quartMin + 1);
}

inline bool tileWithinBounds(const TileRect& t, const LutBounds& b) {
    return t.qLo <= t.qHi && t.rLo <= t.rHi
        && t.qLo >= b.quintMin && t.qHi <= b.quintMax
        && t.rLo >= b.quartMin && t.rHi <= b.quartMax;
}

inline bool tilesOverlap(const TileRect& a, const TileRect& b) {
    return a.qLo <= b.qHi && b.qLo <= a.qHi && a.rLo <= b.rHi && b.rLo <= a.rHi;
}

inline bool tileContainsCell(const TileRect& t, int q, int r) {
    return q >= t.qLo && q <= t.qHi && r >= t.rLo && r <= t.rHi;
}

// Origin tile: 1 quint x up to 100 quarts, clamped to bounds.
inline TileRect firstTile(const LutBounds& bounds) {
    TileRect t;
    t.qLo = bounds.quintMin;
    t.qHi = std::min(bounds.quintMin + SLICE_QUINT_CHUNK - 1, bounds.quintMax);
    t.rLo = bounds.quartMin;
    t.rHi = std::min(bounds.quartMin + SLICE_QUART_CHUNK - 1, bounds.quartMax);
    return t;
}

// Walk quart-first within a quint, then the next quint at quartMin.
inline bool nextTileAfter(const LutBounds& bounds, const TileRect& last, TileRect& out) {
    const int nextR = last.rHi + 1;
    if (nextR <= bounds.quartMax) {
        out.qLo = last.qLo;
        out.qHi = std::min(last.qLo + SLICE_QUINT_CHUNK - 1, bounds.quintMax);
        out.rLo = nextR;
        out.rHi = std::min(nextR + SLICE_QUART_CHUNK - 1, bounds.quartMax);
        return true;
    }
    const int nextQ = last.qHi + 1;
    if (nextQ <= bounds.quintMax) {
        out.qLo = nextQ;
        out.qHi = std::min(nextQ + SLICE_QUINT_CHUNK - 1, bounds.quintMax);
        out.rLo = bounds.quartMin;
        out.rHi = std::min(bounds.quartMin + SLICE_QUART_CHUNK - 1, bounds.quartMax);
        return true;
    }
    return false;
}

// If haveLast is false, start at firstTile; otherwise continue after last.
inline bool nextTiles(const LutBounds& bounds, const TileRect& last, bool haveLast, int maxN, std::vector<TileRect>& out) {
    out.clear();
    TileRect cursor{};
    if (!haveLast) {
        cursor = firstTile(bounds);
        out.push_back(cursor);
    } else if (nextTileAfter(bounds, last, cursor)) {
        out.push_back(cursor);
    } else {
        return false;
    }

    while (static_cast<int>(out.size()) < maxN) {
        TileRect n{};
        if (!nextTileAfter(bounds, out.back(), n)) {
            break;
        }
        out.push_back(n);
    }
    return !out.empty();
}

// Same drain loop the worker uses: insert tiles in batches of maxN until the LUT is exhausted.
inline std::vector<TileRect> enumerateAllTiles(const LutBounds& bounds, int batchSize = SLICE_ENQUEUE_BATCH) {
    std::vector<TileRect> all;
    TileRect last{};
    bool haveLast = false;
    for (;;) {
        std::vector<TileRect> batch;
        if (!nextTiles(bounds, last, haveLast, batchSize, batch)) {
            break;
        }
        all.insert(all.end(), batch.begin(), batch.end());
        last = all.back();
        haveLast = true;
    }
    return all;
}

// Last 1x100-or-shorter tile in the LUT (quintMax, final quart strip).
inline TileRect lastTile(const LutBounds& bounds) {
    const int quartSpan = bounds.quartMax - bounds.quartMin + 1;
    const int nQuartTiles = (quartSpan + SLICE_QUART_CHUNK - 1) / SLICE_QUART_CHUNK;
    TileRect t;
    t.qLo = bounds.quintMax;
    t.qHi = bounds.quintMax;
    t.rLo = bounds.quartMin + (nQuartTiles - 1) * SLICE_QUART_CHUNK;
    t.rHi = bounds.quartMax;
    return t;
}

// GpuQuinticFirst loopRanges order: quint, quart, cubic, x, y, z.
inline std::vector<int> makeQuinticFirstSliceLoopRanges(int quintLo, int quintHi, int quartLo, int quartHi) {
    return {
        quintLo, quintHi,
        quartLo, quartHi,
        USE_DEFAULT, USE_DEFAULT,
        USE_DEFAULT, USE_DEFAULT,
        USE_DEFAULT, USE_DEFAULT,
        USE_DEFAULT, USE_DEFAULT,
        USE_DEFAULT, USE_DEFAULT
    };
}

#endif
