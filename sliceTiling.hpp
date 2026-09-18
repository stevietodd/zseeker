#ifndef SLICE_TILING_HPP
#define SLICE_TILING_HPP

#include "math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
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

struct SlotPlan {
    int quintChunk;
    int quartChunk;
    LutBounds bounds;
    TileRect probe;
};

struct SliceRegion {
    int q0;
    int q1;
    int r0;
    int r1;
};

inline LutBounds lutBoundsForChecker(bool positiveOnlyQuint) {
    return {
        positiveOnlyQuint ? 0 : SLICE_QUINT_MIN,
        SLICE_QUINT_MAX,
        SLICE_QUART_MIN,
        SLICE_QUART_MAX
    };
}

inline TileRect makeProbeTile(const LutBounds& bounds) {
    TileRect probe;
    probe.qLo = bounds.quintMin;
    probe.qHi = std::min(bounds.quintMin + DEFAULT_SLICE_QUINT_CHUNK - 1, bounds.quintMax);
    probe.rLo = bounds.quartMin;
    probe.rHi = std::min(bounds.quartMin + DEFAULT_SLICE_QUART_CHUNK - 1, bounds.quartMax);
    return probe;
}

inline SlotPlan makeSlotPlan(const LutBounds& bounds, int quintChunk, int quartChunk) {
    SlotPlan plan;
    plan.quintChunk = std::max(1, quintChunk);
    plan.quartChunk = std::max(1, quartChunk);
    plan.bounds = bounds;
    plan.probe = makeProbeTile(bounds);
    return plan;
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

inline std::array<SliceRegion, 2> remainingRegions(const SlotPlan& p) {
    return {{
        {p.probe.qLo, p.probe.qHi, p.probe.rHi + 1, p.bounds.quartMax},
        {p.probe.qHi + 1, p.bounds.quintMax, p.bounds.quartMin, p.bounds.quartMax}
    }};
}

inline bool regionNonEmpty(const SliceRegion& reg) {
    return reg.q0 <= reg.q1 && reg.r0 <= reg.r1;
}

inline bool firstTileInRegion(const SlotPlan& p, const SliceRegion& reg, TileRect& out) {
    if (!regionNonEmpty(reg)) {
        return false;
    }
    out.qLo = reg.q0;
    out.qHi = std::min(reg.q0 + p.quintChunk - 1, reg.q1);
    out.rLo = reg.r0;
    out.rHi = std::min(reg.r0 + p.quartChunk - 1, reg.r1);
    return true;
}

inline bool nextTileAfterInRegion(const SlotPlan& p, const SliceRegion& reg, const TileRect& last, TileRect& out) {
    const int nextR = last.rHi + 1;
    if (nextR <= reg.r1) {
        out.qLo = last.qLo;
        out.qHi = std::min(last.qLo + p.quintChunk - 1, reg.q1);
        out.rLo = nextR;
        out.rHi = std::min(nextR + p.quartChunk - 1, reg.r1);
        return true;
    }
    const int nextQ = last.qHi + 1;
    if (nextQ <= reg.q1) {
        out.qLo = nextQ;
        out.qHi = std::min(nextQ + p.quintChunk - 1, reg.q1);
        out.rLo = reg.r0;
        out.rHi = std::min(reg.r0 + p.quartChunk - 1, reg.r1);
        return true;
    }
    return false;
}

inline int regionIndexOfRemaining(const SlotPlan& p, const TileRect& t) {
    if (t.qLo >= p.probe.qLo && t.qLo <= p.probe.qHi && t.rLo > p.probe.rHi) {
        return 0;
    }
    if (t.qLo > p.probe.qHi) {
        return 1;
    }
    return -1;
}

inline bool nextRemainingTiles(const SlotPlan& p, const TileRect& last, bool lastIsProbe, int maxN, std::vector<TileRect>& out) {
    out.clear();
    const auto regs = remainingRegions(p);
    int ri = 0;
    TileRect cursor{};

    auto seekFirstFrom = [&](int startRi) {
        ri = startRi;
        while (ri < 2 && !firstTileInRegion(p, regs[ri], cursor)) {
            ++ri;
        }
        return ri < 2;
    };

    if (lastIsProbe) {
        if (!seekFirstFrom(0)) {
            return false;
        }
        out.push_back(cursor);
    } else {
        ri = regionIndexOfRemaining(p, last);
        if (ri < 0) {
            if (!seekFirstFrom(0)) {
                return false;
            }
            out.push_back(cursor);
        } else if (nextTileAfterInRegion(p, regs[ri], last, cursor)) {
            out.push_back(cursor);
        } else if (!seekFirstFrom(ri + 1)) {
            return false;
        } else {
            out.push_back(cursor);
        }
    }

    while (static_cast<int>(out.size()) < maxN) {
        TileRect n{};
        if (nextTileAfterInRegion(p, regs[ri], out.back(), n)) {
            out.push_back(n);
            continue;
        }
        ++ri;
        if (ri >= 2 || !firstTileInRegion(p, regs[ri], n)) {
            break;
        }
        out.push_back(n);
    }
    return !out.empty();
}

// Same drain loop the worker uses: probe, then remaining tiles in batches.
inline std::vector<TileRect> enumerateAllTiles(const SlotPlan& p, int batchSize = SLICE_ENQUEUE_BATCH) {
    std::vector<TileRect> all;
    all.push_back(p.probe);
    TileRect last = p.probe;
    bool lastIsProbe = true;
    for (;;) {
        std::vector<TileRect> batch;
        if (!nextRemainingTiles(p, last, lastIsProbe, batchSize, batch)) {
            break;
        }
        all.insert(all.end(), batch.begin(), batch.end());
        last = all.back();
        lastIsProbe = false;
    }
    return all;
}

inline std::pair<int, int> chunksFromProbe(double probeSec, const LutBounds& bounds, double targetSec) {
    const long long probeCells = static_cast<long long>(DEFAULT_SLICE_QUINT_CHUNK) * DEFAULT_SLICE_QUART_CHUNK;
    if (probeSec < 1e-9) {
        probeSec = 1e-9;
    }
    if (targetSec <= 0) {
        targetSec = TARGET_SLICE_RUNTIME_SEC;
    }
    long long cells = static_cast<long long>(std::llround(targetSec / (probeSec / static_cast<double>(probeCells))));
    if (cells < probeCells) {
        cells = probeCells;
    }

    const int quartSpan = bounds.quartMax - bounds.quartMin + 1;
    const int quintSpan = bounds.quintMax - bounds.quintMin + 1;
    const int maxQuart = std::min(quartSpan, SLICE_MAX_QUART_RANGE);
    const int maxQuint = std::min(quintSpan, SLICE_MAX_QUINT_RANGE);

    int quartChunk = DEFAULT_SLICE_QUART_CHUNK;
    int quintChunk = DEFAULT_SLICE_QUINT_CHUNK;
    if (cells <= maxQuart) {
        quartChunk = static_cast<int>(cells);
        quintChunk = 1;
    } else {
        quartChunk = maxQuart;
        long long q = (cells + maxQuart - 1) / maxQuart;
        if (q < 1) {
            q = 1;
        }
        if (q > maxQuint) {
            q = maxQuint;
        }
        quintChunk = static_cast<int>(q);
    }
    return {quintChunk, quartChunk};
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
