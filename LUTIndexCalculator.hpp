#ifndef LUT_INDEX_CALCULATOR_HPP
#define LUT_INDEX_CALCULATOR_HPP

#include <cstdint>

// jk, mn, pq, rs are the c^3, c^2, c, and constant LUT indexes of
// (jk)c^3 + (mn)c^2 + (pq)c + (rs) = ζ(3).
struct CoeffIndexes {
    int jk = 0;
    int mn = 0;
    int pq = 0;
    int rs = 0;
};

// Maps a cubic-root id onto those four indexes.
//
// A run fixes one jk. The id is the 1-based position of (mn, pq, rs) in the
// FindConstantsThatMakeZ3 loop, with rs varying fastest:
//   mn in [-2203, 2203], pq in [-555, 555], rs in [-143, 143].
// Id 145 is mn=-2203, pq=-555, rs=1 when jk is the run's jk (0 for the sweep
// already stored). cubicRootId() returns 0 when mn, pq, or rs is out of range.
class LUTIndexCalculator {
public:
    static constexpr int kJkMin = -6087;
    static constexpr int kJkMax = 6087;
    static constexpr int kMnMin = -2203;
    static constexpr int kMnMax = 2203;
    static constexpr int kPqMin = -555;
    static constexpr int kPqMax = 555;
    static constexpr int kRsMin = -143;
    static constexpr int kRsMax = 143;

    static constexpr int kMnCount = kMnMax - kMnMin + 1;
    static constexpr int kPqCount = kPqMax - kPqMin + 1;
    static constexpr int kRsCount = kRsMax - kRsMin + 1;
    static constexpr uint32_t kIdsPerJk =
        static_cast<uint32_t>(kMnCount) * static_cast<uint32_t>(kPqCount) * static_cast<uint32_t>(kRsCount);

    bool inRange(const CoeffIndexes& indexes) const;
    bool fromCubicRootId(uint32_t id, int jkIndex, CoeffIndexes* out) const;
    uint32_t cubicRootId(int mn, int pq, int rs) const;
};

inline bool LUTIndexCalculator::inRange(const CoeffIndexes& indexes) const {
    return indexes.jk >= kJkMin && indexes.jk <= kJkMax
        && indexes.mn >= kMnMin && indexes.mn <= kMnMax
        && indexes.pq >= kPqMin && indexes.pq <= kPqMax
        && indexes.rs >= kRsMin && indexes.rs <= kRsMax;
}

inline bool LUTIndexCalculator::fromCubicRootId(uint32_t id, int jkIndex, CoeffIndexes* out) const {
    if (out == nullptr || id == 0 || id > kIdsPerJk || jkIndex < kJkMin || jkIndex > kJkMax) {
        return false;
    }

    const uint32_t offset = id - 1;
    const uint32_t rsPos = offset % static_cast<uint32_t>(kRsCount);
    const uint32_t pqPos = (offset / static_cast<uint32_t>(kRsCount)) % static_cast<uint32_t>(kPqCount);
    const uint32_t mnPos = offset / (static_cast<uint32_t>(kRsCount) * static_cast<uint32_t>(kPqCount));

    *out = CoeffIndexes{
        jkIndex,
        kMnMin + static_cast<int>(mnPos),
        kPqMin + static_cast<int>(pqPos),
        kRsMin + static_cast<int>(rsPos),
    };
    return true;
}

inline uint32_t LUTIndexCalculator::cubicRootId(int mn, int pq, int rs) const {
    if (mn < kMnMin || mn > kMnMax || pq < kPqMin || pq > kPqMax || rs < kRsMin || rs > kRsMax) {
        return 0;
    }
    const uint32_t mnPos = static_cast<uint32_t>(mn - kMnMin);
    const uint32_t pqPos = static_cast<uint32_t>(pq - kPqMin);
    const uint32_t rsPos = static_cast<uint32_t>(rs - kRsMin);
    return (mnPos * static_cast<uint32_t>(kPqCount) + pqPos) * static_cast<uint32_t>(kRsCount) + rsPos + 1;
}

#endif
