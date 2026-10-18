#ifndef Z3_ROOT_FINDER_HPP
#define Z3_ROOT_FINDER_HPP

#include "LUTIndexCalculator.hpp"

#include <cstdint>

// Real roots of (jk)c^3 + (mn)c^2 + (pq)c + (rs) = ζ(3), in float128.
// root[i] == 0 means that real root does not exist. Slot order matches
// cubicroot_wolfram / quadraticroot_wolfram, so root[0], root[1], root[2]
// are zroot1, zroot2, zroot3.
struct Z3Roots {
    __float128 root[3] = {0, 0, 0};

    double asDouble(int slot) const {
        return static_cast<double>(root[slot]);
    }
};

class Z3RootFinder {
public:
    Z3Roots rootsFor(const CoeffIndexes& indexes) const;
    Z3Roots rootsForId(uint32_t cubicRootId, int jkIndex) const;
};

#endif
