#include "Z3RootFinder.hpp"

#include "lookupTableAccessor.hpp"

#include <iostream>

#include <quadmath.h>

namespace {

__float128 zeta3Float128() {
    static const __float128 z = strtoflt128(
        "1.2020569031595942853997381615114499907649862923404988817922715553", nullptr);
    return z;
}

__float128 lutCoeff(int idx) {
    const int absIdx = idx < 0 ? -idx : idx;
    if (absIdx < 0 || static_cast<std::size_t>(absIdx) >= getLookupTableSize()) {
        return 0;
    }
    const LutRational r = getLookupTableRational()[absIdx];
    const __float128 value = static_cast<__float128>(r.num) / static_cast<__float128>(r.den);
    return idx < 0 ? -value : value;
}

// ax^2 + bx + c = 0. Slot order matches quadraticroot_wolfram.
void quadraticRoots(__float128 a, __float128 b, __float128 c, __float128* root) {
    const __float128 disc = b * b - 4 * a * c;
    if (disc < 0) {
        return;
    }
    if (disc == 0) {
        root[0] = -b / (2 * a);
        return;
    }
    const __float128 s = sqrtq(disc);
    root[0] = (-b + s) / (2 * a);
    root[1] = (-b - s) / (2 * a);
}

// ax^3 + bx^2 + cx + d = 0. Slot order matches cubicroot_wolfram.
void cubicRoots(__float128 a, __float128 b, __float128 c, __float128 d, __float128* root) {
    const __float128 a2 = b / a;
    const __float128 a1 = c / a;
    const __float128 a0 = d / a;
    const __float128 a2Sq = a2 * a2;
    const __float128 a2Cu = a2Sq * a2;
    const __float128 bigq = (3 * a1 - a2Sq) / 9;
    const __float128 bigr = (9 * a2 * a1 - 27 * a0 - 2 * a2Cu) / 54;
    const __float128 bigqCu = bigq * bigq * bigq;
    const __float128 bigd = bigqCu + bigr * bigr;

    if (bigq == 0 && bigr == 0 && bigd == 0) {
        root[0] = cbrtq(a0);
        return;
    }
    if (bigd <= 0) {
        const __float128 denom = sqrtq(-bigqCu);
        if (denom == 0) {
            root[0] = cbrtq(a0);
            return;
        }
        __float128 arg = bigr / denom;
        if (arg > 1) {
            arg = 1;
        } else if (arg < -1) {
            arg = -1;
        }
        const __float128 pi = acosq(-1);
        const __float128 theta = acosq(arg);
        const __float128 scale = 2 * sqrtq(-bigq);
        root[0] = scale * cosq(theta / 3) - a2 / 3;
        root[1] = scale * cosq((theta + 2 * pi) / 3) - a2 / 3;
        root[2] = scale * cosq((theta + 4 * pi) / 3) - a2 / 3;
        return;
    }

    const __float128 sd = sqrtq(bigd);
    root[0] = cbrtq(bigr + sd) + cbrtq(bigr - sd) - a2 / 3;
}

} // namespace

Z3Roots Z3RootFinder::rootsFor(const CoeffIndexes& indexes) const {
    Z3Roots roots;
    if (!LUTIndexCalculator{}.inRange(indexes)) {
        std::cerr << "Error: coefficient indexes out of range while solving a zeta(3) cubic" << std::endl;
        return roots;
    }

    const __float128 a3 = lutCoeff(indexes.jk);
    const __float128 a2 = lutCoeff(indexes.mn);
    const __float128 a1 = lutCoeff(indexes.pq);
    const __float128 a0 = lutCoeff(indexes.rs);
    const __float128 constant = a0 - zeta3Float128();

    if (a3 == 0 && a2 == 0 && a1 == 0) {
        return roots;
    }
    if (a3 == 0 && a2 == 0) {
        roots.root[0] = -constant / a1;
        return roots;
    }
    if (a3 == 0) {
        quadraticRoots(a2, a1, constant, roots.root);
        return roots;
    }
    cubicRoots(a3, a2, a1, constant, roots.root);
    return roots;
}

Z3Roots Z3RootFinder::rootsForId(uint32_t cubicRootId, int jkIndex) const {
    CoeffIndexes indexes;
    if (!LUTIndexCalculator{}.fromCubicRootId(cubicRootId, jkIndex, &indexes)) {
        std::cerr << "Error: cubic_root_id " << cubicRootId << " is outside the coefficient id space" << std::endl;
        return {};
    }
    return rootsFor(indexes);
}
