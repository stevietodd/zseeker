#include "Z3RootFinder.hpp"

#include "lookupTableAccessor.hpp"

#include <iostream>

namespace {

Float128 zeta3Float128() {
    static const Float128 z = float128FromString(
        "1.2020569031595942853997381615114499907649862923404988817922715553", nullptr);
    return z;
}

Float128 lutCoeff(int idx) {
    const int absIdx = idx < 0 ? -idx : idx;
    if (absIdx < 0 || static_cast<std::size_t>(absIdx) >= getLookupTableSize()) {
        return 0;
    }
    const LutRational r = getLookupTableRational()[absIdx];
    const Float128 value = static_cast<Float128>(r.num) / static_cast<Float128>(r.den);
    return idx < 0 ? -value : value;
}

// ax^2 + bx + c = 0. Slot order matches quadraticroot_wolfram.
void quadraticRoots(Float128 a, Float128 b, Float128 c, Float128* root) {
    const Float128 disc = b * b - 4 * a * c;
    if (disc < 0) {
        return;
    }
    if (disc == 0) {
        root[0] = -b / (2 * a);
        return;
    }
    const Float128 s = float128Sqrt(disc);
    root[0] = (-b + s) / (2 * a);
    root[1] = (-b - s) / (2 * a);
}

// ax^3 + bx^2 + cx + d = 0. Slot order matches cubicroot_wolfram.
void cubicRoots(Float128 a, Float128 b, Float128 c, Float128 d, Float128* root) {
    const Float128 a2 = b / a;
    const Float128 a1 = c / a;
    const Float128 a0 = d / a;
    const Float128 a2Sq = a2 * a2;
    const Float128 a2Cu = a2Sq * a2;
    const Float128 bigq = (3 * a1 - a2Sq) / 9;
    const Float128 bigr = (9 * a2 * a1 - 27 * a0 - 2 * a2Cu) / 54;
    const Float128 bigqCu = bigq * bigq * bigq;
    const Float128 bigd = bigqCu + bigr * bigr;

    if (bigq == 0 && bigr == 0 && bigd == 0) {
        root[0] = float128Cbrt(a0);
        return;
    }
    if (bigd <= 0) {
        const Float128 denom = float128Sqrt(-bigqCu);
        if (denom == 0) {
            root[0] = float128Cbrt(a0);
            return;
        }
        Float128 arg = bigr / denom;
        if (arg > 1) {
            arg = 1;
        } else if (arg < -1) {
            arg = -1;
        }
        const Float128 pi = float128Acos(-1);
        const Float128 theta = float128Acos(arg);
        const Float128 scale = 2 * float128Sqrt(-bigq);
        root[0] = scale * float128Cos(theta / 3) - a2 / 3;
        root[1] = scale * float128Cos((theta + 2 * pi) / 3) - a2 / 3;
        root[2] = scale * float128Cos((theta + 4 * pi) / 3) - a2 / 3;
        return;
    }

    const Float128 sd = float128Sqrt(bigd);
    root[0] = float128Cbrt(bigr + sd) + float128Cbrt(bigr - sd) - a2 / 3;
}

} // namespace

Z3Roots Z3RootFinder::rootsFor(const CoeffIndexes& indexes) const {
    Z3Roots roots;
    if (!LUTIndexCalculator{}.inRange(indexes)) {
        std::cerr << "Error: coefficient indexes out of range while solving a zeta(3) cubic" << std::endl;
        return roots;
    }

    const Float128 a3 = lutCoeff(indexes.jk);
    const Float128 a2 = lutCoeff(indexes.mn);
    const Float128 a1 = lutCoeff(indexes.pq);
    const Float128 a0 = lutCoeff(indexes.rs);
    const Float128 constant = a0 - zeta3Float128();

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
