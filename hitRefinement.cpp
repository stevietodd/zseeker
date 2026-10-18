#include "hitRefinement.hpp"

#include "lookupTableAccessor.hpp"
#include "math.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <strings.h>

#include <quadmath.h>

constexpr int kMaxDen = 1000;

static __float128 invDenF128[kMaxDen + 1];
static bool invDenReady = false;

static long double parseLongDoubleOr(const char* name, long double defVal) {
    const char* v = std::getenv(name);
    if (!v || !*v) {
        return defVal;
    }
    char* end = nullptr;
    long double x = std::strtold(v, &end);
    if (end == v) {
        std::cerr << "Warning: invalid " << name << "; using default " << defVal << std::endl;
        return defVal;
    }
    return x;
}

static void ensureInvDenFloat128() {
    if (invDenReady) {
        return;
    }
    invDenF128[0] = 0.0Q;
    for (int d = 1; d <= kMaxDen; d++) {
        invDenF128[d] = 1.0Q / static_cast<__float128>(d);
    }
    invDenReady = true;
}

static __float128 lutCoeffFloat128(int idx, const LutRational* rat) {
    ensureInvDenFloat128();
    const int absIdx = idx < 0 ? -idx : idx;
    const LutRational r = rat[absIdx];
    const __float128 v = static_cast<__float128>(r.num) * invDenF128[r.den];
    return idx < 0 ? -v : v;
}

static __float128 evalQuinticDirectFloat128(
    int i5,
    int i4,
    int i3,
    int i2,
    int i1,
    int i0,
    __float128 c1,
    __float128 c2,
    __float128 c3,
    __float128 c4,
    __float128 c5,
    const LutRational* rat) {
    return lutCoeffFloat128(i0, rat) + lutCoeffFloat128(i1, rat) * c1 + lutCoeffFloat128(i2, rat) * c2
         + lutCoeffFloat128(i3, rat) * c3 + lutCoeffFloat128(i4, rat) * c4 + lutCoeffFloat128(i5, rat) * c5;
}

static __float128 zeta3Float128Value() {
    static const __float128 z = strtoflt128(
        "1.2020569031595942853997381615114499907649862923404988817922715553", nullptr);
    return z;
}

static double quinticCoeffBound(double absC) {
    const double c2 = absC * absC;
    const double c3 = c2 * absC;
    const double c4 = c2 * c2;
    const double c5 = c4 * absC;
    const double sum = 1000.0 * std::fabs(c5) + 500.0 * std::fabs(c4) + 100.0 * std::fabs(c3)
        + 60.0 * std::fabs(c2) + 30.0 * std::fabs(absC) + 15.0;
    return std::max(1000.0, sum);
}

static bool lutIndexInRange(int idx) {
    const int absIdx = idx < 0 ? -idx : idx;
    return absIdx >= 0 && static_cast<std::size_t>(absIdx) < getLookupTableSize();
}

static std::size_t filterHits(
    std::vector<int*>* hits,
    __float128 needle,
    __float128 theConst,
    __float128 tol,
    const char* label) {
    if (!hits || hits->empty()) {
        return hits ? hits->size() : 0;
    }

    const LutRational* rat = getLookupTableRational();
    const std::size_t before = hits->size();
    const __float128 c1 = theConst;
    const __float128 c2 = c1 * c1;
    const __float128 c3 = c2 * c1;
    const __float128 c4 = c2 * c2;
    const __float128 c5 = c4 * c1;
    std::vector<int*> kept;
    kept.reserve(hits->size());
    for (int* h : *hits) {
        const __float128 v =
            evalQuinticDirectFloat128(h[0], h[1], h[2], h[3], h[4], h[5], c1, c2, c3, c4, c5, rat);
        const __float128 d = v > needle ? v - needle : needle - v;
        if (d < tol) {
            kept.push_back(h);
        } else {
            delete[] h;
        }
    }
    hits->swap(kept);
    std::cout << "Hit refinement (" << label << "): " << before << " -> " << hits->size()
              << " (tol=" << static_cast<double>(tol) << ")" << std::endl;
    return hits->size();
}

__float128 zeta5Float128() {
    static const __float128 z = strtoflt128(
        "1.0369277551433699263313654864570341680570809195019128119741926779", nullptr);
    return z;
}

bool polishZeta3Root(
    int jkIndex,
    int mnIndex,
    int pqIndex,
    int rsIndex,
    double zrootDouble,
    __float128* polishedOut) {
    if (!polishedOut) {
        return false;
    }
    if (!lutIndexInRange(jkIndex) || !lutIndexInRange(mnIndex) || !lutIndexInRange(pqIndex)
        || !lutIndexInRange(rsIndex)) {
        std::cerr << "Error: LUT index out of range while polishing zroot" << std::endl;
        return false;
    }

    const LutRational* rat = getLookupTableRational();
    const __float128 a3 = lutCoeffFloat128(jkIndex, rat);
    const __float128 a2 = lutCoeffFloat128(mnIndex, rat);
    const __float128 a1 = lutCoeffFloat128(pqIndex, rat);
    const __float128 a0 = lutCoeffFloat128(rsIndex, rat);
    const __float128 zeta3 = zeta3Float128Value();
    auto valueAt = [&](__float128 c) {
        return ((a3 * c + a2) * c + a1) * c + a0 - zeta3;
    };
    auto derivAt = [&](__float128 c) {
        return (3 * a3 * c + 2 * a2) * c + a1;
    };

    __float128 c = static_cast<__float128>(zrootDouble);
    const __float128 initial = valueAt(c);
    const __float128 absInitial = initial < 0 ? -initial : initial;
    if (absInitial > 1.0e-8Q) {
        std::cerr << "Stored zroot does not satisfy (jk)c^3 + (mn)c^2 + (pq)c + (rs) = zeta(3)"
                  << " (residual=" << static_cast<double>(absInitial)
                  << "). Keeping the double root." << std::endl;
        return false;
    }

    for (int step = 0; step < 2; ++step) {
        const __float128 deriv = derivAt(c);
        if (deriv == 0) {
            std::cerr << "Error: derivative vanished while polishing zroot" << std::endl;
            return false;
        }
        c = c - valueAt(c) / deriv;
    }
    *polishedOut = c;
    return true;
}

std::size_t refineHitsWithFloat128Precision(
    std::vector<int*>* hits,
    double needle,
    double theConst) {
    // default tol used to be 1e-24, but I increased it to 1e-12 to be more lenient
    const __float128 tol = static_cast<__float128>(parseLongDoubleOr("ZSEEKER_REFINE_TOL", 1e-12L));
    return filterHits(
        hits,
        static_cast<__float128>(needle),
        static_cast<__float128>(theConst),
        tol,
        "float128 rational");
}

std::size_t refineHitsWithFloat128Ulp(
    std::vector<int*>* hits,
    __float128 needle,
    __float128 theConst) {
    const double absC = std::fabs(static_cast<double>(theConst));
    const double bound = quinticCoeffBound(absC);
    const __float128 tol = static_cast<__float128>(getFloat128PrecisionBasedOnMaxValue(bound));
    return filterHits(hits, needle, theConst, tol, "float128 ulp");
}

void remapHitsForNegativeZroot(std::vector<int*>* hits, double zroot) {
    if (!hits || zroot >= 0.0) {
        return;
    }
    for (int* h : *hits) {
        h[0] = -h[0];
        h[2] = -h[2];
        h[4] = -h[4];
    }
}
