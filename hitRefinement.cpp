#include "hitRefinement.hpp"

#include "lookupTableAccessor.hpp"

#include <cstdlib>
#include <iostream>
#include <strings.h>

#include <quadmath.h>

namespace {

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

} // namespace

std::size_t refineHitsWithFloat128Precision(
    std::vector<int*>* hits,
    double needle,
    double theConst) {
    if (!hits || hits->empty()) {
        return hits ? hits->size() : 0;
    }

    const LutRational* rat = getLookupTableRational();
    const std::size_t before = hits->size();

	// default tol used to be 1e-24, but I increased it to 1e-12 to be more lenient
    const __float128 tol = parseLongDoubleOr("ZSEEKER_REFINE_TOL", 1e-12L);
    const __float128 c1 = static_cast<__float128>(theConst);
    const __float128 c2 = c1 * c1;
    const __float128 c3 = c2 * c1;
    const __float128 c4 = c2 * c2;
    const __float128 c5 = c4 * c1;
    const __float128 ndl = static_cast<__float128>(needle);
    std::vector<int*> kept;
    kept.reserve(hits->size());
    for (int* h : *hits) {
        const __float128 v =
            evalQuinticDirectFloat128(h[0], h[1], h[2], h[3], h[4], h[5], c1, c2, c3, c4, c5, rat);
        const __float128 d = v > ndl ? v - ndl : ndl - v;
        if (d < tol) {
            kept.push_back(h);
        } else {
            delete[] h;
        }
    }
    hits->swap(kept);
    std::cout << "Hit refinement (float128 rational): " << before << " -> " << hits->size()
              << " (tol=" << (double)tol << ")" << std::endl;
    return hits->size();
}
