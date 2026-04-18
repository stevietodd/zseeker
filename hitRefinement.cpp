#include "hitRefinement.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <strings.h>

#if defined(__GNUC__) && defined(__SIZEOF_FLOAT128__) && __SIZEOF_FLOAT128__ == 16 && !defined(__clang__)
#define ZSEEKER_INTERNAL_HAVE_FLOAT128 1
#include <quadmath.h>
#else
#define ZSEEKER_INTERNAL_HAVE_FLOAT128 0
#endif

static bool envTruthy(const char* v) {
    if (!v || !*v) {
        return false;
    }
    return (std::strcmp(v, "1") == 0) || (strcasecmp(v, "true") == 0) || (strcasecmp(v, "yes") == 0);
}

bool gpuHitRefinementEnabled() {
    return envTruthy(std::getenv("ZSEEKER_REFINE_HITS"));
}

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

static const char* refineMode() {
    const char* m = std::getenv("ZSEEKER_REFINE_MODE");
    return (m && *m) ? m : "ld";
}

static long double lutCoeffLongDouble(int idx, const double* lut) {
    if (idx < 0) {
        return -static_cast<long double>(lut[-idx]);
    }
    return static_cast<long double>(lut[idx]);
}

static long double evalQuinticHornerLongDouble(
    int i5, int i4, int i3, int i2, int i1, int i0, long double c, const double* lut) {
    const long double a5 = lutCoeffLongDouble(i5, lut);
    const long double a4 = lutCoeffLongDouble(i4, lut);
    const long double a3 = lutCoeffLongDouble(i3, lut);
    const long double a2 = lutCoeffLongDouble(i2, lut);
    const long double a1 = lutCoeffLongDouble(i1, lut);
    const long double a0 = lutCoeffLongDouble(i0, lut);
    return (((((a5 * c + a4) * c + a3) * c + a2) * c + a1) * c + a0);
}

#if ZSEEKER_INTERNAL_HAVE_FLOAT128
static __float128 lutCoeffFloat128(int idx, const double* lut) {
    if (idx < 0) {
        return -(__float128)lut[-idx];
    }
    return (__float128)lut[idx];
}

static __float128 evalQuinticHornerFloat128(
    int i5, int i4, int i3, int i2, int i1, int i0, __float128 c, const double* lut) {
    const __float128 a5 = lutCoeffFloat128(i5, lut);
    const __float128 a4 = lutCoeffFloat128(i4, lut);
    const __float128 a3 = lutCoeffFloat128(i3, lut);
    const __float128 a2 = lutCoeffFloat128(i2, lut);
    const __float128 a1 = lutCoeffFloat128(i1, lut);
    const __float128 a0 = lutCoeffFloat128(i0, lut);
    return (((((a5 * c + a4) * c + a3) * c + a2) * c + a1) * c + a0);
}
#endif

std::size_t refineGpuHitsIfConfigured(
    std::vector<int*>* hits,
    double needle,
    double theConst,
    const double* lutDouble) {
    if (!gpuHitRefinementEnabled() || !hits || hits->empty()) {
        return hits ? hits->size() : 0;
    }

    const char* mode = refineMode();
    const std::size_t before = hits->size();

#if ZSEEKER_INTERNAL_HAVE_FLOAT128
    if (strcasecmp(mode, "float128") == 0) {
        const __float128 tol = parseLongDoubleOr("ZSEEKER_REFINE_TOL", 1e-24L);
        const __float128 c = theConst;
        const __float128 ndl = needle;
        std::vector<int*> kept;
        kept.reserve(hits->size());
        for (int* h : *hits) {
            const __float128 v = evalQuinticHornerFloat128(h[0], h[1], h[2], h[3], h[4], h[5], c, lutDouble);
            const __float128 d = v > ndl ? v - ndl : ndl - v;
            if (d < tol) {
                kept.push_back(h);
            } else {
                delete[] h;
            }
        }
        hits->swap(kept);
        std::cout << "Hit refinement (float128): " << before << " -> " << hits->size() << " (tol=" << (double)tol << ")"
                  << std::endl;
        return hits->size();
    }
#else
    if (strcasecmp(mode, "float128") == 0) {
        std::cerr << "ZSEEKER_REFINE_MODE=float128 not available on this toolchain; falling back to ld." << std::endl;
    }
#endif

    // long double (default)
    const long double tol = parseLongDoubleOr("ZSEEKER_REFINE_TOL", 1e-14L);
    const long double c = static_cast<long double>(theConst);
    const long double ndl = static_cast<long double>(needle);
    std::vector<int*> kept;
    kept.reserve(hits->size());
    for (int* h : *hits) {
        const long double v = evalQuinticHornerLongDouble(h[0], h[1], h[2], h[3], h[4], h[5], c, lutDouble);
        const long double d = v > ndl ? v - ndl : ndl - v;
        if (d < tol) {
            kept.push_back(h);
        } else {
            delete[] h;
        }
    }
    hits->swap(kept);
    std::cout << "Hit refinement (long double): " << before << " -> " << hits->size() << " (tol=" << (double)tol << ")"
              << std::endl;
    return hits->size();
}
