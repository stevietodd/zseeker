#ifndef FLOAT128_HPP
#define FLOAT128_HPP

#include <cstddef>
#include <cstdio>

// x86_64 GCC uses __float128 from libquadmath. aarch64 has no libquadmath;
// long double is already IEEE binary128 there.
#if defined(__SIZEOF_FLOAT128__)
using Float128 = __float128;
#define ZS_F128(x) x##Q
#else
using Float128 = long double;
#define ZS_F128(x) x##L
#endif

#if !defined(__CUDACC__)
#if defined(__SIZEOF_FLOAT128__)
#include <quadmath.h>

inline Float128 float128FromString(const char* text, char** = nullptr) {
    return strtoflt128(text, nullptr);
}
inline Float128 float128Sqrt(Float128 value) { return sqrtq(value); }
inline Float128 float128Cbrt(Float128 value) { return cbrtq(value); }
inline Float128 float128Acos(Float128 value) { return acosq(value); }
inline Float128 float128Cos(Float128 value) { return cosq(value); }
inline void formatFloat128(char* buf, std::size_t size, Float128 value) {
    quadmath_snprintf(buf, size, "%.34Qf", value);
}
#else
#include <cmath>
#include <cstdlib>

inline Float128 float128FromString(const char* text, char** = nullptr) {
    return strtold(text, nullptr);
}
inline Float128 float128Sqrt(Float128 value) { return sqrtl(value); }
inline Float128 float128Cbrt(Float128 value) { return cbrtl(value); }
inline Float128 float128Acos(Float128 value) { return acosl(value); }
inline Float128 float128Cos(Float128 value) { return cosl(value); }
inline void formatFloat128(char* buf, std::size_t size, Float128 value) {
    std::snprintf(buf, size, "%.34Lf", value);
}
#endif
#endif

#endif
