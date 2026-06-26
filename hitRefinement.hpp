#ifndef HIT_REFINEMENT_HPP
#define HIT_REFINEMENT_HPP

#include <cstddef>
#include <vector>

// Second-stage filter after GPU "double" hits: re-evaluate the quintic at theConst using LUT
// coefficient rationals (num/den) in __float128 (libquadmath on GCC/x86_64).
//
// ZSEEKER_REFINE_TOL defaults to 1e-12. Used to be 1e-24 from Cursor but I increased it to 1e-12 to be more lenient
//
// Coefficients are computed as sign * num * (1/den) from the compile-time rational LUT, not
// from precomputed double approximations.

// Removes hits that fail |p(theConst) - needle| < tol.
// Frees coefficient arrays for dropped hits. No-op if hits is null or empty.
// Returns the number of hits remaining (same as hits->size() after).
std::size_t refineHitsWithFloat128Precision(
    std::vector<int*>* hits,
    double needle,
    double theConst);

#endif
