#ifndef HIT_REFINEMENT_HPP
#define HIT_REFINEMENT_HPP

#include <cstddef>
#include <vector>

// Optional second-stage filter after GPU "double" hits: re-evaluate the quintic at theConst
// using higher precision than GPU device double (long double Horner, or __float128 when available).
//
// Enable with ZSEEKER_REFINE_HITS=1 (or true/yes).
// ZSEEKER_REFINE_MODE:
//   ld       — long double (default); ZSEEKER_REFINE_TOL defaults to 1e-14
//   float128 — __float128 + libquadmath (GCC/x86_64); ZSEEKER_REFINE_TOL defaults to 1e-24
//
// For arbitrary-precision / algebraic checks beyond this, evaluate elsewhere (e.g. MPFR, Pari) using
// the printed coefficient indices and rationals from the LUT.

bool gpuHitRefinementEnabled();

// If enabled, removes hits that fail |p(theConst) - needle| < tol under the chosen mode.
// Frees coefficient arrays for dropped hits. No-op if disabled.
// Returns the number of hits remaining (same as hits->size() after).
std::size_t refineGpuHitsIfConfigured(
    std::vector<int*>* hits,
    double needle,
    double theConst,
    const double* lutDouble);

#endif
