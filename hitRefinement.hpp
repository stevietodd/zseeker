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
// Call this at the search constant (|zroot|); remap odd-degree signs afterward
// if the original zroot was negative.
//
// This pass uses ZSEEKER_REFINE_TOL (default 1e-12). The worker then calls
// refineHitsWithFloat128Ulp at the polished float128 root.
std::size_t refineHitsWithFloat128Precision(
    std::vector<int*>* hits,
    double needle,
    double theConst);

// ζ(5) to float128 precision.
__float128 zeta5Float128();

// Rebuild one real root of (jk)c^3 + (mn)c^2 + (pq)c + (rs) = ζ(3) from LUT indexes.
// Two Newton steps from zrootDouble. Returns false when that polynomial does not
// already meet the stored double.
bool polishZeta3Root(
    int jkIndex,
    int mnIndex,
    int pqIndex,
    int rsIndex,
    double zrootDouble,
    __float128* polishedOut);

// |p(theConst) - needle| against a binary128 ulp of the same coefficient bound
// used for the float and double gates. theConst is the signed polished root.
std::size_t refineHitsWithFloat128Ulp(
    std::vector<int*>* hits,
    __float128 needle,
    __float128 theConst);

// If zroot < 0, the search and float128 pass ran at |zroot|. Flip odd-degree
// LUT indices so each stored hit evaluates at the original zroot:
// p(-c; a) = p(c; a') with a_odd = -a'_odd.
void remapHitsForNegativeZroot(std::vector<int*>* hits, double zroot);

#endif
