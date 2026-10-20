#include <gtest/gtest.h>

#include "../LUTIndexCalculator.hpp"
#include "../Z3RootFinder.hpp"
#include "../lookupTableAccessor.hpp"

#include <quadmath.h>

namespace {

int indexOf(int num, int den) {
    const LutRational* rat = getLookupTableRational();
    const std::size_t n = getLookupTableSize();
    for (std::size_t i = 0; i < n; ++i) {
        if (rat[i].num == num && rat[i].den == den) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

__float128 absDiff(__float128 a, __float128 b) {
    const __float128 d = a - b;
    return d < 0 ? -d : d;
}

__float128 residual(const CoeffIndexes& indexes, __float128 c) {
    auto coeff = [](int idx) {
        const int absIdx = idx < 0 ? -idx : idx;
        const LutRational r = getLookupTableRational()[absIdx];
        const __float128 value = static_cast<__float128>(r.num) / static_cast<__float128>(r.den);
        return idx < 0 ? -value : value;
    };
    const __float128 zeta3 = strtoflt128(
        "1.2020569031595942853997381615114499907649862923404988817922715553", nullptr);
    const __float128 a3 = coeff(indexes.jk);
    const __float128 a2 = coeff(indexes.mn);
    const __float128 a1 = coeff(indexes.pq);
    const __float128 a0 = coeff(indexes.rs);
    return ((a3 * c + a2) * c + a1) * c + a0 - zeta3;
}

} // namespace

TEST(LUTIndexCalculator, Id145IsTheFirstQuadraticSweepRow) {
    LUTIndexCalculator calc;
    CoeffIndexes indexes;
    ASSERT_TRUE(calc.fromCubicRootId(145, 0, &indexes));
    EXPECT_EQ(0, indexes.jk);
    EXPECT_EQ(-2203, indexes.mn);
    EXPECT_EQ(-555, indexes.pq);
    EXPECT_EQ(1, indexes.rs);
    EXPECT_EQ(145u, calc.cubicRootId(indexes.mn, indexes.pq, indexes.rs));
}

TEST(LUTIndexCalculator, SweepCornersRoundTrip) {
    LUTIndexCalculator calc;
    EXPECT_EQ(LUTIndexCalculator::kIdsPerJk, 1405202799u);

    CoeffIndexes first;
    ASSERT_TRUE(calc.fromCubicRootId(1, -6087, &first));
    EXPECT_EQ(-6087, first.jk);
    EXPECT_EQ(LUTIndexCalculator::kMnMin, first.mn);
    EXPECT_EQ(LUTIndexCalculator::kPqMin, first.pq);
    EXPECT_EQ(LUTIndexCalculator::kRsMin, first.rs);
    EXPECT_EQ(1u, calc.cubicRootId(first.mn, first.pq, first.rs));

    CoeffIndexes last;
    ASSERT_TRUE(calc.fromCubicRootId(LUTIndexCalculator::kIdsPerJk, 6087, &last));
    EXPECT_EQ(6087, last.jk);
    EXPECT_EQ(LUTIndexCalculator::kMnMax, last.mn);
    EXPECT_EQ(LUTIndexCalculator::kPqMax, last.pq);
    EXPECT_EQ(LUTIndexCalculator::kRsMax, last.rs);
    EXPECT_EQ(LUTIndexCalculator::kIdsPerJk, calc.cubicRootId(last.mn, last.pq, last.rs));
}

TEST(LUTIndexCalculator, OutOfRangeIdIsRejected) {
    LUTIndexCalculator calc;
    CoeffIndexes indexes;
    EXPECT_FALSE(calc.fromCubicRootId(0, 0, &indexes));
    EXPECT_FALSE(calc.fromCubicRootId(LUTIndexCalculator::kIdsPerJk + 1, 0, &indexes));
    EXPECT_FALSE(calc.fromCubicRootId(1, 6088, &indexes));
    EXPECT_EQ(0u, calc.cubicRootId(-2203, -555, 144));
    EXPECT_EQ(0u, calc.cubicRootId(-2204, -555, 1));
}

TEST(Z3RootFinder, Id145MatchesKnownQuadraticRoots) {
    const Z3Roots roots = Z3RootFinder{}.rootsForId(145, 0);
    const __float128 expected1 = strtoflt128("-0.7535813351441342316834305500884110381431", nullptr);
    const __float128 expected2 = strtoflt128("-0.2636600441662105959027763464633130997879", nullptr);
    EXPECT_LT(static_cast<double>(absDiff(roots.root[0], expected1)), 1e-30);
    EXPECT_LT(static_cast<double>(absDiff(roots.root[1], expected2)), 1e-30);
    EXPECT_TRUE(roots.root[2] == 0);
}

TEST(Z3RootFinder, LinearRootIsZeta3AndMissingSlotsStayZero) {
    // (1)c + 0 = ζ(3) → c = ζ(3). pq index 1 is the LUT entry 1/1.
    ASSERT_EQ(1, getLookupTableRational()[1].num);
    ASSERT_EQ(1, getLookupTableRational()[1].den);
    const Z3Roots roots = Z3RootFinder{}.rootsFor(CoeffIndexes{0, 0, 1, 0});
    const __float128 zeta3 = strtoflt128(
        "1.2020569031595942853997381615114499907649862923404988817922715553", nullptr);
    EXPECT_TRUE(absDiff(roots.root[0], zeta3) == 0);
    EXPECT_TRUE(roots.root[1] == 0);
    EXPECT_TRUE(roots.root[2] == 0);
}

TEST(Z3RootFinder, NegativeDiscriminantHasNoRealRoots) {
    // c^2 + 2 = ζ(3). rs index 3 is the LUT entry 2/1.
    ASSERT_EQ(2, getLookupTableRational()[3].num);
    ASSERT_EQ(1, getLookupTableRational()[3].den);
    const Z3Roots roots = Z3RootFinder{}.rootsFor(CoeffIndexes{0, 1, 0, 3});
    EXPECT_TRUE(roots.root[0] == 0);
    EXPECT_TRUE(roots.root[1] == 0);
    EXPECT_TRUE(roots.root[2] == 0);
}

TEST(Z3RootFinder, ThreeRealCubicRootsSatisfyThePolynomial) {
    const int fourOverOne = indexOf(4, 1);
    ASSERT_GT(fourOverOne, 0);
    // c^3 - 4c = ζ(3) has three real roots.
    const CoeffIndexes indexes{1, 0, -fourOverOne, 0};
    const Z3Roots roots = Z3RootFinder{}.rootsFor(indexes);
    for (int slot = 0; slot < 3; ++slot) {
        EXPECT_TRUE(roots.root[slot] != 0) << "slot " << slot;
        EXPECT_LT(static_cast<double>(absDiff(residual(indexes, roots.root[slot]), 0)), 1e-30)
            << "slot " << slot;
    }
}
