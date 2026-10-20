#include <gtest/gtest.h>
#include "../hitRefinement.hpp"

#include <cmath>
#include <cstdlib>
#include <vector>

#include <quadmath.h>

namespace {

int* makeHit(int i5, int i4, int i3, int i2, int i1, int i0) {
	return new int[6]{i5, i4, i3, i2, i1, i0};
}

void freeHits(std::vector<int*>* hits) {
	for (int* h : *hits) {
		delete[] h;
	}
	hits->clear();
}

} // namespace

class HitRefinementTestSuite : public ::testing::Test {
protected:
	void SetUp() override {
		unsetenv("ZSEEKER_REFINE_TOL");
	}
};

TEST_F(HitRefinementTestSuite, NullAndEmptyAreNoOps) {
	EXPECT_EQ(0u, refineHitsWithFloat128Precision(nullptr, 1.0, 0.5));

	std::vector<int*> empty;
	EXPECT_EQ(0u, refineHitsWithFloat128Precision(&empty, 1.0, 0.5));
}

TEST_F(HitRefinementTestSuite, KeepsExactHitAndDropsMiss) {
	// LUT[0] = 0, LUT[1] = 1. Hits are (i5,i4,i3,i2,i1,i0).
	// {0,0,0,0,1,0} => p(c) = c. With c = 2, needle = 2 this is exact.
	// {0,0,0,0,0,1} => p(c) = 1, which is far from 2.
	std::vector<int*> hits;
	hits.push_back(makeHit(0, 0, 0, 0, 1, 0));
	hits.push_back(makeHit(0, 0, 0, 0, 0, 1));

	EXPECT_EQ(1u, refineHitsWithFloat128Precision(&hits, 2.0, 2.0));
	ASSERT_EQ(1u, hits.size());
	EXPECT_EQ(0, hits[0][0]);
	EXPECT_EQ(0, hits[0][1]);
	EXPECT_EQ(0, hits[0][2]);
	EXPECT_EQ(0, hits[0][3]);
	EXPECT_EQ(1, hits[0][4]);
	EXPECT_EQ(0, hits[0][5]);

	freeHits(&hits);
}

TEST_F(HitRefinementTestSuite, NegativeLutIndexEvaluatesNegatedCoeff) {
	// {0,0,0,0,-1,0} => p(c) = -c. With c = 3, needle = -3 this is exact.
	std::vector<int*> hits;
	hits.push_back(makeHit(0, 0, 0, 0, -1, 0));

	EXPECT_EQ(1u, refineHitsWithFloat128Precision(&hits, -3.0, 3.0));
	ASSERT_EQ(1u, hits.size());
	EXPECT_EQ(-1, hits[0][4]);

	freeHits(&hits);
}

TEST_F(HitRefinementTestSuite, RemapFlipsOddDegreesOnlyWhenZrootIsNegative) {
	std::vector<int*> hits;
	hits.push_back(makeHit(5, 4, 3, 2, 1, 0));

	remapHitsForNegativeZroot(&hits, 0.5);
	EXPECT_EQ(5, hits[0][0]);
	EXPECT_EQ(4, hits[0][1]);
	EXPECT_EQ(3, hits[0][2]);
	EXPECT_EQ(2, hits[0][3]);
	EXPECT_EQ(1, hits[0][4]);
	EXPECT_EQ(0, hits[0][5]);

	remapHitsForNegativeZroot(&hits, -0.5);
	EXPECT_EQ(-5, hits[0][0]);
	EXPECT_EQ(4, hits[0][1]);
	EXPECT_EQ(-3, hits[0][2]);
	EXPECT_EQ(2, hits[0][3]);
	EXPECT_EQ(-1, hits[0][4]);
	EXPECT_EQ(0, hits[0][5]);

	freeHits(&hits);
}

TEST_F(HitRefinementTestSuite, UlpGateKeepsExactValueAndDropsNearbyMiss) {
	std::vector<int*> hits;
	hits.push_back(makeHit(0, 0, 0, 0, 1, 0));
	EXPECT_EQ(1u, refineHitsWithFloat128Ulp(&hits, 2.0Q, 2.0Q));
	ASSERT_EQ(1u, hits.size());
	freeHits(&hits);

	hits.push_back(makeHit(0, 0, 0, 0, 1, 0));
	EXPECT_EQ(0u, refineHitsWithFloat128Ulp(&hits, 2.0Q + 1.0e-20Q, 2.0Q));
	EXPECT_TRUE(hits.empty());
}

TEST_F(HitRefinementTestSuite, PolishRoot145QuadraticMovesDoubleByLessThanOneUlp) {
	auto closeTo = [](__float128 got, const char* expected) {
		const __float128 target = strtoflt128(expected, nullptr);
		const __float128 err = got > target ? got - target : target - got;
		EXPECT_LT(static_cast<double>(err), 1e-30);
	};

	__float128 polished = 0;
	ASSERT_TRUE(polishZeta3Root(0, -2203, -555, 1, -0.7535813351441343, &polished));
	closeTo(polished, "-0.7535813351441342316834305500884110381431");
	const double delta = static_cast<double>(polished - static_cast<__float128>(-0.7535813351441343));
	EXPECT_LT(std::fabs(delta), 2e-16);

	ASSERT_TRUE(polishZeta3Root(0, -2203, -555, 1, -0.2636600441662106, &polished));
	closeTo(polished, "-0.2636600441662105959027763464633130997879");
	const double delta2 = static_cast<double>(polished - static_cast<__float128>(-0.2636600441662106));
	EXPECT_LT(std::fabs(delta2), 2e-16);
}
