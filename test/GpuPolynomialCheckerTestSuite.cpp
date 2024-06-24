#include <gtest/gtest.h>
#include "../math.hpp"
#include "../GpuPolynomialChecker.hpp"
#include "../lookupTable.hpp"

TEST(GpuPolynomialCheckerTestSuite, VLoopResultsConfirmTest) {
	PolynomialCheckerInterface *checker = new GpuPolynomialChecker();
    std::vector<float*> *hits;
    std::vector<int> *loopRanges = new std::vector<int>{-1,6,-1,6,-1,6,-1,6,-1,1446,-1,-1};

    hits = checker->findHits(ZETA5, M_PI, 5, LUT.data(), loopRanges);

	ASSERT_EQ(28, hits->size());
    //EXPECT_EQ(0, hits->at(27)); TODO Check some actual results
}