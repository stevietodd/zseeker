#include <gtest/gtest.h>
#include "../math.hpp"
#include "../GpuPolynomialChecker.hpp"
#include "../lookupTable.hpp"

TEST(GpuPolynomialCheckerTestSuite, VLoopResultsConfirmTest) {
	PolynomialCheckerInterface *checker = new GpuPolynomialChecker();
    std::vector<float*> *hits;
    std::vector<int> *loopRanges = new std::vector<int>{-1,6,-1,6,-1,6,-1,6,-1,1446,-1,-1};

    //hits = checker->findHits(ZETA5, M_PI, 5, LUT.data(), loopRanges);

    // manual analysis indicates there should be at least 19 hits within .000005 range...
	ASSERT_LE(19, hits->size());

    // ...but there should be closer to 26-28 depending on rounding. If there are too many hits something is wrong
    EXPECT_GE(30, hits->size());

    // now let's ensure some close hits (err <= .000003) were returned
    //coeff = 191426;
    //EXPECT_EQ(0, hits->at(27)); TODO Check some actual results
}