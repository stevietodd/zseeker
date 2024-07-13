#include <gtest/gtest.h>
#include "../math.hpp"
#include "../CpuPolynomialChecker.hpp"

TEST(CpuPolynomialCheckerTestSuite, VLoopResultsConfirmTest) {
	PolynomialCheckerInterface *checker = new CpuPolynomialChecker();
    std::vector<float*> *hits;
    std::vector<int> *loopRanges = new std::vector<int>{-1,6,-1,6,-1,6,-1,6,-1,1446,-1,-1};

    hits = checker->findHits(ZETA5, M_PI, 5, NULL, loopRanges);

	ASSERT_EQ(28, hits->size());
    //EXPECT_EQ(0, hits->at(27)); TODO Check some actual results
}

TEST(CpuPolynomialCheckerTestSuite, Zeta4WithPiTest) {
    PolynomialCheckerInterface *checker = new CpuPolynomialChecker();
    std::vector<float*> *hits;
    std::vector<int> *loopRanges = new std::vector<int>{-1,6,-1,6,-1,6,-1,6,-1,-1,-1,6};

    hits = checker->findHits(ZETA4, M_PI, 5, NULL, loopRanges);

// this does get the right hit when v-loop is on v=9829 which corresponds to 1/90. v5 ends up being 1.08232343 while z4 = 1.082323223
	ASSERT_EQ(28, hits->size());
    //EXPECT_EQ(0, hits->at(27)); TODO Check some actual results
}