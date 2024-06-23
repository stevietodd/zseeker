#include <gtest/gtest.h>
#include "../math.hpp"
#include "../CpuPolynomialChecker.hpp"

TEST(CpuPolynomialCheckerTestSuite, VLoopResultsConfirmTest) {
	PolynomialCheckerInterface *checker = new CpuPolynomialChecker();
    std::vector<float*> *hits;
    std::vector<int> *coeffArray = new std::vector<int>{-1,6,-1,6,-1,6,-1,6,-1,1446,-1,-1};

    hits = checker->findHits(ZETA5, M_PI, 5, NULL, coeffArray);

	ASSERT_EQ(28, hits->size());
    //EXPECT_EQ(0, hits->at(27)); TODO Check some actual results
}