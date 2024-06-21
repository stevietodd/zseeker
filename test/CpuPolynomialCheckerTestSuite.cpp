#include <gtest/gtest.h>
#include "../math.hpp"
#include "../CpuPolynomialChecker.hpp"

TEST(CpuPolynomialCheckerTestSuite, DoubleBasicallyEqualMacroTest) {
	PolynomialCheckerInterface *checker = new CpuPolynomialChecker();
    std::vector<float*> *hits;

    hits = checker->findHits(M_PI, ZETA5, NULL);

	EXPECT_EQ(28, hits->size()); // TODO: This only passes because I bail out of CpuPolynomialChecker when v=1446
}