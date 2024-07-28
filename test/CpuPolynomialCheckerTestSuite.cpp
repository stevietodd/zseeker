#include <gtest/gtest.h>
#include "../math.hpp"
#include "../CpuPolynomialChecker.hpp"

TEST(CpuPolynomialCheckerTestSuite, VLoopResultsConfirmTest) {
	PolynomialCheckerInterface *checker = new CpuPolynomialChecker();
    std::vector<int*> *hits;
    std::vector<int> *loopRanges = new std::vector<int>{-1,6,-1,6,-1,6,-1,6,-1,1446,-1,-1};

    hits = checker->findHits(ZETA5, M_PI, 5, NULL, loopRanges);

	// manual analysis indicates there should be at least 19 hits within .000005 range...
	ASSERT_LE(19, hits->size());

    // ...but there should be closer to 26-28 depending on rounding. If there are too many hits something is wrong
    EXPECT_GE(30, hits->size());

    // now let's ensure some close hits (err <= .000003) were returned
    bool hit1Found = false, hit2Found = false, hit3Found = false;
    int hit1[] = {22,22,22,22,22,22};
    int hit2[] = {2,2,2,2,2,2};
    int hit3[] = {222,222,222,222,222,222};
    for (int* hit : *hits) {
        if (!hit1Found && 0 == std::memcmp(hit, hit1, sizeof(hit1))) {
            hit1Found = true;
            continue;
        }
        if (!hit2Found && 0 == std::memcmp(hit, hit2, sizeof(hit2))) {
            hit2Found = true;
            continue;
        }
        if (!hit3Found && 0 == std::memcmp(hit, hit3, sizeof(hit3))) {
            hit3Found = true;
            continue;
        }
    }

    if (!hit1 || !hit2 || !hit3) {
        // didn't find all 3 hits, log failure, actually split these out TODO
        FAIL();
    }
    //coeff = 191426;
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