#include <gtest/gtest.h>
#include "../math.hpp"
#include "../GpuPolynomialChecker.hpp"
#include "../lookupTable.hpp"
#include <cstring>

TEST(GpuPolynomialCheckerTestSuite, QuinticLastOnlyQuinticQuarticResultsConfirmTest) {
	PolynomialCheckerInterface *checker = new GpuQuinticLastChecker();
    std::vector<int*> *hits;
    std::vector<int> *loopRanges = new std::vector<int>{
		USE_DEFAULT, USE_DEFAULT,
		-1446, 1446,
		0, 0, // skip this loop effectively
		0, 0, // skip this loop effectively
		0, 0, // skip this loop effectively
		0, 0 // skip this loop effectively
	};

    hits = checker->findHits(ZETA5, M_PI, 5, LUT.data(), loopRanges);

    // manual analysis indicates there should be at least 19 hits within .000005 range...
	ASSERT_LE(19, hits->size());

    // ...but there should be closer to 26-28 depending on rounding. If there are too many hits something is wrong
    EXPECT_GE(30, hits->size());

/* 8/17/24 - Commenting this part out for now because it was going to be a hassle to get this original Gpu checker to
           - correctly stitch together all 6 coeffecients on a hit (since only the quartic and quintic were being processed
           - on the GPU) and I was heading toward the quintic-first solution anyway

    // now let's ensure some close hits (err <= .000003) were returned
    bool hit1Found = false, hit2Found = false, hit3Found = false;
    int hit1[] = {191426,231,6,6,6,6}; // -0.372795969c^5 + 1.18181813c^4
    int hit2[] = {944474,307,6,6,6,6}; // -0.724177063c^5 + 2.28571439c^4
    int hit3[] = {596090,1445,6,6,6,6}; // -0.00570613425c^5 + 0.0285714287c^4
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

    if (!hit1Found || !hit2Found || !hit3Found) {
        FAIL() << "Did not find all hits we were expecting. Found Hit1? Hit2?, Hit3? = " << hit1Found << ","
            << hit2Found << "," << hit3Found;
    }
*/
}

/*
 * Takes about 163 seconds on linux box, 727 results. Much slower than CPU so that's not swell
 */
TEST(GpuPolynomialCheckerTestSuite, QuinticFirstZeroAndOneHighDegreesResultsConfirmTest) {
	PolynomialCheckerInterface *checker = new GpuQuinticFirstChecker();
    std::vector<int*> *hits;
    std::vector<int> *loopRanges = new std::vector<int>{
		0, 1,
		0, 1,
		0, 1,
		USE_DEFAULT, USE_DEFAULT,
		USE_DEFAULT, USE_DEFAULT,
		USE_DEFAULT, USE_DEFAULT
	};

    hits = checker->findHits(ZETA5, M_PI, 5, LUT.data(), loopRanges);

    // various setups have netted 721 results...
	ASSERT_LE(720, hits->size());

    // ...but some have had as many as 727. If there are too many hits something is wrong
    EXPECT_GE(730, hits->size());

    // // now let's ensure some close hits (err <= .000003) were returned
    // bool hit1Found = false, hit2Found = false, hit3Found = false;
    // int hit1[] = {191426,231,6,6,6,6}; // -0.372795969c^5 + 1.18181813c^4
    // int hit2[] = {944474,307,6,6,6,6}; // -0.724177063c^5 + 2.28571439c^4
    // int hit3[] = {596090,1445,6,6,6,6}; // -0.00570613425c^5 + 0.0285714287c^4
    // for (int* hit : *hits) {
    //     if (!hit1Found && 0 == std::memcmp(hit, hit1, sizeof(hit1))) {
    //         hit1Found = true;
    //         continue;
    //     }
    //     if (!hit2Found && 0 == std::memcmp(hit, hit2, sizeof(hit2))) {
    //         hit2Found = true;
    //         continue;
    //     }
    //     if (!hit3Found && 0 == std::memcmp(hit, hit3, sizeof(hit3))) {
    //         hit3Found = true;
    //         continue;
    //     }
    // }

    // if (!hit1Found || !hit2Found || !hit3Found) {
    //     FAIL() << "Did not find all hits we were expecting. Found Hit1? Hit2?, Hit3? = " << hit1Found << ","
    //         << hit2Found << "," << hit3Found;
    // }
}

TEST(GpuPolynomialCheckerTestSuite, Zeta4WithPiTest) {
    GTEST_SKIP() << "Probably won't unskip this until qd work is added.";
    PolynomialCheckerInterface *checker = new GpuQuinticFirstChecker();
    std::vector<int*> *hits;
    std::vector<int> *loopRanges = new std::vector<int>{-1,6,-1,6,-1,6,-1,6,-1,-1,-1,6};

    hits = checker->findHits(ZETA4, M_PI, 5, LUT.data(), loopRanges);

// this does get the right hit when v-loop is on v=9829 which corresponds to 1/90. v5 ends up being 1.08232343 while z4 = 1.082323223
	ASSERT_EQ(28, hits->size());
    //EXPECT_EQ(0, hits->at(27)); TODO Check some actual results
}