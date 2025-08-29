#include <gtest/gtest.h>
#include "../math.hpp"
#include "../CpuPolynomialChecker.hpp"
#include <cstring> // for memcmp

/*
    This is a very weird setup of looping through all quintics and some quartics but
    using all zero's for all cubic and below coefficients. It was originally designed this way
    because my first pass at GPU programming (which became GpuQuinticLastChecker) was most
    easily compared to these loop ranges.
 */
TEST(CpuPolynomialCheckerTestSuite, QuinticLastOnlyQuinticQuarticResultsConfirmTest) {
	PolynomialCheckerInterface *checker = new CpuQuinticLastChecker();
    std::vector<int*> *hits;
    std::vector<int> *loopRanges = new std::vector<int>{
		USE_DEFAULT, USE_DEFAULT,
		-773, 773,
		0, 0, // skip this loop effectively
		0, 0, // skip this loop effectively
		0, 0, // skip this loop effectively
		0, 0 // skip this loop effectively
	};

    hits = checker->findHits(ZETA5, M_PI, 5, NULL, loopRanges);

	// manual analysis indicates there should be at least 19 hits within .000005 range...
	ASSERT_LE(19, hits->size());

    // ...but there should be closer to 26-28 depending on rounding. If there are too many hits something is wrong
    EXPECT_GE(30, hits->size());

    // now let's ensure some close hits (err <= .000003) were returned
    bool hit1Found = false, hit2Found = false, hit3Found = false;
    int hit1[] = {-95'710,113,0,0,0,0}; // -0.372795969c^5 + 1.18181813c^4 = -(148/397)c^5 + (13/11)c^4
    int hit2[] = {-472'234,151,0,0,0,0}; // -0.724177063c^5 + 2.28571439c^4 = -(638/881)c^5 + (16/17)c^4
    int hit3[] = {-298'042,720,0,0,0,0}; // -0.00570613425c^5 + 0.0285714287c^4 = -(4/701)c^5 + (1/35)c^4
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
}

/*
 * Takes about 40 seconds on linux box, 723 results
 */
TEST(CpuPolynomialCheckerTestSuite, QuinticLastZeroAndOneHighDegreesResultsConfirmTest) {
	PolynomialCheckerInterface *checker = new CpuQuinticLastChecker();
    std::vector<int*> *hits;
    std::vector<int> *loopRanges = new std::vector<int>{
		0, 1,
		0, 1,
		0, 1,
		USE_DEFAULT, USE_DEFAULT,
		USE_DEFAULT, USE_DEFAULT,
		USE_DEFAULT, USE_DEFAULT
	};

    hits = checker->findHits(ZETA5, M_PI, 5, NULL, loopRanges);

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

TEST(CpuPolynomialCheckerTestSuite, Zeta4WithPiTest) {
    GTEST_SKIP() << "Probably won't unskip this until qd work is added.";
    PolynomialCheckerInterface *checker = new CpuQuinticLastChecker();
    std::vector<int*> *hits;
    std::vector<int> *loopRanges = new std::vector<int>{-1,6,-1,6,-1,6,-1,6,-1,-1,-1,6};

    hits = checker->findHits(ZETA4, M_PI, 5, NULL, loopRanges);

// this does get the right hit when v-loop is on v=9829 which corresponds to 1/90. v5 ends up being 1.08232343 while z4 = 1.082323223
	ASSERT_EQ(28, hits->size());
    //EXPECT_EQ(0, hits->at(27)); TODO Check some actual results
}

TEST(CpuPolynomialCheckerTestSuite, QuinticFirstOnlyQuinticQuarticResultsConfirmTest) {
	PolynomialCheckerInterface *checker = new CpuQuinticFirstChecker();
    std::vector<int*> *hits;
    std::vector<int> *loopRanges = new std::vector<int>{
		USE_DEFAULT, USE_DEFAULT,
		-773, 773,
		0, 0, // skip this loop effectively
		0, 0, // skip this loop effectively
		0, 0, // skip this loop effectively
		0, 0 // skip this loop effectively
	};

    hits = checker->findHits(ZETA5, M_PI, 5, NULL, loopRanges);

	// manual analysis indicates there should be at least 19 hits within .000005 range...
	ASSERT_LE(19, hits->size());

    // ...but there should be closer to 26-28 depending on rounding. If there are too many hits something is wrong
    EXPECT_GE(30, hits->size());

    // now let's ensure some close hits (err <= .000003) were returned
    bool hit1Found = false, hit2Found = false, hit3Found = false;
    int hit1[] = {-95'710,113,0,0,0,0}; // -0.372795969c^5 + 1.18181813c^4 = -(148/397)c^5 + (13/11)c^4
    int hit2[] = {-472'234,151,0,0,0,0}; // -0.724177063c^5 + 2.28571439c^4 = -(638/881)c^5 + (16/17)c^4
    int hit3[] = {-298'042,720,0,0,0,0}; // -0.00570613425c^5 + 0.0285714287c^4 = -(4/701)c^5 + (1/35)c^4
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
}

TEST(CpuPolynomialCheckerTestSuite, QuinticFirstWithBreakoutsOnlyQuinticQuarticResultsConfirmTest) {
	PolynomialCheckerInterface *checker = new CpuQuinticFirstWithBreakoutsChecker();
    std::vector<int*> *hits;
    std::vector<int> *loopRanges = new std::vector<int>{USE_DEFAULT, USE_DEFAULT,
		-773, 773,
		0, 0, // skip this loop effectively
		0, 0, // skip this loop effectively
		0, 0, // skip this loop effectively
		0, 0 // skip this loop effectively
	};

    hits = checker->findHits(ZETA5, M_PI, 5, NULL, loopRanges);

	// manual analysis indicates there should be at least 19 hits within .000005 range...
	ASSERT_LE(19, hits->size());

    // ...but there should be closer to 26-28 depending on rounding. If there are too many hits something is wrong
    EXPECT_GE(30, hits->size());

    // now let's ensure some close hits (err <= .000003) were returned
    bool hit1Found = false, hit2Found = false, hit3Found = false;
    int hit1[] = {-95'710,113,0,0,0,0}; // -0.372795969c^5 + 1.18181813c^4 = -(148/397)c^5 + (13/11)c^4
    int hit2[] = {-472'234,151,0,0,0,0}; // -0.724177063c^5 + 2.28571439c^4 = -(638/881)c^5 + (16/17)c^4
    int hit3[] = {-298'042,720,0,0,0,0}; // -0.00570613425c^5 + 0.0285714287c^4 = -(4/701)c^5 + (1/35)c^4
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
}

TEST(CpuPolynomialCheckerTestSuite, QuinticFirstWithBreakoutsSameAsNonBreakoutsTest) {
    PolynomialCheckerInterface *nonBreakoutsChecker = new CpuQuinticFirstChecker();
	PolynomialCheckerInterface *withBreakoutsChecker = new CpuQuinticFirstWithBreakoutsChecker();
    std::vector<int*> *nonBreakoutHits;
    std::vector<int*> *withBreakoutHits;

    // randomize these ranges in such a way that the non-breakouts checker only takes 1-2 minutes
    srand(time(0));
    int quinticRange = (rand() % 1'216'767) - 608'383; // = quintic max range because we are only doing 1 of these loops
    int quarticRange = (rand() % 304'462) - 152'231; // = quartic max range minus 1 because we are doing 2 of these loops
    int cubicRange = (rand() % 12'144) - 6'087; // = cubic max range minus 31 because we are doing 32 of these loops
    std::vector<int> *loopRanges = new std::vector<int>{
        quinticRange, quinticRange,
        quarticRange, (quarticRange + 1),
        cubicRange, (cubicRange + 31),
        USE_DEFAULT, USE_DEFAULT,
		USE_DEFAULT, USE_DEFAULT,
		USE_DEFAULT, USE_DEFAULT
    };

    nonBreakoutHits = nonBreakoutsChecker->findHits(ZETA5, M_PI, 5, NULL, loopRanges);
    withBreakoutHits = withBreakoutsChecker->findHits(ZETA5, M_PI, 5, NULL, loopRanges);

	// confirm using breakouts produces the exact same number of hits
	ASSERT_EQ(nonBreakoutHits->size(), withBreakoutHits->size());

    printf("Both CPU checkers had %lu hits\n", nonBreakoutHits->size());
}