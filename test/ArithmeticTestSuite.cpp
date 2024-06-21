#include <gtest/gtest.h>
#include "../math.hpp"

TEST(ArithmeticTestSuite, DoubleBasicallyEqualMacroTest) {
	// use pi for our calculations
	const double pi = M_PI;
	
	// assert our DOUBLE_BASICALLY_EQUAL macro works (and also pi constant is accurate)
	EXPECT_FALSE(DOUBLE_BASICALLY_EQUAL(pi, 3.1415926)); // not precise enough
	EXPECT_FALSE(DOUBLE_BASICALLY_EQUAL(pi, 3.14159264)); // more precise but not close enough
	EXPECT_TRUE(DOUBLE_BASICALLY_EQUAL(pi, 3.14159265)); // more precise and close enough
	EXPECT_TRUE(DOUBLE_BASICALLY_EQUAL(pi, 3.141592653)); // even more precise (and accurate)
	EXPECT_TRUE(DOUBLE_BASICALLY_EQUAL(pi, 3.141592659)); // even more precise and technically wrong but within tolerance
}

TEST(ArithmeticTestSuite, DoubleOperationsTest) {
	// use pi for our calculations
	const double pi = M_PI;
	const double pi2 = pi * pi;
	const double pi4 = pi * pi * pi * pi;
	
	// assert our manually-crafted values match what pow returns
	const double pi2pow = pow(pi, 2);
	const double pi4pow = pow(pi, 4);
	EXPECT_TRUE(DOUBLE_BASICALLY_EQUAL(pi2, pi2pow));
	EXPECT_TRUE(DOUBLE_BASICALLY_EQUAL(pi4, pi4pow));

	// assert we can calculate known zeta values (and our zeta constants are accurate...enough)
	EXPECT_TRUE(DOUBLE_BASICALLY_EQUAL((pi2 / 6), ZETA2));
	EXPECT_TRUE(DOUBLE_BASICALLY_EQUAL((pi4 / 90), ZETA4));

	// assert that a bunch of random operations are accurate enough (double-checked these with various high-precision calculators)
	const double random1 = (double)1000 * pi4 +
		((double)222 / (double)17) * pi2 * pi +
		((double)23 / (double)7) * pi2 +
		((double)11 / (double)8) * pi +
		((double)41 / (double)12);
	const double random2 = ((double)-973) * pi4 +
		((double)432 / (double)19) * pi2 * pi -
		((double)729 / (double)47) * pi2 -
		((double)888 / (double)5) * pi +
		((double)913 / (double)11);
	EXPECT_TRUE(DOUBLE_BASICALLY_EQUAL(random1, 97854.161586214948));
	EXPECT_TRUE(DOUBLE_BASICALLY_EQUAL(random2, -94702.091478218225));
}	