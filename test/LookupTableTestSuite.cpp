#include <gtest/gtest.h>
#include "../lookupTable.hpp"

TEST(LookupTableTestSuite, LookupTableGeneralTest) {
	// assert typing and size
	ASSERT_EQ(typeid(LUT), typeid(std::array<float, 608'384>));
	ASSERT_EQ(LUT.size(), 608'384); // redundant but might as well
}

TEST(LookupTableTestSuite, LookupTableRandomValuesTest) {
	// assert the most basic values are where we expect
	EXPECT_FLOAT_EQ(0, LUT[0]);
	EXPECT_FLOAT_EQ(1, LUT[1]);

	// assert values for [1,2] are correct
	EXPECT_FLOAT_EQ(0.5, LUT[2]);
	EXPECT_FLOAT_EQ(2, LUT[3]);

	// assert values for [1,3] are correct
	EXPECT_FLOAT_EQ(0.33333333, LUT[4]);
	EXPECT_FLOAT_EQ(3, LUT[5]);
	EXPECT_FLOAT_EQ(0.66666666, LUT[6]);
	EXPECT_FLOAT_EQ(1.5, LUT[7]);

	// assert values for [1,4] are correct (should have omitted values like 2/4 that already exist)
	EXPECT_FLOAT_EQ(0.25, LUT[8]);
	EXPECT_FLOAT_EQ(4, LUT[9]);
	EXPECT_FLOAT_EQ(0.75, LUT[10]);
	EXPECT_FLOAT_EQ(1.33333333, LUT[11]);

	// assert some totally random values are correct (looked these up manually)
	EXPECT_FLOAT_EQ(0.26829268, LUT[1'000]);
	EXPECT_FLOAT_EQ(0.6640625, LUT[10'000]);
	EXPECT_FLOAT_EQ(0.09113300, LUT[100'000]);
	EXPECT_FLOAT_EQ(1.00100100, LUT[608'383]);
}