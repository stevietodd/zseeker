#include <gtest/gtest.h>
#include "../lookupTable.hpp"

TEST(LookupTableTestSuite, LookupTableGeneralTest) {
	// assert typing and size
	ASSERT_EQ(typeid(LUT), typeid(std::array<float, 1'216'773>));
	ASSERT_EQ(LUT.size(), 1'216'773); // redundant but might as well
	
	// assert cutoff values are accurate
	EXPECT_EQ(292, LUT[0]);
	EXPECT_EQ(1'116, LUT[1]);
	EXPECT_EQ(4'412, LUT[2]);
	EXPECT_EQ(12'180, LUT[3]);
	EXPECT_EQ(304'468, LUT[4]);
	EXPECT_EQ(1'216'772, LUT[5]);
}

TEST(LookupTableTestSuite, LookupTableRandomValuesTest) {
	// assert the most basic values are where we expect
	EXPECT_FLOAT_EQ(0, LUT[6]);
	EXPECT_FLOAT_EQ(1, LUT[7]);
	EXPECT_FLOAT_EQ(-1, LUT[8]);

	// assert values for [1,2] are correct
	EXPECT_FLOAT_EQ(0.5, LUT[9]);
	EXPECT_FLOAT_EQ(-0.5, LUT[10]);
	EXPECT_FLOAT_EQ(2, LUT[11]);
	EXPECT_FLOAT_EQ(-2, LUT[12]);

	// assert values for [1,3] are correct
	EXPECT_FLOAT_EQ(0.33333333, LUT[13]);
	EXPECT_FLOAT_EQ(-0.33333333, LUT[14]);
	EXPECT_FLOAT_EQ(3, LUT[15]);
	EXPECT_FLOAT_EQ(-3, LUT[16]);
	EXPECT_FLOAT_EQ(0.66666666, LUT[17]);
	EXPECT_FLOAT_EQ(-0.66666666, LUT[18]);
	EXPECT_FLOAT_EQ(1.5, LUT[19]);
	EXPECT_FLOAT_EQ(-1.5, LUT[20]);

	// assert values for [1,4] are correct (should have omitted values like 2/4 that already exist)
	EXPECT_FLOAT_EQ(0.25, LUT[21]);
	EXPECT_FLOAT_EQ(-0.25, LUT[22]);
	EXPECT_FLOAT_EQ(4, LUT[23]);
	EXPECT_FLOAT_EQ(-4, LUT[24]);
	EXPECT_FLOAT_EQ(0.75, LUT[25]);
	EXPECT_FLOAT_EQ(-0.75, LUT[26]);
	EXPECT_FLOAT_EQ(1.3333333, LUT[27]);
	EXPECT_FLOAT_EQ(-1.3333333, LUT[28]);

	// assert a few pairs are positive/negative (this property isn't important but in the current design
	// should always be true)
	EXPECT_FLOAT_EQ(LUT[99], -LUT[100]);
	EXPECT_FLOAT_EQ(LUT[221], -LUT[222]);

	// assert some totally random values are correct (looked these up manually)
	EXPECT_FLOAT_EQ(-4.1428571, LUT[1'000]);
	EXPECT_FLOAT_EQ(0.26373628, LUT[10'001]);
	EXPECT_FLOAT_EQ(-2.9895833, LUT[100'000]);
	EXPECT_FLOAT_EQ(0.51819181, LUT[1'000'001]);
	EXPECT_FLOAT_EQ(-1.001001, LUT[1'216'772]);
}