#include <gtest/gtest.h>
#include "../lookupTable.hpp"

TEST(LookupTableTestSuite, LookupTableGeneralTest) {
	// assert typing and size
	ASSERT_EQ(typeid(LUT), typeid(std::array<float, 1'216'773>));
	ASSERT_EQ(LUT.size(), 1'216'773); // redundant but might as well
	
	// assert cutoff values are accurate
	ASSERT_EQ(292, LUT[0]);
	ASSERT_EQ(1'116, LUT[1]);
	ASSERT_EQ(4'412, LUT[2]);
	ASSERT_EQ(12'180, LUT[3]);
	ASSERT_EQ(304'468, LUT[4]);
	ASSERT_EQ(1'216'772, LUT[5]);
}

TEST(LookupTableTestSuite, LookupTableRandomValuesTest) {
	// assert the most basic values are where we expect
	ASSERT_EQ(0, LUT[6]);
	ASSERT_EQ(1, LUT[7]);
	ASSERT_EQ(-1, LUT[8]);

	// assert values for [1,2] are correct
	ASSERT_EQ(0.5, LUT[9]);
	ASSERT_EQ(-0.5, LUT[10]);
	ASSERT_EQ(2, LUT[11]);
	ASSERT_EQ(-2, LUT[12]);

	// assert values for [1,3] are correct
	ASSERT_EQ(0.33333333, LUT[13]);
	ASSERT_EQ(-0.33333333, LUT[14]);
	ASSERT_EQ(3, LUT[15]);
	ASSERT_EQ(-3, LUT[16]);
	ASSERT_EQ(0.66666666, LUT[17]);
	ASSERT_EQ(-0.66666666, LUT[18]);
	ASSERT_EQ(1.5, LUT[19]);
	ASSERT_EQ(-1.5, LUT[20]);

	// assert values for [1,4] are correct (should have omitted values like 2/4 that already exist)
	ASSERT_EQ(0.25, LUT[21]);
	ASSERT_EQ(-0.25, LUT[22]);
	ASSERT_EQ(4, LUT[23]);
	ASSERT_EQ(-4, LUT[24]);
	ASSERT_EQ(0.75, LUT[25]);
	ASSERT_EQ(-0.75, LUT[26]);
	ASSERT_EQ(1.3333333, LUT[27]);
	ASSERT_EQ(-1.3333333, LUT[28]);

	// assert some totally random values are correct (looked these up manually)
	ASSERT_EQ(0, LUT[1'000]);
	ASSERT_EQ(0, LUT[10'000]);
	ASSERT_EQ(0, LUT[100'000]);
	ASSERT_EQ(0, LUT[1'000'000]);
	ASSERT_EQ(0, LUT[1'216'772]);
}