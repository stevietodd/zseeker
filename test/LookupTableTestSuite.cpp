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
	ASSERT_EQ(0.33, LUT[13]);
	ASSERT_EQ(-0.33, LUT[14]);
	ASSERT_EQ(3, LUT[15]);
	ASSERT_EQ(-3, LUT[16]);
	ASSERT_EQ(0.66, LUT[17]);
	ASSERT_EQ(-0.66, LUT[18]);
	ASSERT_EQ(1.5, LUT[19]);
	ASSERT_EQ(-1.5, LUT[20]);

	// assert some random values are as expected
}