#include <gtest/gtest.h>
#include <limits>
#include "../math.hpp"
using namespace std;

TEST(PrecisionTestSuite, IntSizeTest) {
	typedef numeric_limits<int> limint;

   EXPECT_EQ(4, sizeof(int)) << "Size in bytes was unexpected: int";
   EXPECT_EQ(31, limint::digits) << "Digits limit was unexpected: int";
   EXPECT_EQ(9, limint::digits10) << "Digits10 limit was unexpected: int";
   EXPECT_EQ(0, limint::max_digits10) << "MaxDigits10 limit was unexpected: int";
}

TEST(PrecisionTestSuite, FloatSizeTest) {
	typedef numeric_limits<float> limfloat;

   EXPECT_EQ(4, sizeof(float)) << "Size in bytes was unexpected: float";
   EXPECT_EQ(24, limfloat::digits) << "Digits limit was unexpected: float";
   EXPECT_EQ(6, limfloat::digits10) << "Digits10 limit was unexpected: float";
   EXPECT_EQ(9, limfloat::max_digits10) << "MaxDigits10 limit was unexpected: float";
}

TEST(PrecisionTestSuite, DoubleSizeTest) {
	typedef numeric_limits<double> limdouble;

   EXPECT_EQ(8, sizeof(double)) << "Size in bytes was unexpected: double";
   EXPECT_EQ(53, limdouble::digits) << "Digits limit was unexpected: double";
   EXPECT_EQ(15, limdouble::digits10) << "Digits10 limit was unexpected: double";
   EXPECT_EQ(17, limdouble::max_digits10) << "MaxDigits10 limit was unexpected: double";
}

TEST(PrecisionTestSuite, AllowedFloatPrecisionBasedOnMaxValueTest) {
   EXPECT_FLOAT_EQ(0.000000238418579, getFloatPrecisionBasedOnMaxValue(1));
   EXPECT_FLOAT_EQ(0.000000238418579, getFloatPrecisionBasedOnMaxValue(0));
   EXPECT_FLOAT_EQ(0.000000238418579, getFloatPrecisionBasedOnMaxValue(-1));

   EXPECT_FLOAT_EQ(0.000000238418579, getFloatPrecisionBasedOnMaxValue(1.99));
   EXPECT_FLOAT_EQ(0.000000476837158, getFloatPrecisionBasedOnMaxValue(2));
   EXPECT_FLOAT_EQ(0.000000476837158, getFloatPrecisionBasedOnMaxValue(2.22222));

   EXPECT_FLOAT_EQ(0.03125, getFloatPrecisionBasedOnMaxValue(222'222));
   EXPECT_FLOAT_EQ(1, getFloatPrecisionBasedOnMaxValue(8'388'607));
   EXPECT_FLOAT_EQ(1, getFloatPrecisionBasedOnMaxValue(-8'388'607));
   EXPECT_FLOAT_EQ(2, getFloatPrecisionBasedOnMaxValue(8'388'608));
}

TEST(PrecisionTestSuite, AllowedDoublePrecisionBasedOnMaxValueTest) {
   EXPECT_DOUBLE_EQ(0.00000000000000044408920985006262, getDoublePrecisionBasedOnMaxValue(1));
   EXPECT_DOUBLE_EQ(0.00000000000000044408920985006262, getDoublePrecisionBasedOnMaxValue(0));
   EXPECT_DOUBLE_EQ(0.00000000000000044408920985006262, getDoublePrecisionBasedOnMaxValue(-1));

   EXPECT_DOUBLE_EQ(0.00000000000000044408920985006262, getDoublePrecisionBasedOnMaxValue(1.99));
   EXPECT_DOUBLE_EQ(0.00000000000000088817841970012523, getDoublePrecisionBasedOnMaxValue(2));
   EXPECT_DOUBLE_EQ(0.00000000000000088817841970012523, getDoublePrecisionBasedOnMaxValue(2.22222));

   EXPECT_DOUBLE_EQ(0.000000000058207660913467407, getDoublePrecisionBasedOnMaxValue(222'222));
   EXPECT_DOUBLE_EQ(1, getDoublePrecisionBasedOnMaxValue(4'503'599'627'370'468)); // note (2^52 - 1) didn't work here
   EXPECT_DOUBLE_EQ(1, getDoublePrecisionBasedOnMaxValue(-4'503'599'627'370'468));
   EXPECT_DOUBLE_EQ(2, getDoublePrecisionBasedOnMaxValue(4'503'599'627'370'496));
}