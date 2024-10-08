#include <gtest/gtest.h>
#include <limits>
using namespace std;

TEST(PrecisionTestSuite, IntTest) {
	typedef numeric_limits<int> limint;

   EXPECT_EQ(sizeof(int), 4) << "Size in bytes was unexpected: int";
   EXPECT_EQ(limint::digits, 31) << "Digits limit was unexpected: int";
   EXPECT_EQ(limint::digits10, 9) << "Digits10 limit was unexpected: int";
   EXPECT_EQ(limint::max_digits10, 0) << "MaxDigits10 limit was unexpected: int";
}

TEST(PrecisionTestSuite, FloatTest) {
	typedef numeric_limits<float> limfloat;

   EXPECT_EQ(sizeof(float), 4) << "Size in bytes was unexpected: float";
   EXPECT_EQ(limfloat::digits, 24) << "Digits limit was unexpected: float";
   EXPECT_EQ(limfloat::digits10, 6) << "Digits10 limit was unexpected: float";
   EXPECT_EQ(limfloat::max_digits10, 9) << "MaxDigits10 limit was unexpected: float";
}

TEST(PrecisionTestSuite, DoubleTest) {
	typedef numeric_limits<double> limdouble;

   EXPECT_EQ(sizeof(double), 8) << "Size in bytes was unexpected: double";
   EXPECT_EQ(limdouble::digits, 53) << "Digits limit was unexpected: double";
   EXPECT_EQ(limdouble::digits10, 15) << "Digits10 limit was unexpected: double";
   EXPECT_EQ(limdouble::max_digits10, 17) << "MaxDigits10 limit was unexpected: double";
}