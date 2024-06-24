#ifndef MATH_HPP
#define MATH_HPP

#include <cmath>

#define DOUBLE_POS_ERROR .00000001 // approximate precision in digits = EIGHT
#define DOUBLE_NEG_ERROR -DOUBLE_POS_ERROR
#define DOUBLE_BASICALLY_EQUAL(d1,d2) ((d1 - d2) >= DOUBLE_NEG_ERROR && (d1 - d2) <= DOUBLE_POS_ERROR)

// Possible TODO: Original checkz3constantswithz5usingLUTandCPU.cpp used .0000001, not .00001
#define FLOAT_POS_ERROR .00001 // approximate precision in digits = FIVE
#define FLOAT_NEG_ERROR -FLOAT_POS_ERROR
#define FLOAT_BASICALLY_EQUAL(f1,f2) ((f1 - f2) >= FLOAT_NEG_ERROR && (f1 - f2) <= FLOAT_POS_ERROR)

#define ZETA2 1.64493406684822643647
#define ZETA4 1.08232323371113819152
#define ZETA5 1.03692775514336992633 // 1.036927755143369926331365486457034168L

#endif // MATH_HPP