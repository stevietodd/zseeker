#ifndef MATH_HPP
#define MATH_HPP

#define DOUBLE_POS_ERROR .00000001 // approximate precision in digits = EIGHT
#define DOUBLE_NEG_ERROR -DOUBLE_POS_ERROR
#define DOUBLE_BASICALLY_EQUAL(d1,d2) ((d1 - d2) >= DOUBLE_NEG_ERROR && (d1 - d2) <= DOUBLE_POS_ERROR)
              
#define ZETA2 1.64493406684822643647
#define ZETA4 1.08232323371113819152

#endif // MATH_HPP