// cudawrapper.hpp
#include <vector>

bool InitCUDA(bool b);

// will calculate ax^5 + bx^4 + cx^3 + dx^2 + ex + f - zeta(5) and return x if
// the value is within a "close enough" bound to zero
// cons = x
// cubicSum = cx^3 + dx^2 + ex + f (pre-calculated)
// coeffArray = array of all possible a/b/c/d/e/f coeff values
// quartLastIndex = last index to loop through for b values
// quintLastIndex = last index to loop through for a values
template<typename T>
std::vector<T>* testForZeta5OnGPU(T cons, T cubicSum, const T *coeffArray, int quartLastIndex, int quintLastIndex);