#ifndef POLYNOMIAL_CHECKER_INTERFACE_HPP
#define POLYNOMIAL_CHECKER_INTERFACE_HPP

#include <stdio.h>
#include <vector>

class PolynomialCheckerInterface
{
    public:
        PolynomialCheckerInterface(){}
        virtual ~PolynomialCheckerInterface(){}

        // solve for needle = coeff0 + coeff1*theConst + coeff2*theConst^2 + ... (a polynomial with the given degree)
        // note that the loopRanges define indexes of coeffArray to search through
        virtual std::vector<int*>* findHits(
            const float needle,
            const float theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges
        ) = 0;

    
    protected:
        void printHit(const float* coeffArray, const int i5, const int i4, const int i3, const int i2, const int i1, const int i0)
        {
            printf("%10.10lf*c^5 + %10.10lf*c^4 + %10.10lf*c^3 + %10.10lf*c^2 + %10.10lf*c + %10.10lf # HIT! coeffs=(%d,%d,%d,%d,%d,%d)\n",
				coeffArray[i5], coeffArray[i4], coeffArray[i3], coeffArray[i2], coeffArray[i1], coeffArray[i0],
                i5, i4, i3, i2, i1, i0);
        }
};

#endif // POLYNOMIAL_CHECKER_INTERFACE_HPP