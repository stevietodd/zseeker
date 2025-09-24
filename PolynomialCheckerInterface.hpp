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
            const double needle,
            const double theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges,
            long& floatHitCount
        ) = 0;

    
    protected:
        void printHit(const float* coeffArray, const int i5, const int i4, const int i3, const int i2, const int i1, const int i0)
        {
            printf("%10.10lf*c^5 + %10.10lf*c^4 + %10.10lf*c^3 + %10.10lf*c^2 + %10.10lf*c + %10.10lf # HIT! coeffs=(%d,%d,%d,%d,%d,%d)\n",
				(i5 < 0) ? -coeffArray[-i5] : coeffArray[i5],
				(i4 < 0) ? -coeffArray[-i4] : coeffArray[i4],
				(i3 < 0) ? -coeffArray[-i3] : coeffArray[i3],
				(i2 < 0) ? -coeffArray[-i2] : coeffArray[i2],
				(i1 < 0) ? -coeffArray[-i1] : coeffArray[i1],
				(i0 < 0) ? -coeffArray[-i0] : coeffArray[i0],
                i5, i4, i3, i2, i1, i0);
        }
};

#endif // POLYNOMIAL_CHECKER_INTERFACE_HPP