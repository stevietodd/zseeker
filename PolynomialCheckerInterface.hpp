#ifndef POLYNOMIAL_CHECKER_INTERFACE_HPP
#define POLYNOMIAL_CHECKER_INTERFACE_HPP

#include <vector>

class PolynomialCheckerInterface
{
    public:
        PolynomialCheckerInterface(){}
        virtual ~PolynomialCheckerInterface(){}

        // solve for needle = coeff0 + coeff1*theConst + coeff2*theConst^2 + ... (a polynomial with the given degree)
        // note that the loopRanges define indexes of coeffArray to search through
        virtual std::vector<float*>* findHits(
            const float needle,
            const float theConst,
            const int degree,
            const std::vector<float> *coeffArray,
            const std::vector<int> *loopRanges
        ) = 0;
};

#endif // POLYNOMIAL_CHECKER_INTERFACE_HPP