#ifndef POLYNOMIAL_CHECKER_INTERFACE_HPP
#define POLYNOMIAL_CHECKER_INTERFACE_HPP

#include <vector>

class PolynomialCheckerInterface
{
    public:
        PolynomialCheckerInterface(){}
        virtual ~PolynomialCheckerInterface(){}
        virtual std::vector<float>* findHits(const float theConst, const float needle, std::vector<float> *coeffArray) = 0;
};

#endif // POLYNOMIAL_CHECKER_INTERFACE_HPP