#ifndef POLYNOMIAL_CHECKER_INTERFACE_HPP
#define POLYNOMIAL_CHECKER_INTERFACE_HPP

#include <vector>

class PolynomialCheckerInterface
{
    public:
        PolynomialCheckerInterface(){}
        virtual ~PolynomialCheckerInterface(){}
        virtual std::vector<float>* findHits() = 0;
};

#endif // POLYNOMIAL_CHECKER_INTERFACE_HPP