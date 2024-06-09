#ifndef CPU_POLYNOMIAL_CHECKER_HPP
#define CPU_POLYNOMIAL_CHECKER_HPP

#include "PolynomialCheckerInterface.hpp"

class CpuPolynomialChecker : public PolynomialCheckerInterface
{
    public:
        CpuPolynomialChecker(){}
        ~CpuPolynomialChecker(){}
        std::vector<float>* findHits();
};

#endif // CPU_POLYNOMIAL_CHECKER_HPP