#ifndef CPU_POLYNOMIAL_CHECKER_HPP
#define CPU_POLYNOMIAL_CHECKER_HPP

#include "PolynomialCheckerInterface.hpp"

class CpuPolynomialChecker : public PolynomialCheckerInterface
{
    public:
        CpuPolynomialChecker(){}
        ~CpuPolynomialChecker(){}

        std::vector<int*>* findHits(
            const float needle,
            const float theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges
        );
};

#endif // CPU_POLYNOMIAL_CHECKER_HPP