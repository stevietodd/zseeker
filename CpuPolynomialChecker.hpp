#ifndef CPU_POLYNOMIAL_CHECKER_HPP
#define CPU_POLYNOMIAL_CHECKER_HPP

#include "PolynomialCheckerInterface.hpp"

class CpuPolynomialChecker : public PolynomialCheckerInterface
{
    public:
        CpuPolynomialChecker(){}
        ~CpuPolynomialChecker(){}
        
        virtual std::vector<float*>* findHits(
            const float needle,
            const float theConst,
            const int degree,
            const std::vector<float> *coeffArray,
            const std::vector<int> *loopRanges
        );
};

#endif // CPU_POLYNOMIAL_CHECKER_HPP