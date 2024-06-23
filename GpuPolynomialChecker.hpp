#ifndef GPU_POLYNOMIAL_CHECKER_HPP
#define GPU_POLYNOMIAL_CHECKER_HPP

#include "PolynomialCheckerInterface.hpp"

class GpuPolynomialChecker : public PolynomialCheckerInterface
{
    public:
        GpuPolynomialChecker(){}
        ~GpuPolynomialChecker(){}
        
        std::vector<float*>* findHits(
            const float needle,
            const float theConst,
            const int degree,
            const std::vector<float> *coeffArray,
            const std::vector<int> *loopRanges
        );
};

#endif // GPU_POLYNOMIAL_CHECKER_HPP