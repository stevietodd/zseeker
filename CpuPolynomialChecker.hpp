#ifndef CPU_POLYNOMIAL_CHECKER_HPP
#define CPU_POLYNOMIAL_CHECKER_HPP

#include "PolynomialCheckerInterface.hpp"

class CpuQuinticLastChecker : public PolynomialCheckerInterface
{
    public:
        CpuQuinticLastChecker(){}
        ~CpuQuinticLastChecker(){}

        std::vector<int*>* findHits(
            const float needle,
            const float theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges
        );
};

class CpuQuinticFirstChecker : public PolynomialCheckerInterface
{
    public:
        CpuQuinticFirstChecker(){}
        ~CpuQuinticFirstChecker(){}

        std::vector<int*>* findHits(
            const float needle,
            const float theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges
        );
};

class CpuQuinticFirstWithBreakoutsChecker : public PolynomialCheckerInterface
{
    public:
        CpuQuinticFirstWithBreakoutsChecker(){}
        ~CpuQuinticFirstWithBreakoutsChecker(){}

        std::vector<int*>* findHits(
            const float needle,
            const float theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges
        );
};

#endif // CPU_POLYNOMIAL_CHECKER_HPP