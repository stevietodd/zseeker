#ifndef GPU_POLYNOMIAL_CHECKER_HPP
#define GPU_POLYNOMIAL_CHECKER_HPP

#include "PolynomialCheckerInterface.hpp"

class GpuQuinticLastChecker : public PolynomialCheckerInterface
{
    public:
        GpuQuinticLastChecker(){}
        ~GpuQuinticLastChecker(){}

        std::vector<int*>* findHits(
            const double needle,
            const double theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges,
            long& floatHitCount
        );
};

class GpuQuinticFirstChecker : public PolynomialCheckerInterface
{
    public:
        GpuQuinticFirstChecker(){}
        ~GpuQuinticFirstChecker(){}

        std::vector<int*>* findHits(
            const double needle,
            const double theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges,
            long& floatHitCount
        );
};

class GpuNoLookupTableChecker : public PolynomialCheckerInterface
{
    public:
		GpuNoLookupTableChecker(){}
        ~GpuNoLookupTableChecker(){}

        std::vector<int*>* findHits(
            const double needle,
            const double theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges,
            long& floatHitCount
        );
};

#endif // GPU_POLYNOMIAL_CHECKER_HPP