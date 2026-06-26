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

class GpuQuinticFirstCheckerPositiveOnly : public PolynomialCheckerInterface
{
    public:
        GpuQuinticFirstCheckerPositiveOnly(){}
        ~GpuQuinticFirstCheckerPositiveOnly(){}

        std::vector<int*>* findHits(
            const double needle,
            const double theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges,
            long& floatHitCount
        );
};

class GpuQuinticFirstCheckerPositiveOnlyTopFour : public PolynomialCheckerInterface
{
    public:
        GpuQuinticFirstCheckerPositiveOnlyTopFour(){}
        ~GpuQuinticFirstCheckerPositiveOnlyTopFour(){}

        std::vector<int*>* findHits(
            const double needle,
            const double theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges,
            long& floatHitCount
        );
};

/** Flatten (quint,x,y) in the launch; inner loop over z only (megaman4). */
class GpuQuinticFirstCheckerPositiveOnlyTopFive : public PolynomialCheckerInterface
{
    public:
        GpuQuinticFirstCheckerPositiveOnlyTopFive(){}
        ~GpuQuinticFirstCheckerPositiveOnlyTopFive(){}

        std::vector<int*>* findHits(
            const double needle,
            const double theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges,
            long& floatHitCount
        );
};

/** Same geometry as TopFive; z loop is compiler-unrolled in-kernel (megaman5). */
class GpuQuinticFirstCheckerPositiveOnlyTopSix : public PolynomialCheckerInterface
{
    public:
        GpuQuinticFirstCheckerPositiveOnlyTopSix(){}
        ~GpuQuinticFirstCheckerPositiveOnlyTopSix(){}

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