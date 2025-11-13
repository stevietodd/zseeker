#include "GpuPolynomialChecker.hpp"
#include "lookupTableAccessor.hpp"
#include "math.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{
constexpr int kLoopLevels = 6;
constexpr int kLoopBoundsCount = kLoopLevels * 2;
constexpr int kMaxDeviceHits = 1'048'576; // 1M hits * 6 ints ≈ 24 MB

struct LoopDeviceBounds
{
    int start[kLoopLevels];
    int end[kLoopLevels];
    int uCount;
};

struct BreakoutBounds
{
    float v1Low;
    float v1High;
    float v2Low;
    float v2High;
    float v3Low;
    float v3High;
    float v4Low;
    float v4High;
    float v5Low;
    float v5High;
};

struct SearchParams
{
    float needle;
    float theConst;
    float theConst2;
    float theConst3;
    float theConst4;
    float theConst5;
    float maxLowerDegreesValue;
    int lutSize;
    BreakoutBounds breakouts;
};

template <typename T>
inline T scalarMax(const T a, const T b)
{
    return (a > b) ? a : b;
}

inline void throwIfCudaFailed(cudaError_t err, const char* context)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error in %s: %s\n", context, cudaGetErrorString(err));
        std::abort();
    }
}

__device__ __forceinline__ float deviceGetFloatPrecisionBasedOnMaxValue(float maxValue)
{
    float absVal = fabsf(maxValue);
    if (absVal < 1.0f)
    {
        absVal = 1.0f;
    }
    const float exponent = floorf(log2f(absVal)) - 22.0f;
    return exp2f(exponent);
}

__device__ __forceinline__ bool lutLookup(
    const float* __restrict__ lut,
    int index,
    int lutSize,
    float& value,
    int* __restrict__ stats)
{
    const int absIndex = (index < 0) ? -index : index;
    if (stats != nullptr)
    {
        atomicMax(&stats[0], absIndex);
    }
    if (absIndex >= lutSize)
    {
        if (stats != nullptr)
        {
            atomicAdd(&stats[1], 1);
            stats[2] = index;
        }
        return false;
    }
    const float lutValue = lut[absIndex];
    value = (index < 0) ? -lutValue : lutValue;
    return true;
}

__global__ void gpuSearchKernel(
    const float* __restrict__ lut,
    LoopDeviceBounds bounds,
    SearchParams params,
    int maxHits,
    int* __restrict__ outHits,
    int* __restrict__ hitCount,
    int* __restrict__ overflowFlag,
    int* __restrict__ stats)
{
    const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx >= bounds.uCount)
    {
        return;
    }

    const int u = bounds.start[0] + globalIdx;
    float coeffU;
    if (!lutLookup(lut, u, params.lutSize, coeffU, stats))
    {
        return;
    }
    const float v5 = coeffU * params.theConst5;

    if (v5 < params.breakouts.v5Low || v5 > params.breakouts.v5High)
    {
        return;
    }

    const float maxValue = fmaxf(params.maxLowerDegreesValue, fmaxf(fabsf(v5), fabsf(coeffU)));
    const float floatTol = deviceGetFloatPrecisionBasedOnMaxValue(maxValue);

    for (int v = bounds.start[1]; v <= bounds.end[1]; ++v)
    {
        float coeffV;
        if (!lutLookup(lut, v, params.lutSize, coeffV, stats))
        {
            continue;
        }
        const float v4 = v5 + coeffV * params.theConst4;
        if (v4 < params.breakouts.v4Low || v4 > params.breakouts.v4High)
        {
            continue;
        }

        for (int w = bounds.start[2]; w <= bounds.end[2]; ++w)
        {
            float coeffW;
            if (!lutLookup(lut, w, params.lutSize, coeffW, stats))
            {
                continue;
            }
            const float v3 = v4 + coeffW * params.theConst3;
            if (v3 < params.breakouts.v3Low || v3 > params.breakouts.v3High)
            {
                continue;
            }

            for (int x = bounds.start[3]; x <= bounds.end[3]; ++x)
            {
                float coeffX;
                if (!lutLookup(lut, x, params.lutSize, coeffX, stats))
                {
                    continue;
                }
                const float v2 = v3 + coeffX * params.theConst2;
                if (v2 < params.breakouts.v2Low || v2 > params.breakouts.v2High)
                {
                    continue;
                }

                for (int y = bounds.start[4]; y <= bounds.end[4]; ++y)
                {
                    float coeffY;
                    if (!lutLookup(lut, y, params.lutSize, coeffY, stats))
                    {
                        continue;
                    }
                    const float v1 = v2 + coeffY * params.theConst;
                    if (v1 < params.breakouts.v1Low || v1 > params.breakouts.v1High)
                    {
                        continue;
                    }

                    for (int z = bounds.start[5]; z <= bounds.end[5]; ++z)
                    {
                        float coeffZ;
                        if (!lutLookup(lut, z, params.lutSize, coeffZ, stats))
                        {
                            continue;
                        }
                        const float v0 = v1 + coeffZ;
                        if (fabsf(v0 - params.needle) <= floatTol)
                        {
                            const int slot = atomicAdd(hitCount, 1);
                            if (slot < maxHits)
                            {
                                const int base = slot * 6;
                                outHits[base + 0] = u;
                                outHits[base + 1] = v;
                                outHits[base + 2] = w;
                                outHits[base + 3] = x;
                                outHits[base + 4] = y;
                                outHits[base + 5] = z;
                            }
                            else
                            {
                                atomicExch(overflowFlag, 1);
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace

std::vector<int*>* GpuQuinticFirstWithBreakoutsChecker::findHits(
    const double needle,
    const double theConst,
    const int degree,
    const float* coeffArray,
    const std::vector<int>* loopRanges,
    long& floatHitCount)
{
    (void)coeffArray; // coefficients come from the lookup table on GPU as well
    int loopStartEnds[kLoopBoundsCount] = {
        -608'383, 608'383,
        -152'231, 152'231,
        -6'087, 6'087,
        -2'203, 2'203,
        -555, 555,
        -143, 143
    };

    if (loopRanges != nullptr)
    {
        const int expected = 2 * (degree + 1);
        for (int loopRangeInd = 0; loopRangeInd < expected && loopRangeInd < kLoopBoundsCount; ++loopRangeInd)
        {
            if (loopRanges->at(loopRangeInd) < USE_DEFAULT)
            {
                loopStartEnds[loopRangeInd] = loopRanges->at(loopRangeInd);
            }
        }
    }

    const int maxIndex = static_cast<int>(getLookupTableSize()) - 1;
    for (int idx = 0; idx < kLoopBoundsCount; ++idx)
    {
        if (loopStartEnds[idx] > maxIndex)
        {
            loopStartEnds[idx] = maxIndex;
        }
        else if (loopStartEnds[idx] < -maxIndex)
        {
            loopStartEnds[idx] = -maxIndex;
        }
    }

    for (int level = 0; level < kLoopLevels; ++level)
    {
        const int startIdx = level * 2;
        const int endIdx = startIdx + 1;
        if (loopStartEnds[startIdx] > loopStartEnds[endIdx])
        {
            const int tmp = loopStartEnds[startIdx];
            loopStartEnds[startIdx] = loopStartEnds[endIdx];
            loopStartEnds[endIdx] = tmp;
        }
    }

    LoopDeviceBounds bounds{};
    for (int level = 0; level < kLoopLevels; ++level)
    {
        bounds.start[level] = loopStartEnds[level * 2];
        bounds.end[level] = loopStartEnds[(level * 2) + 1];
    }

    bounds.uCount = (bounds.end[0] - bounds.start[0]) + 1;
    if (bounds.uCount <= 0)
    {
        floatHitCount = 0;
        return new std::vector<int*>();
    }

    const double theConst2 = theConst * theConst;
    const double theConst3 = theConst2 * theConst;
    const double theConst4 = theConst3 * theConst;
    const double theConst5 = theConst4 * theConst;

    const float needlef = static_cast<float>(needle);
    const float theConstf = static_cast<float>(theConst);
    const float theConst2f = static_cast<float>(theConst2);
    const float theConst3f = static_cast<float>(theConst3);
    const float theConst4f = static_cast<float>(theConst4);
    const float theConst5f = static_cast<float>(theConst5);

    const float uplim4 = 500.0f;
    const float uplim3 = 100.0f;
    const float uplim2 = 60.0f;
    const float uplim1 = 30.0f;
    const float uplim0 = 15.0f;

    const float v0max = uplim0;
    const float v1max = uplim1 * theConstf + v0max;
    const float v2max = uplim2 * theConst2f + v1max;
    const float v3max = uplim3 * theConst3f + v2max;
    const float v4max = uplim4 * theConst4f + v3max;

    const float tolerance = needlef;

    BreakoutBounds breakouts{};
    breakouts.v1Low = needlef - v0max - tolerance;
    breakouts.v1High = needlef + v0max + tolerance;
    breakouts.v2Low = needlef - v1max - tolerance;
    breakouts.v2High = needlef + v1max + tolerance;
    breakouts.v3Low = needlef - v2max - tolerance;
    breakouts.v3High = needlef + v2max + tolerance;
    breakouts.v4Low = needlef - v3max - tolerance;
    breakouts.v4High = needlef + v3max + tolerance;
    breakouts.v5Low = needlef - v4max - tolerance;
    breakouts.v5High = needlef + v4max + tolerance;

    const float maxLowerDegreesValue = (std::fabs(theConstf) < 1.0f)
        ? scalarMax(
              500.0f,
              500.0f * theConst4f + 100.0f * theConst3f + 60.0f * theConst2f + 30.0f * theConstf + 15.0f)
        : scalarMax(
              scalarMax(
                  std::fabs(theConst5f),
                  500.0f * theConst4f),
              scalarMax(
                  500.0f,
                  500.0f * theConst4f + 100.0f * theConst3f + 60.0f * theConst2f + 30.0f * theConstf + 15.0f));

    const std::size_t lutElements = getLookupTableSize();
    const float* lutHost = getLookupTableFloat();
    const double* doubleLutHost = getLookupTableDouble();

    SearchParams params{};
    params.needle = needlef;
    params.theConst = theConstf;
    params.theConst2 = theConst2f;
    params.theConst3 = theConst3f;
    params.theConst4 = theConst4f;
    params.theConst5 = theConst5f;
    params.maxLowerDegreesValue = maxLowerDegreesValue;
    params.lutSize = static_cast<int>(lutElements);
    params.breakouts = breakouts;

    float* d_lut = nullptr;
    int* d_hits = nullptr;
    int* d_hitCount = nullptr;
    int* d_overflow = nullptr;
    int* d_stats = nullptr;

    const std::size_t lutBytes = lutElements * sizeof(float);
    const std::size_t hitBytes = static_cast<std::size_t>(kMaxDeviceHits) * 6 * sizeof(int);

    throwIfCudaFailed(cudaMalloc(reinterpret_cast<void**>(&d_lut), lutBytes), "cudaMalloc(d_lut)");
    throwIfCudaFailed(cudaMalloc(reinterpret_cast<void**>(&d_hits), hitBytes), "cudaMalloc(d_hits)");
    throwIfCudaFailed(cudaMalloc(reinterpret_cast<void**>(&d_hitCount), sizeof(int)), "cudaMalloc(d_hitCount)");
    throwIfCudaFailed(cudaMalloc(reinterpret_cast<void**>(&d_overflow), sizeof(int)), "cudaMalloc(d_overflow)");
    throwIfCudaFailed(cudaMalloc(reinterpret_cast<void**>(&d_stats), sizeof(int) * 3), "cudaMalloc(d_stats)");

    throwIfCudaFailed(cudaMemcpy(d_lut, lutHost, lutBytes, cudaMemcpyHostToDevice), "cudaMemcpy(d_lut)");
    throwIfCudaFailed(cudaMemset(d_hits, 0, hitBytes), "cudaMemset(d_hits)");
    throwIfCudaFailed(cudaMemset(d_hitCount, 0, sizeof(int)), "cudaMemset(d_hitCount)");
    throwIfCudaFailed(cudaMemset(d_overflow, 0, sizeof(int)), "cudaMemset(d_overflow)");
    throwIfCudaFailed(cudaMemset(d_stats, 0, sizeof(int) * 3), "cudaMemset(d_stats)");

    const int threadsPerBlock = 256;
    const int blocks = (bounds.uCount + threadsPerBlock - 1) / threadsPerBlock;

    gpuSearchKernel<<<blocks, threadsPerBlock>>>(d_lut, bounds, params, kMaxDeviceHits, d_hits, d_hitCount, d_overflow, d_stats);
    throwIfCudaFailed(cudaGetLastError(), "gpuSearchKernel launch");
    throwIfCudaFailed(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    int h_hitCount = 0;
    int h_overflow = 0;
    throwIfCudaFailed(cudaMemcpy(&h_hitCount, d_hitCount, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(hitCount)");
    throwIfCudaFailed(cudaMemcpy(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(overflow)");
    int h_stats[3] = {0, 0, 0};
    throwIfCudaFailed(cudaMemcpy(h_stats, d_stats, sizeof(int) * 3, cudaMemcpyDeviceToHost), "cudaMemcpy(stats)");

    const int hitsToCopy = (h_hitCount < kMaxDeviceHits) ? h_hitCount : kMaxDeviceHits;
    std::vector<int> rawHits(static_cast<size_t>(hitsToCopy) * 6);
    if (hitsToCopy > 0)
    {
        throwIfCudaFailed(
            cudaMemcpy(rawHits.data(), d_hits, rawHits.size() * sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy(rawHits)");
    }

    throwIfCudaFailed(cudaFree(d_lut), "cudaFree(d_lut)");
    throwIfCudaFailed(cudaFree(d_hits), "cudaFree(d_hits)");
    throwIfCudaFailed(cudaFree(d_hitCount), "cudaFree(d_hitCount)");
    throwIfCudaFailed(cudaFree(d_overflow), "cudaFree(d_overflow)");
    throwIfCudaFailed(cudaFree(d_stats), "cudaFree(d_stats)");

    auto* results = new std::vector<int*>();
    results->reserve(hitsToCopy);

    const double maxLowerDegreesValueD = static_cast<double>(maxLowerDegreesValue);

    for (int hitIdx = 0; hitIdx < hitsToCopy; ++hitIdx)
    {
        const int u = rawHits[6 * hitIdx + 0];
        const int v = rawHits[6 * hitIdx + 1];
        const int w = rawHits[6 * hitIdx + 2];
        const int x = rawHits[6 * hitIdx + 3];
        const int y = rawHits[6 * hitIdx + 4];
        const int z = rawHits[6 * hitIdx + 5];

        const double coeffU = (u < 0) ? -doubleLutHost[-u] : doubleLutHost[u];
        const double coeffV = (v < 0) ? -doubleLutHost[-v] : doubleLutHost[v];
        const double coeffW = (w < 0) ? -doubleLutHost[-w] : doubleLutHost[w];
        const double coeffX = (x < 0) ? -doubleLutHost[-x] : doubleLutHost[x];
        const double coeffY = (y < 0) ? -doubleLutHost[-y] : doubleLutHost[y];
        const double coeffZ = (z < 0) ? -doubleLutHost[-z] : doubleLutHost[z];

        const double v5 = coeffU * theConst5;
        const double maxValue = scalarMax(
            maxLowerDegreesValueD,
            scalarMax(std::fabs(v5), std::fabs(coeffU)));
        const double doubleTol = getDoublePrecisionBasedOnMaxValue(maxValue);

        const double doubleValue =
            v5 +
            coeffV * theConst4 +
            coeffW * theConst3 +
            coeffX * theConst2 +
            coeffY * theConst +
            coeffZ;

        if (DOUBLE_BASICALLY_EQUAL(doubleValue, needle, doubleTol))
        {
            results->push_back(new int[6]{u, v, w, x, y, z});
        }
    }

    floatHitCount = h_hitCount;

    if (h_overflow != 0)
    {
        fprintf(stderr, "WARNING: GPU hit buffer overflowed; results may be truncated.\n");
    }

    if (h_stats[1] != 0)
    {
        fprintf(stderr, "WARNING: GPU LUT lookup failures=%d, maxAbsIndex=%d, lastIndex=%d\n",
                h_stats[1],
                h_stats[0],
                h_stats[2]);
    }

    return results;
}

