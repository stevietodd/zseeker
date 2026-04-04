#include "GpuPolynomialChecker.hpp"
#include "math.hpp"
#include "lookupTableAccessor.hpp"

#include "cudawrapper.hpp"
#include <iostream>
using namespace std;

// "Top Five" variant:
// Flatten (quint, x, y) into grid.x (with optional multi-launch batching via combinedBaseIndex).
// quart/cubic remain on grid.y / grid.z. Each thread runs only the z loop.
__global__ static void compareToZeta5loopPositiveOnlyTopFive(
    int *out,
    const double theConst,
    const double needle,
    const float *coeffArray,
    const double *doubleCoeffArray,
    const int *loopStartEnds,
    int *floatHitCount,
    int *doubleHitCount,
    const double theConst2,
    const double theConst3,
    const double theConst4,
    const double theConst5,
    const float floatTol,
    const double doubleTol,
    const float v3BreakoutHigh,
    const float v3BreakoutLow,
    const float v2BreakoutHigh,
    const float v2BreakoutLow,
    const float v1BreakoutHigh,
    const float v1BreakoutLow,
    const long long combinedBaseIndex
) {
    __shared__ int s_loopStartEnds[12];
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        for (int i = 0; i < 12; i++) {
            s_loopStartEnds[i] = loopStartEnds[i];
        }
    }
    __syncthreads();

    const int quintStart = s_loopStartEnds[0];
    const int quintEnd = s_loopStartEnds[1];
    const int quintRange = quintEnd - quintStart + 1;

    const int quartStart = s_loopStartEnds[2];
    const int quartEnd = s_loopStartEnds[3];
    const int quartRange = quartEnd - quartStart + 1;

    const int cubicStart = s_loopStartEnds[4];
    const int cubicEnd = s_loopStartEnds[5];
    const int cubicRange = cubicEnd - cubicStart + 1;

    const int xStart = s_loopStartEnds[6];
    const int xEnd = s_loopStartEnds[7];
    const int xRange = xEnd - xStart + 1;

    const int yStart = s_loopStartEnds[8];
    const int yEnd = s_loopStartEnds[9];
    const int yRange = yEnd - yStart + 1;

    const int zStart = s_loopStartEnds[10];
    const int zEnd = s_loopStartEnds[11];

    const long long quintXYThreadIdx =
        combinedBaseIndex + (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    const long long combinedTotal = (long long)quintRange * (long long)xRange * (long long)yRange;
    if (quintXYThreadIdx >= combinedTotal) return;

    const int yOffset = (int)(quintXYThreadIdx % (long long)yRange);
    long long t = quintXYThreadIdx / (long long)yRange;
    const int xOffset = (int)(t % (long long)xRange);
    const int quintOffset = (int)(t / (long long)xRange);

    const int quintInd = quintStart + quintOffset;
    const int x = xStart + xOffset;
    const int y = yStart + yOffset;

    const int quartThreadIdx = blockIdx.y * blockDim.y + threadIdx.y;
    const int cubicThreadIdx = blockIdx.z * blockDim.z + threadIdx.z;
    if (quartThreadIdx >= quartRange || cubicThreadIdx >= cubicRange) return;

    const int quartInd = quartStart + quartThreadIdx;
    const int cubicInd = cubicStart + cubicThreadIdx;

    const int quintAbs = (quintInd < 0) ? -quintInd : quintInd;
    const int quartAbs = (quartInd < 0) ? -quartInd : quartInd;
    const int cubicAbs = (cubicInd < 0) ? -cubicInd : cubicInd;

    if (quintAbs >= 608'384 || quartAbs >= 608'384 || cubicAbs >= 608'384) return;

    if (quintInd == 0 && quartInd < 0) return;
    if (quintInd == 0 && quartInd == 0 && cubicInd < 0) return;

    const float theConstf = (float)theConst;
    const float theConst2f = (float)theConst2;
    const float theConst3f = (float)theConst3;
    const float theConst4f = (float)theConst4;
    const float theConst5f = (float)theConst5;

    const float needlef = (float)needle;
    const float negNeedlef = -needlef;
    const double negNeedle = -needle;

    const float quartSignF = (quartInd < 0) ? -1.0f : 1.0f;
    const float cubicSignF = (cubicInd < 0) ? -1.0f : 1.0f;
    const double quartSignD = (quartInd < 0) ? -1.0 : 1.0;
    const double cubicSignD = (cubicInd < 0) ? -1.0 : 1.0;

    const int xAbs = (x < 0) ? -x : x;
    const float xSignF = (x < 0) ? -1.0f : 1.0f;
    const double xSignD = (x < 0) ? -1.0 : 1.0;

    const int yAbs = (y < 0) ? -y : y;
    const float ySignF = (y < 0) ? -1.0f : 1.0f;
    const double ySignD = (y < 0) ? -1.0 : 1.0;

    const float topThreeTerms =
        (__ldg(&coeffArray[quintAbs]) * theConst5f) +
        (__ldg(&coeffArray[quartAbs]) * quartSignF * theConst4f) +
        (__ldg(&coeffArray[cubicAbs]) * cubicSignF * theConst3f);

    if (topThreeTerms < v3BreakoutLow || topThreeTerms > v3BreakoutHigh) return;

    const float v2 = topThreeTerms + (__ldg(&coeffArray[xAbs]) * xSignF * theConst2f);
    if (v2 < v2BreakoutLow || v2 > v2BreakoutHigh) return;

    float v1 = v2 + (__ldg(&coeffArray[yAbs]) * ySignF * theConstf);
    if (v1 < v1BreakoutLow || v1 > v1BreakoutHigh) return;

    for (int z = zStart; z <= zEnd; z++) {
        const int zAbs = (z < 0) ? -z : z;
        const float zSignF = (z < 0) ? -1.0f : 1.0f;
        const double zSignD = (z < 0) ? -1.0 : 1.0;

        const float v0 = v1 + (__ldg(&coeffArray[zAbs]) * zSignF);

        const bool matchesNeedle = FLOAT_BASICALLY_EQUAL(v0, needlef, floatTol);
        const bool matchesNegNeedle = FLOAT_BASICALLY_EQUAL(v0, negNeedlef, floatTol);
        if (!matchesNeedle && !matchesNegNeedle) continue;

        const double doubleValue =
            (__ldg(&doubleCoeffArray[quintAbs]) * theConst5) +
            (__ldg(&doubleCoeffArray[quartAbs]) * quartSignD * theConst4) +
            (__ldg(&doubleCoeffArray[cubicAbs]) * cubicSignD * theConst3) +
            (__ldg(&doubleCoeffArray[xAbs]) * xSignD * theConst2) +
            (__ldg(&doubleCoeffArray[yAbs]) * ySignD * theConst) +
            (__ldg(&doubleCoeffArray[zAbs]) * zSignD);

        const double targetNeedle = matchesNeedle ? needle : negNeedle;
        if (!DOUBLE_BASICALLY_EQUAL(doubleValue, targetNeedle, doubleTol)) continue;

        const int neg = matchesNeedle ? 1 : -1;
        const int i = atomicAdd(doubleHitCount, 1);
        out[6 * i]     = neg * quintInd;
        out[6 * i + 1] = neg * quartInd;
        out[6 * i + 2] = neg * cubicInd;
        out[6 * i + 3] = neg * x;
        out[6 * i + 4] = neg * y;
        out[6 * i + 5] = neg * z;
    }
}

std::vector<int*>* GpuQuinticFirstCheckerPositiveOnlyTopFive::findHits(
    const double needle,
    const double theConst,
    const int degree,
    const float *coeffArray,
    const std::vector<int> *loopRanges,
    long& floatHitCount
) {
    int loopStartEnds[12] = {0, 608'383, -152'231, 152'231, -6'087, 6'087, -2'203, 2'203, -555, 555, -143, 143};
    int coeffArraySize = 608'384;

    if (loopRanges != NULL) {
        for (int loopRangeInd = 0; loopRangeInd < (2 * (degree + 1)); loopRangeInd++) {
            if (loopRanges->at(loopRangeInd) < USE_DEFAULT) {
                if (loopRangeInd == 0 && loopRanges->at(loopRangeInd) < 0) {
                    std::cout << "WARNING: You have set a negative quint start. Ignoring negative quint start and using quint start of 0." << std::endl;
                    continue;
                }
                loopStartEnds[loopRangeInd] = loopRanges->at(loopRangeInd);
            }
        }
    }

    float floatTol = FLOAT_POS_ERROR_DEFAULT;
    double doubleTol = DOUBLE_POS_ERROR_DEFAULT;
    const double theConst2 = powl(theConst, (double)2);
    const double theConst3 = powl(theConst, (double)3);
    const double theConst4 = powl(theConst, (double)4);
    const double theConst5 = powl(theConst, (double)5);

    const float needlef = (float)needle;
    const float theConst5f = (float)theConst5;
    const float theConst4f = (float)theConst4;
    const float theConst3f = (float)theConst3;
    const float theConst2f = (float)theConst2;
    const float theConstf = (float)theConst;

    const int quintStart = loopStartEnds[0];
    const int quintEnd = loopStartEnds[1];
    const int quintRange = quintEnd - quintStart + 1;

    const int quartStart = loopStartEnds[2];
    const int quartEnd = loopStartEnds[3];
    const int quartRange = quartEnd - quartStart + 1;

    const int cubicStart = loopStartEnds[4];
    const int cubicEnd = loopStartEnds[5];
    const int cubicRange = cubicEnd - cubicStart + 1;

    const int xStart = loopStartEnds[6];
    const int xEnd = loopStartEnds[7];
    const int xRange = xEnd - xStart + 1;

    const int yStart = loopStartEnds[8];
    const int yEnd = loopStartEnds[9];
    const int yRange = yEnd - yStart + 1;

    std::vector<int*> *results = new std::vector<int*>();
    float *d_coeffArray;
    double *d_doubleCoeffArray;
    int *d_out;
    int *d_loopStartEnds;
    int *out = new int[coeffArraySize * 6];
    int h_hitCount = 0;
    int h_doubleHitCount = 0;

    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        cerr << "CUDA device initialization error: " << cudaGetErrorString(err) << endl;
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        cerr << "CUDA get device properties error: " << cudaGetErrorString(err) << endl;
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    floatTol = getFloatPrecisionBasedOnMaxValue(
        std::max(1000.0f,
            1000.0f * std::abs(theConst5f) +
            500.0f * std::abs(theConst4f) +
            100.0f * std::abs(theConst3f) +
            60.0f * std::abs(theConst2f) +
            30.0f * std::abs(theConstf) +
            15.0f
        )
    );
    doubleTol = getDoublePrecisionBasedOnMaxValue(
        std::max(1000.0f,
            1000.0f * std::abs(theConst5f) +
            500.0f * std::abs(theConst4f) +
            100.0f * std::abs(theConst3f) +
            60.0f * std::abs(theConst2f) +
            30.0f * std::abs(theConstf) +
            15.0f
        )
    );

    const float uplim0 = 15.0f;
    const float uplim1 = 30.0f;
    const float uplim2 = 60.0f;
    const float uplim3 = 100.0f;

    const float v1max = uplim1 * theConstf + uplim0;
    const float v2max = uplim2 * theConst2f + v1max;
    const float v3max = uplim3 * theConst3f + v2max;
    (void)v3max;

    const float tolerance = std::abs(needlef);
    const float v1BreakoutHigh = needlef + uplim0 + tolerance;
    const float v1BreakoutLow = needlef - uplim0 - tolerance;
    const float v2BreakoutHigh = needlef + v1max + tolerance;
    const float v2BreakoutLow = needlef - v1max - tolerance;
    const float v3BreakoutHigh = needlef + v2max + tolerance;
    const float v3BreakoutLow = needlef - v2max - tolerance;

    const float negNeedlef = -needlef;
    const float v1BreakoutHighNeg = negNeedlef + uplim0 + tolerance;
    const float v1BreakoutLowNeg = negNeedlef - uplim0 - tolerance;
    const float v2BreakoutHighNeg = negNeedlef + v1max + tolerance;
    const float v2BreakoutLowNeg = negNeedlef - v1max - tolerance;
    const float v3BreakoutHighNeg = negNeedlef + v2max + tolerance;
    const float v3BreakoutLowNeg = negNeedlef - v2max - tolerance;

    const float v1BH = std::max(v1BreakoutHigh, v1BreakoutHighNeg);
    const float v1BL = std::min(v1BreakoutLow, v1BreakoutLowNeg);
    const float v2BH = std::max(v2BreakoutHigh, v2BreakoutHighNeg);
    const float v2BL = std::min(v2BreakoutLow, v2BreakoutLowNeg);
    const float v3BH = std::max(v3BreakoutHigh, v3BreakoutHighNeg);
    const float v3BL = std::min(v3BreakoutLow, v3BreakoutLowNeg);

    int *d_hitCount = nullptr;
    int *d_doubleHitCount = nullptr;

    err = cudaMalloc((void**)&d_hitCount, sizeof(int));
    if (err != cudaSuccess) { delete[] out; floatHitCount = 0; return results; }
    err = cudaMalloc((void**)&d_doubleHitCount, sizeof(int));
    if (err != cudaSuccess) { cudaFree(d_hitCount); delete[] out; floatHitCount = 0; return results; }
    err = cudaMalloc((void**)&d_coeffArray, sizeof(float) * coeffArraySize);
    if (err != cudaSuccess) { cudaFree(d_doubleHitCount); cudaFree(d_hitCount); delete[] out; floatHitCount = 0; return results; }
    err = cudaMalloc((void**)&d_doubleCoeffArray, sizeof(double) * coeffArraySize);
    if (err != cudaSuccess) { cudaFree(d_coeffArray); cudaFree(d_doubleHitCount); cudaFree(d_hitCount); delete[] out; floatHitCount = 0; return results; }
    err = cudaMalloc((void**)&d_out, sizeof(int) * coeffArraySize * 6);
    if (err != cudaSuccess) { cudaFree(d_doubleCoeffArray); cudaFree(d_coeffArray); cudaFree(d_doubleHitCount); cudaFree(d_hitCount); delete[] out; floatHitCount = 0; return results; }
    err = cudaMalloc((void**)&d_loopStartEnds, sizeof(int) * 12);
    if (err != cudaSuccess) { cudaFree(d_out); cudaFree(d_doubleCoeffArray); cudaFree(d_coeffArray); cudaFree(d_doubleHitCount); cudaFree(d_hitCount); delete[] out; floatHitCount = 0; return results; }

    for (int o = 0; o < coeffArraySize * 6; o++) out[o] = 0;

    err = cudaMemcpy(d_hitCount, &h_hitCount, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_loopStartEnds); cudaFree(d_out); cudaFree(d_doubleCoeffArray); cudaFree(d_coeffArray); cudaFree(d_doubleHitCount); cudaFree(d_hitCount);
        delete[] out; floatHitCount = 0; return results;
    }

    err = cudaMemcpy(d_doubleHitCount, &h_doubleHitCount, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_loopStartEnds); cudaFree(d_out); cudaFree(d_doubleCoeffArray); cudaFree(d_coeffArray); cudaFree(d_doubleHitCount); cudaFree(d_hitCount);
        delete[] out; floatHitCount = 0; return results;
    }

    err = cudaMemcpy(d_coeffArray, coeffArray, sizeof(float) * coeffArraySize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_loopStartEnds); cudaFree(d_out); cudaFree(d_doubleCoeffArray); cudaFree(d_coeffArray); cudaFree(d_doubleHitCount); cudaFree(d_hitCount);
        delete[] out; floatHitCount = 0; return results;
    }

    const double* doubleCoeffArray = getLookupTableDouble();
    err = cudaMemcpy(d_doubleCoeffArray, doubleCoeffArray, sizeof(double) * coeffArraySize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_loopStartEnds); cudaFree(d_out); cudaFree(d_doubleCoeffArray); cudaFree(d_coeffArray); cudaFree(d_doubleHitCount); cudaFree(d_hitCount);
        delete[] out; floatHitCount = 0; return results;
    }

    err = cudaMemcpy(d_loopStartEnds, loopStartEnds, sizeof(int) * 12, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_loopStartEnds); cudaFree(d_out); cudaFree(d_doubleCoeffArray); cudaFree(d_coeffArray); cudaFree(d_doubleHitCount); cudaFree(d_hitCount);
        delete[] out; floatHitCount = 0; return results;
    }

    dim3 blocksizes(8, 8, 8);
    const long long combinedTotal = (long long)quintRange * (long long)xRange * (long long)yRange;
    const unsigned long long threadsPerBlockX = (unsigned long long)blocksizes.x;
    const unsigned long long maxGridX = (unsigned long long)prop.maxGridSize[0];
    const unsigned long long maxThreadsPerLaunch = maxGridX * threadsPerBlockX;

    const unsigned int gridDimY =
        (quintRange == 0) ? 0U : (unsigned int)((quartRange + blocksizes.y - 1) / (unsigned long long)blocksizes.y);
    const unsigned int gridDimZ =
        (unsigned int)((cubicRange + blocksizes.z - 1) / (unsigned long long)blocksizes.z);

    cout << "GpuQuinticFirstCheckerPositiveOnlyTopFive: quart/cubic grid: " << gridDimY << "," << gridDimZ
         << " (ranges: quint=" << quintRange << ",quart=" << quartRange << ",cubic=" << cubicRange
         << ",x=" << xRange << ",y=" << yRange << ", linear=" << combinedTotal << ")" << endl;

    cudaEvent_t kernelStart, kernelStop;
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);
    cudaEventRecord(kernelStart);

    float kernelTimeMs = 0.0f;
    for (long long base = 0; base < combinedTotal; base += (long long)maxThreadsPerLaunch) {
        unsigned long long remaining = (unsigned long long)(combinedTotal - base);
        unsigned long long thisChunk = remaining < maxThreadsPerLaunch ? remaining : maxThreadsPerLaunch;
        unsigned int gridDimX = (unsigned int)((thisChunk + threadsPerBlockX - 1) / threadsPerBlockX);
        dim3 gridsizes(gridDimX, gridDimY, gridDimZ);

        compareToZeta5loopPositiveOnlyTopFive<<<gridsizes, blocksizes>>>(
            d_out,
            theConst,
            needle,
            d_coeffArray,
            d_doubleCoeffArray,
            d_loopStartEnds,
            d_hitCount,
            d_doubleHitCount,
            theConst2, theConst3, theConst4, theConst5,
            floatTol, doubleTol,
            v3BH, v3BL,
            v2BH, v2BL,
            v1BH, v1BL,
            base
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cerr << "CUDA kernel launch error (TopFive): " << cudaGetErrorString(err) << endl;
            cudaEventDestroy(kernelStart); cudaEventDestroy(kernelStop);
            cudaFree(d_loopStartEnds); cudaFree(d_out); cudaFree(d_doubleCoeffArray); cudaFree(d_coeffArray); cudaFree(d_doubleHitCount); cudaFree(d_hitCount);
            delete[] out; floatHitCount = 0; return results;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cerr << "CUDA kernel execution error (TopFive): " << cudaGetErrorString(err) << endl;
            cudaEventDestroy(kernelStart); cudaEventDestroy(kernelStop);
            cudaFree(d_loopStartEnds); cudaFree(d_out); cudaFree(d_doubleCoeffArray); cudaFree(d_coeffArray); cudaFree(d_doubleHitCount); cudaFree(d_hitCount);
            delete[] out; floatHitCount = 0; return results;
        }
    }

    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);
    cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop);
    cout << "Kernel execution time (topFive, batched): " << kernelTimeMs << " ms" << endl;
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);

    err = cudaMemcpy(&h_hitCount, d_hitCount, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_loopStartEnds); cudaFree(d_out); cudaFree(d_doubleCoeffArray); cudaFree(d_coeffArray); cudaFree(d_doubleHitCount); cudaFree(d_hitCount);
        delete[] out; floatHitCount = 0; return results;
    }

    err = cudaMemcpy(&h_doubleHitCount, d_doubleHitCount, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_loopStartEnds); cudaFree(d_out); cudaFree(d_doubleCoeffArray); cudaFree(d_coeffArray); cudaFree(d_doubleHitCount); cudaFree(d_hitCount);
        delete[] out; floatHitCount = 0; return results;
    }

    err = cudaMemcpy(out, d_out, sizeof(int) * coeffArraySize * 6, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_loopStartEnds); cudaFree(d_out); cudaFree(d_doubleCoeffArray); cudaFree(d_coeffArray); cudaFree(d_doubleHitCount); cudaFree(d_hitCount);
        delete[] out; floatHitCount = 0; return results;
    }

    for (int j = 0; j < h_doubleHitCount; j++) {
        results->push_back(new int[6] {out[6 * j], out[6 * j + 1], out[6 * j + 2], out[6 * j + 3], out[6 * j + 4], out[6 * j + 5]});
    }

    cudaFree(d_loopStartEnds);
    cudaFree(d_out);
    cudaFree(d_doubleCoeffArray);
    cudaFree(d_coeffArray);
    cudaFree(d_doubleHitCount);
    cudaFree(d_hitCount);
    delete[] out;

    floatHitCount = h_hitCount;
    return results;
}
