#include "GpuPolynomialChecker.hpp"
#include "hitRefinement.hpp"
#include "math.hpp"
#include "lookupTableAccessor.hpp"

// cuda.cu
#include "cudawrapper.hpp"
#include <iostream>
using namespace std;

// Positive-only quint loop variant: loops 0..quintEnd, compares against both needle and -needle.
// When v0 matches -needle, records -quintInd in output (the negative solution).
__global__ static void compareToZeta5loopPositiveOnly(
    int *__restrict__ out,
    const double theConst,
    const double needle,
    const float *__restrict__ coeffArray,
    const double *__restrict__ doubleCoeffArray,
    const int *__restrict__ loopStartEnds,
    int *__restrict__ floatHitCount,
    int *__restrict__ doubleHitCount,
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
    const float v1BreakoutLow
) {
    // Shared memory cache for loopStartEnds (all threads in block read same values, reduces global memory traffic)
    __shared__ int s_loopStartEnds[12];
    // TODO: MegaMan maybe experiement to see if this shared memory voodoo actually helps speed up the kernel
    // First thread in block loads loopStartEnds into shared memory
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        for (int i = 0; i < 12; i++) {
            s_loopStartEnds[i] = loopStartEnds[i];
        }
    }
    __syncthreads();

    // Precompute float constants from precomputed double powers (avoid expensive pow() calls)
    const float theConstf = (float)theConst;
    const float theConst2f = (float)theConst2;
    const float theConst3f = (float)theConst3;
    const float theConst4f = (float)theConst4;
    const float theConst5f = (float)theConst5;
    const float needlef = (float)needle;
    const float negNeedlef = -needlef;
    const double negNeedle = -needle;

    // Get thread indices
    const int quintThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int quartThreadIdx = blockIdx.y * blockDim.y + threadIdx.y;
    const int cubicThreadIdx = blockIdx.z * blockDim.z + threadIdx.z;

    // Calculate ranges from shared memory (faster than global memory)
    // NOTE: quintStart is forced to 0 by host; we only iterate positive quint values
    const int quintStart = s_loopStartEnds[0];
    const int quintEnd = s_loopStartEnds[1];
    const int quintRange = quintEnd - quintStart + 1;

    const int quartStart = s_loopStartEnds[2];
    const int quartEnd = s_loopStartEnds[3];
    const int quartRange = quartEnd - quartStart + 1;

    const int cubicStart = s_loopStartEnds[4];
    const int cubicEnd = s_loopStartEnds[5];
    const int cubicRange = cubicEnd - cubicStart + 1;

    // Check if thread is within range
    if (quintThreadIdx >= quintRange || quartThreadIdx >= quartRange || cubicThreadIdx >= cubicRange) {
        return;
    }

    // Map thread indices to actual coefficient indices
    // NOTE: quintInd is always >= 0 (positive-only loop)
    const int quintInd = quintStart + quintThreadIdx;
    const int quartInd = quartStart + quartThreadIdx;
    const int cubicInd = cubicStart + cubicThreadIdx;

    // Get absolute values for array access and check bounds
    const int quintAbs = (quintInd < 0) ? -quintInd : quintInd;
    const int quartAbs = (quartInd < 0) ? -quartInd : quartInd;
    const int cubicAbs = (cubicInd < 0) ? -cubicInd : cubicInd;

    // this is just a sanity check to make sure we don't access out of bounds
    if (quintAbs >= 608'384 || quartAbs >= 608'384 || cubicAbs >= 608'384) {
        return;
    }

    // quintInd is always >= 0 for this positive-only variant (quintStart forced to 0 on host),
    // so the quint sign is +1. quart/cubic can be negative depending on thread indices.
    const float quartSignF = (quartInd < 0) ? -1.0f : 1.0f;
    const float cubicSignF = (cubicInd < 0) ? -1.0f : 1.0f;
    const double quartSignD = (quartInd < 0) ? -1.0 : 1.0;
    const double cubicSignD = (cubicInd < 0) ? -1.0 : 1.0;

    // When quintInd == 0 the quintic term is zero; (0,q,c,...) and (0,-q,-c,...) are the same polynomial up to sign.
    // Skip the negative half of (quart,cubic) so we do half the work for the quint==0 slice.
    // We also skip the negative half of (cubic) if quint==0 and quart==0 since we've already visited the positive half.
    if (quintInd == 0 && quartInd < 0) return;
    if (quintInd == 0 && quartInd == 0 && cubicInd < 0) return;

    // Calculate top three terms using precomputed powers (much faster than pow()).
    // Use read-only cache (_ldg) for LUT loads.
    const float topThreeTerms =
        (__ldg(&coeffArray[quintAbs]) * theConst5f)
        + (__ldg(&coeffArray[quartAbs]) * quartSignF * theConst4f)
        + (__ldg(&coeffArray[cubicAbs]) * cubicSignF * theConst3f);

    // Check v3 breakout
    if (topThreeTerms < v3BreakoutLow || topThreeTerms > v3BreakoutHigh) {
        return; // can't possibly get back to needle OR -needle
    }

    // Inner loops - note these use <= (less than or EQUAL TO)
    // Order: x (quadratic) outermost, y (linear) middle, z (constant) innermost (matching CPU)
    const int xStart = s_loopStartEnds[6];
    const int xEnd = s_loopStartEnds[7];
    const int yStart = s_loopStartEnds[8];
    const int yEnd = s_loopStartEnds[9];
    const int zStart = s_loopStartEnds[10];
    const int zEnd = s_loopStartEnds[11];

    float v0, v1, v2;
    for (int x = xStart; x <= xEnd; x++) {
        const int xAbs = (x < 0) ? -x : x;
        const float xSignF = (x < 0) ? -1.0f : 1.0f;
        const double xSignD = (x < 0) ? -1.0 : 1.0;
        // Calculate v2 = topThreeTerms + quadratic term
        v2 = topThreeTerms + (__ldg(&coeffArray[xAbs]) * xSignF * theConst2f);
        
        // Check v2 breakout
        if (v2 < v2BreakoutLow || v2 > v2BreakoutHigh) {
            continue; // can't possibly get back to needle, skip to next x
        }

        for (int y = yStart; y <= yEnd; y++) {
            const int yAbs = (y < 0) ? -y : y;
            const float ySignF = (y < 0) ? -1.0f : 1.0f;
            const double ySignD = (y < 0) ? -1.0 : 1.0;
            // Calculate v1 = v2 + linear term
            v1 = v2 + (__ldg(&coeffArray[yAbs]) * ySignF * theConstf);
            
            // Check v1 breakout
            if (v1 < v1BreakoutLow || v1 > v1BreakoutHigh) {
                continue; // can't possibly get back to needle, skip to next y
            }

            for (int z = zStart; z <= zEnd; z++) {
                const int zAbs = (z < 0) ? -z : z;
                const float zSignF = (z < 0) ? -1.0f : 1.0f;
                const double zSignD = (z < 0) ? -1.0 : 1.0;
                // Calculate v0 = v1 + constant term (final sum)
                v0 = v1 + (__ldg(&coeffArray[zAbs]) * zSignF);

                // Compare against both needle and -needle
                const bool matchesNeedle = FLOAT_BASICALLY_EQUAL(v0, needlef, floatTol);
                const bool matchesNegNeedle = FLOAT_BASICALLY_EQUAL(v0, negNeedlef, floatTol);
                if (!matchesNeedle && !matchesNegNeedle) continue;

                // Calculate double precision value for verification (using precomputed powers)
                const double doubleValue =
                    (__ldg(&doubleCoeffArray[quintAbs]) * theConst5)
                    + (__ldg(&doubleCoeffArray[quartAbs]) * quartSignD * theConst4)
                    + (__ldg(&doubleCoeffArray[cubicAbs]) * cubicSignD * theConst3)
                    + (__ldg(&doubleCoeffArray[xAbs]) * xSignD * theConst2)
                    + (__ldg(&doubleCoeffArray[yAbs]) * ySignD * theConst)
                    + (__ldg(&doubleCoeffArray[zAbs]) * zSignD);

                const double targetNeedle = matchesNeedle ? needle : negNeedle;
                if (!DOUBLE_BASICALLY_EQUAL(doubleValue, targetNeedle, doubleTol)) continue;

                // For needle: record as-is. For -needle: negate all coefficient indices
                const int neg = matchesNeedle ? 1 : -1;
                const int i = atomicAdd(doubleHitCount, 1);
                out[6*i] = neg * quintInd;
                out[(6*i) + 1] = neg * quartInd;
                out[(6*i) + 2] = neg * cubicInd;
                out[(6*i) + 3] = neg * x;
                out[(6*i) + 4] = neg * y;
                out[(6*i) + 5] = neg * z;
            }
        }
    }
}

std::vector<int*>* GpuQuinticFirstCheckerPositiveOnly::findHits(
            const double needle,
            const double theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges,
            long& floatHitCount
)
{
    // Updated loop boundaries to go from negative to positive ranges instead of starting from 6 (except quint starts at 0 for this positive-only variant)
    int loopStartEnds[12] = {0, 608'383, -152'231, 152'231, -6'087, 6'087, -2'203, 2'203, -555, 555, -143, 143};

    // TODO: This also sucks. Change this
    int coeffArraySize = 608'384;

    // 9/22/24 !NOTE! We ignore looprange starts on the Gpu (even numbered indices) and only care about ends (odds)
    //TODO: Use degree for way more things than just processing loopRanges
    // if loopRanges is non-null, find first level with positive values (-1 indicates use default) and use those
    // (WRONG) note that we ignore any level after that since we don't want to skip coeffs in later loops (WRONG)
    // note that we DO allow all levels to be updated now but warn the user that they may have an incomplete search
    if (loopRanges != NULL) {
        // loopRanges must have (2*(degree+1)) elements. Format is [zStart, zEnd, yStart, yEnd, ...]
        for (int loopRangeInd = 0; loopRangeInd < (2*(degree+1)); loopRangeInd++) {
            //TODO: Make this not so hacky and stupid
            if (loopRanges->at(loopRangeInd) < USE_DEFAULT) {
                if (loopRangeInd == 0 && loopRanges->at(loopRangeInd) < 0) {
                    // they are setting a negative quint start, so ignore it and warn them
                    std::cout << "WARNING: You have set a negative quint start. Unless your quint end has the same absolute value, your search may be incomplete. Ignoring negative quint start and using quint start of 0." << std::endl;
                    continue;
                }
                // they are setting a non-default value, so update loopStartEnds
                loopStartEnds[loopRangeInd] = loopRanges->at(loopRangeInd);
                std::cout << "WARNING: You have set a non-standard loop range. Your search may be incomplete" << std::endl;
            }
        }
    }
// HUGE TODO: Just trying to get to compile on 9/24/25 so changed parameters needle and theConst to double above but have not changed anything below this line to accomodate
// (except casting as float so GPU call would stay the same)
    

    float maxValue = 0, floatTol = FLOAT_POS_ERROR_DEFAULT;
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

    // Calculate actual ranges for grid sizing
    const int quintStart = loopStartEnds[0];
    const int quintEnd = loopStartEnds[1];
    const int quintRange = quintEnd - quintStart + 1;

    const int quartStart = loopStartEnds[2];
    const int quartEnd = loopStartEnds[3];
    const int quartRange = quartEnd - quartStart + 1;

    const int cubicStart = loopStartEnds[4];
    const int cubicEnd = loopStartEnds[5];
    const int cubicRange = cubicEnd - cubicStart + 1;
    std::vector<int*> *results = new std::vector<int*>();

    float *d_coeffArray;
    double *d_doubleCoeffArray;
    int *d_out;
    int *d_loopStartEnds;
    int *out = new int[coeffArraySize * 6];  // 6 spots for each result

    typedef std::numeric_limits< float > ldbl;
    cout.precision(ldbl::max_digits10);

    maxValue = std::max(
        1000.0f, // largest numerical coefficient
        (
            1000.0f * std::abs(theConst5f) +
                500.0f * std::abs(theConst4f) +
                100.0f * std::abs(theConst3f) +
                60.0f * std::abs(theConst2f) +
                30.0f * std::abs(theConstf) +
                15.0f
            ) // largest possible sum
    );

    floatTol = getFloatPrecisionBasedOnMaxValue(maxValue);
    doubleTol = getDoublePrecisionBasedOnMaxValue(maxValue);
    printf("Float tolerance: %10.10lf, Double tolerance: %10.20lf\n", floatTol, doubleTol);
    printf("Max value: %f\n", maxValue);

    // Calculate breakout bounds (matching CPU logic)
    // These values represent the largest coefficient per term
    const float uplim5 = 1000.0f;
    const float uplim4 = 500.0f;
    const float uplim3 = 100.0f;
    const float uplim2 = 60.0f;
    const float uplim1 = 30.0f;
    const float uplim0 = 15.0f;

    // These values represent the largest possible sum of each term plus those below them
    const float v0max = uplim0;
    const float v1max = uplim1 * theConstf + v0max;
    const float v2max = uplim2 * theConst2f + v1max;
    const float v3max = uplim3 * theConst3f + v2max;

    // Calculate breakout bounds (tolerance is set to abs(needlef), matching CPU)
    const float tolerance = std::abs(needlef);
    const float v1BreakoutHigh = needlef + v0max + tolerance;
    const float v1BreakoutLow = needlef - v0max - tolerance;
    const float v2BreakoutHigh = needlef + v1max + tolerance;
    const float v2BreakoutLow = needlef - v1max - tolerance;
    const float v3BreakoutHigh = needlef + v2max + tolerance;
    const float v3BreakoutLow = needlef - v2max - tolerance;

    // Also need breakout range to include -needle for positive-only search
    const float negNeedlef = -needlef;
    const float v1BreakoutHighNeg = negNeedlef + v0max + tolerance;
    const float v1BreakoutLowNeg = negNeedlef - v0max - tolerance;
    const float v2BreakoutHighNeg = negNeedlef + v1max + tolerance;
    const float v2BreakoutLowNeg = negNeedlef - v1max - tolerance;
    const float v3BreakoutHighNeg = negNeedlef + v2max + tolerance;
    const float v3BreakoutLowNeg = negNeedlef - v2max - tolerance;

    // Use union of both ranges so we don't miss -needle
    const float v1BH = std::max(v1BreakoutHigh, v1BreakoutHighNeg);
    const float v1BL = std::min(v1BreakoutLow, v1BreakoutLowNeg);
    const float v2BH = std::max(v2BreakoutHigh, v2BreakoutHighNeg);
    const float v2BL = std::min(v2BreakoutLow, v2BreakoutLowNeg);
    const float v3BH = std::max(v3BreakoutHigh, v3BreakoutHighNeg);
    const float v3BL = std::min(v3BreakoutLow, v3BreakoutLowNeg);

    printf("v3BreakoutHigh=%10.10lf, v3BreakoutLow=%10.10lf\n", v3BreakoutHigh, v3BreakoutLow);
    printf("v2BreakoutHigh=%10.10lf, v2BreakoutLow=%10.10lf\n", v2BreakoutHigh, v2BreakoutLow);
    printf("v1BreakoutHigh=%10.10lf, v1BreakoutLow=%10.10lf\n", v1BreakoutHigh, v1BreakoutLow);
    printf("v0max=%10.10lf\n", v0max);
    printf("v1max=%10.10lf\n", v1max);
    printf("v2max=%10.10lf\n", v2max);
    printf("v3max=%10.10lf\n", v3max);
    printf("tolerance=%10.10lf\n", tolerance);
    printf("needlef=%10.10lf\n", needlef);
    printf("theConstf=%10.10lf\n", theConstf);
    printf("theConst2f=%10.10lf\n", theConst2f);
    printf("(PositiveOnly uses union v3BH=%10.10lf, v3BL=%10.10lf for kernel)\n", v3BH, v3BL);

    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        cerr << "CUDA device initialization error: " << cudaGetErrorString(err) << " (code: " << err << ")" << endl;
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    // Get device properties to verify compatibility
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        cerr << "CUDA get device properties error: " << cudaGetErrorString(err) << " (code: " << err << ")" << endl;
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    int h_hitCount = 0;
    int h_doubleHitCount = 0;
    int *d_hitCount = nullptr;
    int *d_doubleHitCount = nullptr;

    // Allocate device memory with error checking
    err = cudaMalloc((void**)&d_hitCount, sizeof(int));
    if (err != cudaSuccess) {
        cerr << "CUDA malloc error (d_hitCount): " << cudaGetErrorString(err) << " (code: " << err << ")" << endl;
        cerr << "Device: " << prop.name << ", Compute Capability: " << prop.major << "." << prop.minor << endl;
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    err = cudaMalloc((void**)&d_doubleHitCount, sizeof(int));
    if (err != cudaSuccess) {
        cerr << "CUDA malloc error (d_doubleHitCount): " << cudaGetErrorString(err) << " (code: " << err << ")" << endl;
        cerr << "Device: " << prop.name << ", Compute Capability: " << prop.major << "." << prop.minor << endl;
        cudaFree(d_hitCount);
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    err = cudaMalloc((void**)&d_coeffArray, sizeof(float) * coeffArraySize);
    if (err != cudaSuccess) {
        cerr << "CUDA malloc error (d_coeffArray): " << cudaGetErrorString(err) << " (code: " << err << ")" << endl;
        cerr << "Device: " << prop.name << ", Compute Capability: " << prop.major << "." << prop.minor << endl;
        cudaFree(d_doubleHitCount);
        cudaFree(d_hitCount);
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    // Get double lookup table and allocate device memory
    const double* doubleCoeffArray = getLookupTableDouble();
    err = cudaMalloc((void**)&d_doubleCoeffArray, sizeof(double) * coeffArraySize);
    if (err != cudaSuccess) {
        cerr << "CUDA malloc error (d_doubleCoeffArray): " << cudaGetErrorString(err) << " (code: " << err << ")" << endl;
        cerr << "Device: " << prop.name << ", Compute Capability: " << prop.major << "." << prop.minor << endl;
        cudaFree(d_coeffArray);
        cudaFree(d_doubleHitCount);
        cudaFree(d_hitCount);
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    // need 6 spots for each hit
    err = cudaMalloc((void**)&d_out, sizeof(int) * coeffArraySize * 6);
    if (err != cudaSuccess) {
        cerr << "CUDA malloc error (d_out): " << cudaGetErrorString(err) << " (code: " << err << ")" << endl;
        cerr << "Device: " << prop.name << ", Compute Capability: " << prop.major << "." << prop.minor << endl;
        cudaFree(d_doubleCoeffArray);
        cudaFree(d_coeffArray);
        cudaFree(d_doubleHitCount);
        cudaFree(d_hitCount);
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    err = cudaMalloc((void**)&d_loopStartEnds, sizeof(int) * 12);
    if (err != cudaSuccess) {
        cerr << "CUDA malloc error (d_loopStartEnds): " << cudaGetErrorString(err) << endl;
        cudaFree(d_out);
        cudaFree(d_doubleCoeffArray);
        cudaFree(d_coeffArray);
        cudaFree(d_doubleHitCount);
        cudaFree(d_hitCount);
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    // Initialize output array
    for (int o = 0; o < coeffArraySize; o++) {
        out[o] = 0;
    }

    // Transfer data from host to device memory
    err = cudaMemcpy(d_hitCount, &h_hitCount, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "CUDA memcpy error (d_hitCount init): " << cudaGetErrorString(err) << endl;
        cudaFree(d_loopStartEnds);
        cudaFree(d_out);
        cudaFree(d_doubleCoeffArray);
        cudaFree(d_coeffArray);
        cudaFree(d_doubleHitCount);
        cudaFree(d_hitCount);
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    err = cudaMemcpy(d_doubleHitCount, &h_doubleHitCount, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "CUDA memcpy error (d_doubleHitCount init): " << cudaGetErrorString(err) << endl;
        cudaFree(d_loopStartEnds);
        cudaFree(d_out);
        cudaFree(d_doubleCoeffArray);
        cudaFree(d_coeffArray);
        cudaFree(d_doubleHitCount);
        cudaFree(d_hitCount);
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    err = cudaMemcpy(d_coeffArray, coeffArray, sizeof(float) * coeffArraySize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "CUDA memcpy error (d_coeffArray): " << cudaGetErrorString(err) << endl;
        cudaFree(d_loopStartEnds);
        cudaFree(d_out);
        cudaFree(d_doubleCoeffArray);
        cudaFree(d_coeffArray);
        cudaFree(d_doubleHitCount);
        cudaFree(d_hitCount);
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    err = cudaMemcpy(d_doubleCoeffArray, doubleCoeffArray, sizeof(double) * coeffArraySize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "CUDA memcpy error (d_doubleCoeffArray): " << cudaGetErrorString(err) << endl;
        cudaFree(d_loopStartEnds);
        cudaFree(d_out);
        cudaFree(d_doubleCoeffArray);
        cudaFree(d_coeffArray);
        cudaFree(d_doubleHitCount);
        cudaFree(d_hitCount);
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    err = cudaMemcpy(d_loopStartEnds, loopStartEnds, sizeof(int) * 12, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "CUDA memcpy error (d_loopStartEnds): " << cudaGetErrorString(err) << endl;
        cudaFree(d_loopStartEnds);
        cudaFree(d_out);
        cudaFree(d_doubleCoeffArray);
        cudaFree(d_coeffArray);
        cudaFree(d_doubleHitCount);
        cudaFree(d_hitCount);
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    // Smaller block (8,8,8)=512 to avoid "too many resources for launch" with block hit buffer + __ldg
    dim3 blocksizes(8, 8, 8);
    dim3 gridsizes(
        (quintRange + blocksizes.x - 1) / blocksizes.x,
        (quartRange + blocksizes.y - 1) / blocksizes.y,
        (cubicRange + blocksizes.z - 1) / blocksizes.z
    );
    cout << "GpuQuinticFirstCheckerPositiveOnly: Grid sizes: " << gridsizes.x << "," << gridsizes.y << "," << gridsizes.z << " (ranges: " << quintRange << "," << quartRange << "," << cubicRange << ")" << endl;

    // Create CUDA events for kernel timing
    cudaEvent_t kernelStart, kernelStop;
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);

    // Execute kernel with timing
    cudaEventRecord(kernelStart);
    compareToZeta5loopPositiveOnly<<<gridsizes, blocksizes>>>(d_out, theConst, needle, d_coeffArray, d_doubleCoeffArray, d_loopStartEnds, d_hitCount, d_doubleHitCount, theConst2, theConst3, theConst4, theConst5, floatTol, doubleTol, v3BH, v3BL, v2BH, v2BL, v1BH, v1BL);
    cudaEventRecord(kernelStop);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << endl;
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelStop);
        cudaFree(d_loopStartEnds);
        cudaFree(d_out);
        cudaFree(d_doubleCoeffArray);
        cudaFree(d_coeffArray);
        cudaFree(d_doubleHitCount);
        cudaFree(d_hitCount);
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    // Synchronize and check for runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cerr << "CUDA kernel execution error: " << cudaGetErrorString(err) << endl;
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelStop);
        cudaFree(d_loopStartEnds);
        cudaFree(d_out);
        cudaFree(d_doubleCoeffArray);
        cudaFree(d_coeffArray);
        cudaFree(d_doubleHitCount);
        cudaFree(d_hitCount);
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    // Calculate and print kernel execution time - TODO: Remove this later to see if it helps with speed?
    float kernelTimeMs = 0.0f;
    cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop);
    cout << "Kernel execution time: " << kernelTimeMs << " ms" << endl;
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);

    // Transfer data back to host memory
    err = cudaMemcpy(&h_hitCount, d_hitCount, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "CUDA memcpy error (hitCount): " << cudaGetErrorString(err) << endl;
        cudaFree(d_loopStartEnds);
        cudaFree(d_out);
        cudaFree(d_doubleCoeffArray);
        cudaFree(d_coeffArray);
        cudaFree(d_doubleHitCount);
        cudaFree(d_hitCount);
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    err = cudaMemcpy(&h_doubleHitCount, d_doubleHitCount, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "CUDA memcpy error (doubleHitCount): " << cudaGetErrorString(err) << endl;
        cudaFree(d_loopStartEnds);
        cudaFree(d_out);
        cudaFree(d_doubleCoeffArray);
        cudaFree(d_coeffArray);
        cudaFree(d_doubleHitCount);
        cudaFree(d_hitCount);
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    // need 6 spots for each hit
    err = cudaMemcpy(out, d_out, sizeof(int) * coeffArraySize * 6, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "CUDA memcpy error (out): " << cudaGetErrorString(err) << endl;
        cudaFree(d_loopStartEnds);
        cudaFree(d_out);
        cudaFree(d_doubleCoeffArray);
        cudaFree(d_coeffArray);
        cudaFree(d_doubleHitCount);
        cudaFree(d_hitCount);
        delete[] out;
        floatHitCount = 0;
        return results;
    }

    cout << "Float hits: " << h_hitCount << ", Double hits: " << h_doubleHitCount << endl;

    // Only process double-verified hits (h_doubleHitCount should equal number of results)
    for (int j = 0; j < h_doubleHitCount; j++) {
        //printHit(coeffArray, out[6*j], out[6*j+1], out[6*j+2], out[6*j+3], out[6*j+4], out[6*j+5]);
        results->push_back(new int[6] {out[6*j], out[6*j+1], out[6*j+2], out[6*j+3], out[6*j+4], out[6*j+5]});
    }

    // Deallocate device memory
    cudaFree(d_loopStartEnds);
    cudaFree(d_doubleCoeffArray);
    cudaFree(d_coeffArray);
    cudaFree(d_out);
    cudaFree(d_doubleHitCount);
    cudaFree(d_hitCount);

    delete[] out;

    // floatHitCount tracks all float matches, results only contains double-verified hits
    floatHitCount = h_hitCount;
    refineHitsWithFloat128Precision(results, needle, theConst);
    return results;
}