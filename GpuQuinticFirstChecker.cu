#include "GpuPolynomialChecker.hpp"
#include "math.hpp"
#include "lookupTableAccessor.hpp"

// cuda.cu
#include "cudawrapper.hpp"
#include <iostream>
using namespace std;

// calculates val * [vector elems] and returns if close enough to needle
__global__ static void compareToZeta5loop(int *out, const double theConst, const double needle, const float *coeffArray, const double *doubleCoeffArray, const int *loopStartEnds, int *floatHitCount, int *doubleHitCount, const double theConst2, const double theConst3, const double theConst4, const double theConst5, const float floatTol, const double doubleTol, const float v3BreakoutHigh, const float v3BreakoutLow, const float v2BreakoutHigh, const float v2BreakoutLow, const float v1BreakoutHigh, const float v1BreakoutLow) {
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

	// Get thread indices
	const int quintThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int quartThreadIdx = blockIdx.y * blockDim.y + threadIdx.y;
	const int cubicThreadIdx = blockIdx.z * blockDim.z + threadIdx.z;

	// Calculate ranges from shared memory (faster than global memory)
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

	// Calculate top three terms using precomputed powers (much faster than pow())
	const float topThreeTerms =(coeffArray[quintAbs] * ((quintInd < 0) ? -1.0f : 1.0f) * theConst5f)
		+ (coeffArray[quartAbs] * ((quartInd < 0) ? -1.0f : 1.0f) * theConst4f)
		+ (coeffArray[cubicAbs] * ((cubicInd < 0) ? -1.0f : 1.0f) * theConst3f);

	// Check v3 breakout
	if (topThreeTerms < v3BreakoutLow || topThreeTerms > v3BreakoutHigh) {
		return; // can't possibly get back to needle
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
		// Calculate v2 = topThreeTerms + quadratic term
		v2 = topThreeTerms + (coeffArray[xAbs] * ((x < 0) ? -1.0f : 1.0f) * theConst2f);
		
		// Check v2 breakout
		if (v2 < v2BreakoutLow || v2 > v2BreakoutHigh) {
			continue; // can't possibly get back to needle, skip to next x
		}

		for (int y = yStart; y <= yEnd; y++) {
			const int yAbs = (y < 0) ? -y : y;
			// Calculate v1 = v2 + linear term
			v1 = v2 + (coeffArray[yAbs] * ((y < 0) ? -1.0f : 1.0f) * theConstf);
			
			// Check v1 breakout
			if (v1 < v1BreakoutLow || v1 > v1BreakoutHigh) {
				continue; // can't possibly get back to needle, skip to next y
			}

			for (int z = zStart; z <= zEnd; z++) {
				const int zAbs = (z < 0) ? -z : z;
				// Calculate v0 = v1 + constant term (final sum)
				v0 = v1 + (coeffArray[zAbs] * ((z < 0) ? -1.0f : 1.0f));

				// First check with float precision (v0 is the final sum)
				if (FLOAT_BASICALLY_EQUAL(v0, needlef, floatTol)) {
					//atomicAdd(floatHitCount, 1); TODO: MegaMan removed 12/13/25 for testing if this speeds up, ADD BACK MAYBE!?

					// Calculate double precision value for verification (using precomputed powers)
					const double doubleValue = 
						(doubleCoeffArray[quintAbs] * ((quintInd < 0) ? -1.0f : 1.0f) * theConst5)	
						+ (doubleCoeffArray[quartAbs] * ((quartInd < 0) ? -1.0f : 1.0f) * theConst4)
						+ (doubleCoeffArray[cubicAbs] * ((cubicInd < 0) ? -1.0f : 1.0f) * theConst3)
						+ (doubleCoeffArray[xAbs] * ((x < 0) ? -1.0f : 1.0f) * theConst2)
						+ (doubleCoeffArray[yAbs] * ((y < 0) ? -1.0f : 1.0f) * theConst)
						+ (doubleCoeffArray[zAbs] * ((z < 0) ? -1.0f : 1.0f));

					// Check with double precision - only record if this passes
					if (DOUBLE_BASICALLY_EQUAL(doubleValue, needle, doubleTol)) {
						const int i = atomicAdd(doubleHitCount, 1);
						out[6*i] = quintInd;
						out[(6*i) + 1] = quartInd;
						out[(6*i) + 2] = cubicInd;
						out[(6*i) + 3] = x;
						out[(6*i) + 4] = y;
						out[(6*i) + 5] = z;
					}
				}
			}
		}
	}
}

std::vector<int*>* GpuQuinticFirstChecker::findHits(
            const double needle,
            const double theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges,
            long& floatHitCount
)
{
    // Updated loop boundaries to go from negative to positive ranges instead of starting from 6
    int loopStartEnds[12] = {-608'383, 608'383, -152'231, 152'231, -6'087, 6'087, -2'203, 2'203, -555, 555, -143, 143};

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

	// Calculate breakout bounds (tolerance is set to needlef, matching CPU)
	const float tolerance = std::abs(needlef);
	const float v1BreakoutHigh = needlef + v0max + tolerance;
	const float v1BreakoutLow = needlef - v0max - tolerance;
	const float v2BreakoutHigh = needlef + v1max + tolerance;
	const float v2BreakoutLow = needlef - v1max - tolerance;
	const float v3BreakoutHigh = needlef + v2max + tolerance;
	const float v3BreakoutLow = needlef - v2max - tolerance;
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

	// Initialize CUDA device (ensures proper initialization before API calls)
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
		cerr << "CUDA malloc error (d_hitCount): " << cudaGetErrorString(err) << " (error code: " << err << ")" << endl;
		cerr << "Device: " << prop.name << ", Compute Capability: " << prop.major << "." << prop.minor << endl;
		delete[] out;
		floatHitCount = 0;
		return results;
	}
	
	err = cudaMalloc((void**)&d_doubleHitCount, sizeof(int));
	if (err != cudaSuccess) {
		cerr << "CUDA malloc error (d_doubleHitCount): " << cudaGetErrorString(err) << endl;
		cudaFree(d_hitCount);
		delete[] out;
		floatHitCount = 0;
		return results;
	}
	
	err = cudaMalloc((void**)&d_coeffArray, sizeof(float) * coeffArraySize);
	if (err != cudaSuccess) {
		cerr << "CUDA malloc error (d_coeffArray): " << cudaGetErrorString(err) << endl;
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
		cerr << "CUDA malloc error (d_doubleCoeffArray): " << cudaGetErrorString(err) << endl;
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
		cerr << "CUDA malloc error (d_out): " << cudaGetErrorString(err) << endl;
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

	cout << "cat\n";

	// Optimized block size: (8, 8, 16) = 1024 threads
	// This reduces register pressure compared to (16, 8, 8) while maintaining same thread count
	// Smaller x dimension can help with memory coalescing patterns
	dim3 blocksizes(8, 8, 16);
	dim3 gridsizes(
		(quintRange + blocksizes.x - 1) / blocksizes.x,
		(quartRange + blocksizes.y - 1) / blocksizes.y,
		(cubicRange + blocksizes.z - 1) / blocksizes.z
	);
	cout << "Grid sizes: " << gridsizes.x << "," << gridsizes.y << "," << gridsizes.z << " (ranges: " << quintRange << "," << quartRange << "," << cubicRange << ")" << endl;

	// Create CUDA events for kernel timing
	cudaEvent_t kernelStart, kernelStop;
	cudaEventCreate(&kernelStart);
	cudaEventCreate(&kernelStop);
	
	// Execute kernel with timing
	cudaEventRecord(kernelStart);
	compareToZeta5loop<<<gridsizes, blocksizes>>>(d_out, theConst, needle, d_coeffArray, d_doubleCoeffArray, d_loopStartEnds, d_hitCount, d_doubleHitCount, theConst2, theConst3, theConst4, theConst5, floatTol, doubleTol, v3BreakoutHigh, v3BreakoutLow, v2BreakoutHigh, v2BreakoutLow, v1BreakoutHigh, v1BreakoutLow);
	cudaEventRecord(kernelStop);
	
	// Check for kernel launch errors
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << endl;
		cudaEventDestroy(kernelStart);
		cudaEventDestroy(kernelStop);
		// Cleanup and return
		cudaFree(d_loopStartEnds);
		cudaFree(d_doubleCoeffArray);
		cudaFree(d_coeffArray);
		cudaFree(d_out);
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
		// Cleanup and return
		cudaFree(d_loopStartEnds);
		cudaFree(d_doubleCoeffArray);
		cudaFree(d_coeffArray);
		cudaFree(d_out);
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
		cudaFree(d_doubleCoeffArray);
		cudaFree(d_coeffArray);
		cudaFree(d_out);
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
		cudaFree(d_doubleCoeffArray);
		cudaFree(d_coeffArray);
		cudaFree(d_out);
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
		cudaFree(d_doubleCoeffArray);
		cudaFree(d_coeffArray);
		cudaFree(d_out);
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

	cout << "lizard\n";

	delete[] out;

	// floatHitCount tracks all float matches, results only contains double-verified hits
	floatHitCount = h_hitCount;
	return results;


    // //TODO: Use degree for way more things than just processing loopRanges
    // // if loopRanges is non-null, find first level with positive values (-1 indicates use default) and use those
    // // note that we ignore any level after that since we don't want to skip coeffs in later loops
    // if (loopRanges != NULL) {
    //     // loopRanges must have (2*(degree+1)) elements. Format is [zStart, zEnd, yStart, yEnd, ...]
    //     for (int loopRangeInd = 0; loopRangeInd < (2*(degree+1)); loopRangeInd++) {
    //         //TODO: Make this not so hacky and stupid
    //         if (loopRanges->at(loopRangeInd) >= 0) {
    //             // they are setting a non-default value, so update loopStartEnds
    //             loopStartEnds[loopRangeInd] = loopRanges->at(loopRangeInd);
    //             std::cout << "WARNING: You have set a non-standard loop range. Your search may be incomplete" << std::endl;
    //         }
    //     }
    // }
}