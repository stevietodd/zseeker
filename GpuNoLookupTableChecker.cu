#include "GpuPolynomialChecker.hpp"
#include "math.hpp"

// cuda.cu
#include "cudawrapper.hpp"
#include <cuda/std/algorithm>
#include <iostream>

// calculates val * [vector elems] and returns if close enough to needle
__global__ static void compareToNeedleLoop(int *out, const double theConst, const double needle, int *hitCount) {
	const float needlef = (float)needle;
	
	const double theConst2 = theConst * theConst;
	const double theConst3 = theConst2 * theConst;
	const double theConst4 = theConst3 * theConst;
	const double theConst5 = theConst4 * theConst;
	const float theConstf = (float)theConst;
	const float theConst2f = (float)theConst2;
	const float theConst3f = (float)theConst3;
	const float theConst4f = (float)theConst4;
	const float theConst5f = (float)theConst5;
	float floatTol = 0.0001 /*TODO: FIX!!*/, maxValue, v5, v4, v0;

	// our quint num/denom is encoded in one number as 1001*1024 + 1000, so we shift right to get num
	// and modulo 1024 (bitwise-and 1023) to get denom
	const float quintNumerator = (float)((blockIdx.x * blockDim.x + threadIdx.x) >> 10); 
	const float quintDenominator = (float)((blockIdx.x * blockDim.x + threadIdx.x) & 1023);
	const float quartNumerator = (float)(blockIdx.y * blockDim.y + threadIdx.y);
	const float quartDenominator = (float)(blockIdx.z * blockDim.z + threadIdx.z);

	// breakout
	if (quintNumerator > 1000.0f || quintDenominator == 0.0f || quintDenominator > 1000.0f ||
		quartNumerator > 500.0f || quartDenominator == 0.0f || quartDenominator > 500.0f) {
			//TODO: Also do gcd checks here???
		return;
	}

	v5 = theConst5f * quintNumerator / quintDenominator;
		//TODO: ADD BACK! maxValue = std::max({/*maxLowerDegreesValue, */std::abs(v5), quintNumerator});
		// floatTol = getFloatPrecisionBasedOnMaxValue(maxValue);
		// doubleTol = getDoublePrecisionBasedOnMaxValue(maxValue);
	v4 = v5 + (theConst4f * quartNumerator / quartDenominator);

	// for (float j = 0.0f; j <= 100.0f; j++) {
	// 	for (float k = 1.0f; k <= 100.0f; k++) {
	// 		for (float m = 0.0f; m <= 60.0f; m++) {
	// 			for (float n = 1.0f; n <= 60.0f; n++) {
	// 				for (float p = 0.0f; p <= 30.0f; p++) {
	// 					for (float q = 1.0f; q <= 30.0f; q++) {
	// 						for (float r = 0.0f; r <= 15.0f; r++) {
	// 							for (float s = 1.0f; s <= 15.0f; s++) {
	// 								v0 = v4 + (theConst3f * j / k) + (theConst2f * m / n) + 
	// 									(theConstf * p / q) + (r / s);
	for (int j = 0; j <= 100; j++) {
		for (int k = 1; k <= 100; k++) {
			for (int m = 0; m <= 60; m++) {
				for (int n = 1; n <= 60; n++) {
					for (int p = 0; p <= 30; p++) {
						for (int q = 1; q <= 30; q++) {
							for (int r = 0; r <= 15; r++) {
								for (int s = 1; s <= 15; s++) {
									v0 = v4 + (theConst3f * j / k) + (theConst2f * m / n) + 
										(theConstf * p / q) + (r / s);
										if (r == 1) {return;}
									if (FLOAT_BASICALLY_EQUAL(v0, needlef, floatTol)) {
										// TODO: Increment counter of float hits
										(*hitCount)++; // TODO: Add this backfloatHitCount++;
										//printf("floatHitCount=%ld\n", floatHitCount);
										//printf("double first two here is %10.10lf and %10.10lf\n", doubleLUT[u], doubleLUT[v]);
										//printHit(LUT.data(), u,v,w,x,y,z);
								
										// since our float was in range, calculate the double value and check for a "real hit"
										// doubleValue = ((u < 0) ? -doubleLUT[-u] : doubleLUT[u]) * theConst5
										// 	+ ((v < 0) ? -doubleLUT[-v] : doubleLUT[v]) * theConst4
										// 	+ ((w < 0) ? -doubleLUT[-w] : doubleLUT[w]) * theConst3
										// 	+ ((x < 0) ? -doubleLUT[-x] : doubleLUT[x]) * theConst2
										// 	+ ((y < 0) ? -doubleLUT[-y] : doubleLUT[y]) * (double)theConst
										// 	+ ((z < 0) ? -doubleLUT[-z] : doubleLUT[z]);
										// if (DOUBLE_BASICALLY_EQUAL(doubleValue, needle, doubleTol)) {
										// 	// TODO: These are the real hits!
										// 	hit = new int[6] {u, v, w, x, y, z};
										// 	hits->push_back(hit);
										// 	printf("Real hit!\n");
										}
								}
							}
						}
					}
				}
			}
		}
	}
	
	
}

std::vector<int*>* GpuNoLookupTableChecker::findHits(
            const double needle,
            const double theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges,
            long& floatHitCount
)
{
    // Updated loop boundaries to match checkz3constantswithz5usingLUTandCPU::main
    // note that these are now negative to positive ranges instead of starting from 6
    int loopStartEnds[12] = {-608'383, 608'383, -152'231, 152'231, -6'087, 6'087, -2'203, 2'203, -555, 555, -143, 143};

	//TODO: Pass in needle instead of hardcoded use of z5 here!
	const float z5 = 1.036927755143369926331365486457034168L; //riemann_zetal((long double)5);
	float currentQuart;
	float quarticSum;
	const float theConst2 = powl(theConst, (float)2);
	const float theConst3 = powl(theConst, (float)3);
	const float theConst4 = powl(theConst, (float)4);
	const float theConst5 = powl(theConst, (float)5);
	const int quintLastIndex = 1'216'772;
	const int quartLastIndex = 304'468;
	const int cubicLastIndex = 12'180;
	std::vector<int*> *results = new std::vector<int*>();

	int *d_out;
	int *out = new int[quintLastIndex];

	typedef std::numeric_limits< float > ldbl;
	std::cout.precision(ldbl::max_digits10);

	int h_hitCount = 0;
	int *d_hitCount = 0;
	cudaMalloc((void**) &d_hitCount, sizeof(int) );
	cudaMemcpy(d_hitCount, &h_hitCount , sizeof(int), cudaMemcpyHostToDevice);

	// initialize output array
	for (int o = 0; o < quintLastIndex; o++) {
		out[o] = 0;
	}

	// Allocate device memory 
    cudaMalloc((void**)&d_out, sizeof(int) * quintLastIndex);		//TODO: Can probably make this 10% as large as quintLastIndex??

	std::cout << "dog\n";

	// Transfer data from host to device memory
    // cudaMemcpy(d_coeffArray, coeffArray, sizeof(float) * quintLastIndex, cudaMemcpyHostToDevice);

	std::cout << "cat\n";

	//int block_size = 1024;
	// this block size setup may seem arbitrary, but the most important part is y = 8 because max grid (not block) index
	// is 65,535 and we need 65,535 * block.y > 304,468
	dim3 blocksizes(16, 8, 8);

    //int grid_size = ((quintLastIndex + block_size) / block_size);
	// we add 1 to each to make sure we actually do all of the work
	//dim3 gridsizes((quintLastIndex / blocksizes.x) + 1, (quartLastIndex / blocksizes.y) + 1, (cubicLastIndex / blocksizes.z) + 1);
	// CONVOLUTED 
	// dim3 gridsizes((1001*1000*501 / blocksizes.x) + 1, (500*101 / blocksizes.y) + 1, (100*61 / blocksizes.z) + 1);
	// SIMPLE (except we use 1024 instead of 1000 for bit shiftability)
	//dim3 gridsizes(((1001*1024 + 1000) / blocksizes.x) + 1, (501 / blocksizes.y) + 1, (500 / blocksizes.z) + 1);
	dim3 gridsizes(1, 1, 1);
	//cout << gridsizes << "\n";

	// Execute kernel
	compareToNeedleLoop<<<gridsizes, blocksizes>>>(d_out, theConst, needle, d_hitCount);
std::cout << cudaPeekAtLastError() << std::endl;
	// Transfer data back to host memory
	cudaMemcpy(&h_hitCount , d_hitCount, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(out, d_out, sizeof(int) * quintLastIndex, cudaMemcpyDeviceToHost);

	std::cout << h_hitCount << std::endl;
	std::cout << "YOOOO!" << std::endl;

	// for (int j = 0; j < h_hitCount; j++) {
	// 	// TODO: Update these lines! Second param in printHit and first array value in pushback was i but I haven't figured out how to tie those together now
	// 	// TODO: Also this should be using the out variable, not just the j-indices duh
	// 	printHit(j, j, cubicSum, coeffArray);
	// 	results->push_back(new int[2] {j, j});
	// }

	//cout << i << "\n";
    
	// cout << "bird\n";

    // Deallocate device memory
    cudaFree(d_out);
	cudaFree(d_hitCount);

	std::cout << "lizard\n";

	delete out;

	floatHitCount = results->size();
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

    // const float theConst2 = powl(theConst, (float)2);
	// const float theConst3 = powl(theConst, (float)3);
	// const float theConst4 = powl(theConst, (float)4);
	// const float theConst5 = powl(theConst, (float)5);

    // float v0, v1, v2, v3, v4, v5, *hit;

    // std::vector<int*> *hits = new std::vector<int*>();

    // // note that these loops use <= (less than or EQUAL TO)
    // for (int z = loopStartEnds[0]; z <= loopStartEnds[1]; z++) {
	// 	v0 = coeffArray[z];
	
	// 	for (int y = loopStartEnds[2]; y <= loopStartEnds[3]; y++) {
	// 		v1 = v0 + coeffArray[y] * theConst;

	// 		for (int x = loopStartEnds[4]; x <= loopStartEnds[5]; x++) {
	// 			v2 = v1 + coeffArray[x] * theConst2;

	// 			for (int w = loopStartEnds[6]; w <= loopStartEnds[7]; w++) {
	// 				printf("dog\n");
	// 				v3 = v2 + coeffArray[w] * theConst3;
	// 				//TODO: Shouldn't overwrite this every time. Also need to take the 2 results returned
	// 				// and make a new "hit" with all 6 coeff indices to return
    //                 hits = testForZeta5OnGPU(theConst, v3, coeffArray, loopStartEnds[9], loopStartEnds[11]);
    //             }
    //         }
    //     }
    // }

    // return hits;
}