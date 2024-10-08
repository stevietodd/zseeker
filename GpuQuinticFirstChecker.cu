#include "GpuPolynomialChecker.hpp"
#include "math.hpp"

// cuda.cu
#include "cudawrapper.hpp"
#include <iostream>
using namespace std;

// calculates val * [vector elems] and returns if close enough to needle
template<typename T>
__global__ static void compareToZeta5loop(T *out, const T theConst, const T needle, const T *coeffArray, int n, int *hitCount) {
	const float theConst2 = pow(theConst, (float)2);
	float v0, v1, v2;
	const int quintInd = blockIdx.x * blockDim.x + threadIdx.x;
	const int quartInd = blockIdx.y * blockDim.y + threadIdx.y;
	const int cubicInd = blockIdx.z * blockDim.z + threadIdx.z;

	// breakout
	if (quintInd >= 1'216'772 || quartInd >= 304'468 || cubicInd >= 12'180) {
		return;
	}

	const float topThreeTerms = (coeffArray[quintInd] * pow(theConst, (float)5))
		+ (coeffArray[quartInd] * pow(theConst, (float)4))
		+ (coeffArray[cubicInd] * pow(theConst, (float)3));
    //T expr;
	//register int i;
	// if (quartInd % 60000 == 0 && cubicInd % 60000 == 0) {
	// printf("Quint = %d, quart = %d, cubic = %d. n=%d\n", quintInd, quartInd, cubicInd, n);
	// }

    // Handling arbitrary vector size
	// // note that these loops use <= (less than or EQUAL TO)
	for (int z = 6; z <= 292; z++) {
	//for (int z = loopStartEnds[0]; z <= loopStartEnds[1]; z++) {
		v0 = coeffArray[z];
		//printf("%d,", z);
	
		for (int y = 6; y <= 1'116; y++) {
		//for (int y = loopStartEnds[2]; y <= loopStartEnds[3]; y++) {
			v1 = v0 + coeffArray[y] * theConst;

			for (int x = 6; x <= 4'412; x++) {
			//for (int x = loopStartEnds[4]; x <= loopStartEnds[5]; x++) {
				v2 = v1 + coeffArray[x] * theConst2;

				if (FLOAT_BASICALLY_EQUAL((topThreeTerms + v2), needle)) {
					printf("(%d,%d,%d,%d,%d,%d): %10.10lf*c^5 + %10.10lf*c^4 + %10.10lf*c^3 + %10.10lf*c^2 + %10.10lf*c + %10.10lf = HIT!\n",
						quintInd, quartInd, cubicInd, x, y, z, coeffArray[quintInd], coeffArray[quartInd],
						coeffArray[cubicInd], coeffArray[x], coeffArray[y], coeffArray[z]);
					// i = atomicAdd(hitCount, 1);
					// out[i] = tid; //TODO: MAKE THIS ATOMIC AND DYNAMIC instead of only populating "matching" coeffs in array [0, 0, HIT, 0, 0, 0, HIT, etc.]		
				}
			}
		}
	}
}

std::vector<int*>* GpuQuinticFirstChecker::findHits(
            const float needle,
            const float theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges
)
{
    // TODO: This sucks. Change this
    // note that even elements are LUT[0] through LUT[5]
    int loopStartEnds[12] = {6, 1'216'772, 6, 304'468, 6, 12'180, 6, 4'412, 6, 1'116, 6, 292};

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

	float *d_coeffArray, *d_out;
	int *out = new int[quintLastIndex];

	typedef std::numeric_limits< float > ldbl;
	cout.precision(ldbl::max_digits10);

	int h_hitCount = 0;
	int *d_hitCount = 0;
	cudaMalloc((void**) &d_hitCount, sizeof(int) );
	cudaMemcpy(d_hitCount, &h_hitCount , sizeof(int), cudaMemcpyHostToDevice);

	// initialize output array
	for (int o = 0; o < quintLastIndex; o++) {
		out[o] = 0;
	}

	// Allocate device memory 
    cudaMalloc((void**)&d_coeffArray, sizeof(float) * quintLastIndex);
    cudaMalloc((void**)&d_out, sizeof(int) * quintLastIndex);		//TODO: Can probably make this 10% as large as quintLastIndex??

	cout << "dog\n";

	// Transfer data from host to device memory
    cudaMemcpy(d_coeffArray, coeffArray, sizeof(float) * quintLastIndex, cudaMemcpyHostToDevice);

	cout << "cat\n";

	//int block_size = 1024;
	// this block size setup may seem arbitrary, but the most important part is y = 8 because max grid (not block) index
	// is 65,535 and we need 65,535 * block.y > 304,468
	dim3 blocksizes(16, 8, 8);

    //int grid_size = ((quintLastIndex + block_size) / block_size);
	// we add 1 to each to make sure we actually do all of the work
	dim3 gridsizes((quintLastIndex / blocksizes.x) + 1, (quartLastIndex / blocksizes.y) + 1, (cubicLastIndex / blocksizes.z) + 1);
	//cout << gridsizes << "\n";

	// Execute kernel
	compareToZeta5loop<<<gridsizes, blocksizes>>>(d_out, theConst, z5, d_coeffArray, quintLastIndex, d_hitCount);

	// Transfer data back to host memory
	cudaMemcpy(&h_hitCount , d_hitCount, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(out, d_out, sizeof(int) * quintLastIndex, cudaMemcpyDeviceToHost);

	cout << h_hitCount << endl;
	cout << "YOOOO!" << endl;

	// for (int j = 0; j < h_hitCount; j++) {
	// 	// TODO: Update these lines! Second param in printHit and first array value in pushback was i but I haven't figured out how to tie those together now
	// 	// TODO: Also this should be using the out variable, not just the j-indices duh
	// 	printHit(j, j, cubicSum, coeffArray);
	// 	results->push_back(new int[2] {j, j});
	// }

	//cout << i << "\n";
    
	// cout << "bird\n";

    // Deallocate device memory
    cudaFree(d_coeffArray);
    cudaFree(d_out);
	cudaFree(d_hitCount);

	cout << "lizard\n";

	delete out;

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