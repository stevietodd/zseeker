#include "GpuPolynomialChecker.hpp"

// cuda.cu
#include "cudawrapper.hpp"
#include <iostream>
using namespace std;

// calculates val * [vector elems] and returns if close enough to needle
template<typename T>
__global__ static void compareToZeta5(T *out, const T val, const T needle, const T *coeffArray, int n, int *hitCount) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    T expr;
	int i;
	//printf("Hello from block %d, thread %d => tid %d. n=%d\n", blockIdx.x, threadIdx.x, tid, n);

    // Handling arbitrary vector size
    if (tid < n){
		//printf("coeff=%f\n", coeffArray[tid]);
		expr = (val * coeffArray[tid]) - needle;
		if (expr < .0000001 && expr > -.0000001) {
			printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
			i = atomicAdd(hitCount, 1);
			out[i] = coeffArray[tid]; //TODO: MAKE THIS ATOMIC AND DYNAMIC instead of only populating "matching" coeffs in array [0, 0, HIT, 0, 0, 0, HIT, etc.]
		}
	}
}

bool InitCUDA(bool b) {
    /* CUDA Initialization */
}

template<typename T>
void printHit(int i5, int i4, const T cubicSum, const T *coeffArray)
{
	cout << "(" << i5 << "," << i4 << ",?,?,?,?): " <<
		coeffArray[i5] << "c^5 + " << coeffArray[i4] << "c^4 + " << cubicSum << "= HIT!\n";
}

// will calculate ax^5 + bx^4 + cx^3 + dx^2 + ex + f - zeta(5) and return x if
// the value is within a "close enough" bound to zero
// cons = x
// cubicSum = cx^3 + dx^2 + ex + f (pre-calculated)
// coeffArray = array of all possible a/b/c/d/e/f coeff values
// quartLastIndex = last index to loop through for b values
// quintLastIndex = last index to loop through for a values
std::vector<float*>* testForZeta5OnGPU(float cons, float cubicSum, const float *coeffArray, int quartLastIndex, int quintLastIndex)
{
	const float z5 = 1.036927755143369926331365486457034168L; //riemann_zetal((long double)5);
	float currentQuart;
	float quarticSum;
	float consFourth = pow(cons, (float)4);
	float consFifth = consFourth * cons;
	std::vector<float*> *results = new std::vector<float*>();

	float *d_coeffArray, *d_out;
	float *out = new float[quintLastIndex];

	int *hitCount;
	cudaMallocHost(&hitCount, sizeof(int));
    memset(hitCount, 0, sizeof(int));

	// initialize output array
	for (int o = 0; o < quintLastIndex; o++) {
		out[o] = 0;
	}

	// Allocate device memory 
    cudaMalloc((void**)&d_coeffArray, sizeof(float) * quintLastIndex);
    cudaMalloc((void**)&d_out, sizeof(float) * quintLastIndex);		//TODO: Can probably make this 10% as large as quintLastIndex??

	cout << "dog\n";

	// Transfer data from host to device memory
    cudaMemcpy(d_coeffArray, coeffArray, sizeof(float) * quintLastIndex, cudaMemcpyHostToDevice);

	cout << "cat\n";

	int block_size = 1024;
    int grid_size = ((quintLastIndex + block_size) / block_size);
	cout << grid_size << "\n";

	// loop through quarts
	for (int i = 6; i <= quartLastIndex; i++) {
		currentQuart = coeffArray[i];

		quarticSum = cubicSum + currentQuart * consFourth;

		// now we want gpu to calculate ax^5 + quarticSum - zeta(5) and return if close enough to zero
		// or ax^5 + quarticSum == zeta(5) => zeta(5) - quarticSum is close enough to ax^5
		// in other words, figure out what blah - zeta(5) and cons^5 and send those to GPU along with coeff array
		
		// Executing kernel
		compareToZeta5<<<grid_size,block_size>>>(d_out, consFifth, (z5 - quarticSum), d_coeffArray, quintLastIndex, hitCount);
	}

	cout << *hitCount << endl;

	// Transfer data back to host memory
	cudaMemcpy(out, d_out, sizeof(float) * quintLastIndex, cudaMemcpyDeviceToHost);
	//cout << "ferret\n";
	// //cudaDeviceSynchronize();

	// for (int j = 0; j < quintLastIndex; j++) {
	// 	if (out[j] != 0) {
	// 		printHit(j, i, cubicSum, coeffArray);
	// 		results->push_back(new float[2] {coeffArray[i], coeffArray[j]});
	// 	}
	// }

	//cout << i << "\n";
    
	// cout << "bird\n";

    // Deallocate device memory
    cudaFree(d_coeffArray);
    cudaFree(d_out);

	cout << "lizard\n";

	delete out;

	return results;
}

std::vector<float*>* GpuPolynomialChecker::findHits(
            const float needle,
            const float theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges
)
{
    // TODO: This sucks. Change this
    // note that even elements are LUT[0] through LUT[5]
    int loopStartEnds[12] = {6, 292, 6, 1'116, 6, 4'412, 6, 12'180, 6, 304'468, 6, 1'216'772};

    //TODO: Use degree for way more things than just processing loopRanges
    // if loopRanges is non-null, find first level with positive values (-1 indicates use default) and use those
    // note that we ignore any level after that since we don't want to skip coeffs in later loops
    if (loopRanges != NULL) {
        // loopRanges must have (2*(degree+1)) elements. Format is [zStart, zEnd, yStart, yEnd, ...]
        for (int loopRangeInd = 0; loopRangeInd < (2*(degree+1)); loopRangeInd++) {
            //TODO: Make this not so hacky and stupid
            if (loopRanges->at(loopRangeInd) >= 0) {
                // they are setting a non-default value, so update loopStartEnds
                loopStartEnds[loopRangeInd] = loopRanges->at(loopRangeInd);
                std::cout << "WARNING: You have set a non-standard loop range. Your search may be incomplete" << std::endl;
            }
        }
    }

    const float theConst2 = powl(theConst, (float)2);
	const float theConst3 = powl(theConst, (float)3);
	const float theConst4 = powl(theConst, (float)4);
	const float theConst5 = powl(theConst, (float)5);

    float v0, v1, v2, v3, v4, v5, *hit;

    std::vector<float*> *hits = new std::vector<float*>();

    // note that these loops use <= (less than or EQUAL TO)
    for (int z = loopStartEnds[0]; z <= loopStartEnds[1]; z++) {
		v0 = coeffArray[z];
	
		for (int y = loopStartEnds[2]; y <= loopStartEnds[3]; y++) {
			v1 = v0 + coeffArray[y] * theConst;

			for (int x = loopStartEnds[4]; x <= loopStartEnds[5]; x++) {
				v2 = v1 + coeffArray[x] * theConst2;

				for (int w = loopStartEnds[6]; w <= loopStartEnds[7]; w++) {
					v3 = v2 + coeffArray[w] * theConst3;
                        hits = testForZeta5OnGPU(theConst, v3, coeffArray, loopStartEnds[9], loopStartEnds[11]);
                }
            }
        }
    }

    return hits;
}