#include "GpuPolynomialChecker.hpp"
#include "math.hpp"

// cuda.cu
#include "cudawrapper.hpp"
#include <iostream>
using namespace std;

// calculates val * [vector elems] and returns if close enough to needle
template<typename T>
__global__ static void compareToZeta5(T *out, const T val, const T needle, const T *coeffArray, int n, int *hitCount) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    T expr;
	register int i;
	//printf("Hello from block %d, thread %d => tid %d. n=%d\n", blockIdx.x, threadIdx.x, tid, n);

    // Handling arbitrary vector size
    if (tid < n){
		if (FLOAT_BASICALLY_EQUAL(coeffArray[tid] * val, needle)) {
			// printf("LUT[this]=%10.10lf,theConst5=%10.10lf,needle=?,v4=?,(needle-v4)=%10.10lf\n", coeffArray[tid], val, needle);
			// printf(
			// 	"Hit found in block %d, thread %d: %f + coeffArray[%d]*c^5 (%f) within %f\n",
			// 	blockIdx.x,
			// 	threadIdx.x,
			// 	(M_PI-needle),
			// 	tid,
			// 	coeffArray[tid],
			// 	expr
			// );
			i = atomicAdd(hitCount, 1);
			out[i] = tid; //TODO: MAKE THIS ATOMIC AND DYNAMIC instead of only populating "matching" coeffs in array [0, 0, HIT, 0, 0, 0, HIT, etc.]
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
std::vector<int*>* testForZeta5OnGPU(float cons, float cubicSum, const float *coeffArray, int quartLastIndex, int quintLastIndex)
{
	//TODO: Pass in needle instead of hardcoded use of z5 here!
	const float z5 = 1.036927755143369926331365486457034168L; //riemann_zetal((long double)5);
	float currentQuart;
	float quarticSum;
	float consFourth = pow(cons, (float)4);
	float consFifth = pow(cons, (float)5);
	std::vector<int*> *results = new std::vector<int*>();

	float *d_coeffArray, *d_out;
	int *out = new int[quintLastIndex];

	typedef std::numeric_limits< float > ldbl;
	cout.precision(ldbl::max_digits10);
cout << cons << "," << consFourth << "," << consFifth << endl;
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
		compareToZeta5<<<grid_size,block_size>>>(d_out, consFifth, (z5 - quarticSum), d_coeffArray, quintLastIndex, d_hitCount);
	}

	// Transfer data back to host memory
	cudaMemcpy(&h_hitCount , d_hitCount, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(out, d_out, sizeof(int) * quintLastIndex, cudaMemcpyDeviceToHost);
	//cout << "ferret\n";
	// //cudaDeviceSynchronize();

	cout << h_hitCount << endl;

	for (int j = 0; j < h_hitCount; j++) {
		// TODO: Update these lines! Second param in printHit and first array value in pushback was i but I haven't figured out how to tie those together now
		// TODO: Also this should be using the out variable, not just the j-indices duh
		printHit(j, j, cubicSum, coeffArray);
		results->push_back(new int[2] {j, j});
	}

	//cout << i << "\n";
    
	// cout << "bird\n";

    // Deallocate device memory
    cudaFree(d_coeffArray);
    cudaFree(d_out);
	cudaFree(d_hitCount);

	cout << "lizard\n";

	delete out;

	return results;
}

std::vector<int*>* GpuPolynomialChecker::findHits(
            const float needle,
            const float theConst,
            const int degree,
            const float *coeffArray,
            const std::vector<int> *loopRanges
)
{
    // TODO: This sucks. Change this
    // note that even elements are LUT[0] through LUT[5]
	// TODO 8/16/24 Make this {6, 1'216'772, 6, 304'468, 6, 12'180, 6, 4'412, 6, 1'116, 6, 292}; instead!!!!
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

    std::vector<int*> *hits = new std::vector<int*>();

    // note that these loops use <= (less than or EQUAL TO)
    for (int z = loopStartEnds[0]; z <= loopStartEnds[1]; z++) {
		v0 = coeffArray[z];
	
		for (int y = loopStartEnds[2]; y <= loopStartEnds[3]; y++) {
			v1 = v0 + coeffArray[y] * theConst;

			for (int x = loopStartEnds[4]; x <= loopStartEnds[5]; x++) {
				v2 = v1 + coeffArray[x] * theConst2;

				for (int w = loopStartEnds[6]; w <= loopStartEnds[7]; w++) {
					printf("dog\n");
					v3 = v2 + coeffArray[w] * theConst3;
					//TODO: Shouldn't overwrite this every time. Also need to take the 2 results returned
					// and make a new "hit" with all 6 coeff indices to return
                    hits = testForZeta5OnGPU(theConst, v3, coeffArray, loopStartEnds[9], loopStartEnds[11]);
                }
            }
        }
    }

    return hits;
}