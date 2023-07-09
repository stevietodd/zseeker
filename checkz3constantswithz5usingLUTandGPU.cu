// #include <iostream>
// #include <sstream>
// #include <cmath>
// #include <ctime>
// #include <numeric>
// #include <limits>
// using namespace std;

// char* getCurrentTimeString() {
// 	std::time_t currTime = std::time(nullptr);
// 	return std::asctime(std::localtime(&currTime));
// }

// #include <array>
// using ResultT = float;
// constexpr ResultT f(int i, int j)
// {
//     return ((float)i / j);
// }

// cuda.cu
#include "cudawrapper.hpp"
#include <iostream>
using namespace std;

// calculates val * [vector elems] and returns if close enough to needle
template<typename T>
__global__ static void compareToZeta5(T *out, const T val, const T needle, const T *coeffArray, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    T expr;
	//printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);

    // Handling arbitrary vector size
    if (tid < n){
		expr = (val * coeffArray[tid]) - needle;
		if (expr < .0000001 && expr > -.0000001) {
			printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
			out[tid] = coeffArray[tid]; //TODO: MAKE THIS ATOMIC AND DYNAMIC instead of only populating "matching" coeffs in array [0, 0, HIT, 0, 0, 0, HIT, etc.]
		}
    }
}

bool InitCUDA(bool b) {
    /* CUDA Initialization */
}

// will calculate ax^5 + bx^4 + cx^3 + dx^2 + ex + f - zeta(5) and return x if
// the value is within a "close enough" bound to zero
// cons = x
// cubicSum = cx^3 + dx^2 + ex + f (pre-calculated)
// coeffArray = array of all possible a/b/c/d/e/f coeff values
// quartLastIndex = last index to loop through for b values
// quintLastIndex = last index to loop through for a values
template<typename T>
std::vector<T>* testForZeta5OnGPU(T cons, T cubicSum, const T *coeffArray, int quartLastIndex, int quintLastIndex)
{
	const T z5 = 1.036927755143369926331365486457034168L; //riemann_zetal((long double)5);
	T currentQuart;
	T quarticSum;
	T consFifth = pow(cons, (T)5);

	T *d_coeffArray, *d_out;
	T *out = new T[quintLastIndex];

	// Allocate device memory 
    cudaMalloc((void**)&d_coeffArray, sizeof(T) * quintLastIndex);
    cudaMalloc((void**)&d_out, sizeof(T) * quintLastIndex);		//TODO: Can probably make this 10% as large as quintLastIndex??

	cout << "dog\n";

	// Transfer data from host to device memory
    cudaMemcpy(d_coeffArray, coeffArray, sizeof(T) * quintLastIndex, cudaMemcpyHostToDevice);

	cout << "cat\n";

	int block_size = 1024;
    int grid_size = ((quintLastIndex + block_size) / block_size);
	cout << grid_size << "\n";

	// loop through quarts
	for (int i = 6; i <= 7; i++) { // TODO: Set limit back to quartIndex!
		currentQuart = coeffArray[i];

		quarticSum = cubicSum + currentQuart * pow(cons, (T)4);

		// now we want gpu to calculate ax^5 + quarticSum - zeta(5) and return if close enough to zero
		// or ax^5 + quarticSum == zeta(5) => zeta(5) - quarticSum is close enough to ax^5
		// in other words, figure out what blah - zeta(5) and cons^5 and send those to GPU along with coeff array
		
		// Executing kernel
		compareToZeta5<<<grid_size,block_size>>>(d_out, consFifth, (z5 - quarticSum), coeffArray, quintLastIndex);

		// // Transfer data back to host memory
		cudaMemcpy(out, d_out, sizeof(T) * quintLastIndex, cudaMemcpyDeviceToHost);
		cout << "ferret\n";
		// //cudaDeviceSynchronize();

		for (int j = 0; j < quintLastIndex; j++) {
			if (out[j] < .0000001 && out[j] > .0000001) {
				cout << "i=" << i << ", out[" << j << "] = " << out[j] << "\n";
			}
		}

		//cout << i << "\n";
	}

    
	// cout << "bird\n";

    // Deallocate device memory
    cudaFree(d_coeffArray);
    cudaFree(d_out);

	cout << "lizard\n";

	delete out;
}

template std::vector<int>* testForZeta5OnGPU<int>(int cons, int cubicSum, const int *coeffArray, int quartLastIndex, int quintLastIndex);
template std::vector<float>* testForZeta5OnGPU<float>(float cons, float cubicSum, const float *coeffArray, int quartLastIndex, int quintLastIndex);
template std::vector<double>* testForZeta5OnGPU<double>(double cons, double cubicSum, const double *coeffArray, int quartLastIndex, int quintLastIndex);