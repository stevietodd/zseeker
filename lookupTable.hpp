#ifndef LOOKUPTABLE_HPP
#define LOOKUPTABLE_HPP

#include <array>
#include <numeric>
#include <vector>
//using ResultT = float;
constexpr float f(int i, int j)
{
    return ((float)i / j);
}

inline constexpr auto LUT = []
{
    constexpr auto LUT_LoopSize = 1000;
    //std::array<ResultT, (LUT_LoopSize*LUT_LoopSize*2)> arr = {};
	//ResultT arr[1'216'773] {0};					//TODO THIS IS WHERE I LEFT OFF 6/6/23
	//std::array<float, (1'216'773)> arr = {};
	std::array<float, (608'384)> arr = {};

	// 8/21/25 I am refactoring the LUT to no longer have the 6 cutoffs at the front nor have negative values.
	
	// I'm having trouble figuring out how to save off the cutoff15, cutoff30, etc.
	// variables separately so I'm hacking this up and making the resulting lookup table
	// start with the 6 cutoff values I'm looking for. Hence why pos starts at 6. In other words:
	// LUT[0] = cutoff15 = 143 now, was 149 (so 150) only positives, all 292 (so array size needs to be 293)
	// LUT[1] = cutoff30 = 555 now, was 561 (so 562) only positives, all 1,116 (so 1,117)
	// LUT[2] = cutoff60 = 2,203 now, was 2,209 (so 2,210) only positives, all 4,412 (so 4,413)
	// LUT[3] = cutoff100 = 6,087 now, was 6,093 (so 6,094) only positives, all 12,180 (so 12,181)
	// LUT[4] = cutoff500 = 152.231 now, was 152,237 (so 152,238) only positives, all 304,468 (so 304,469)
	// LUT[5] = cutoff1000 = actualSize = 608,383 now, was 608,389 (so 608,390) only positives, all 1,216,772 (so 1,216,773)
	// LUT[6] = 0;
	// LUT[7] = 1;
	// LUT[8] = -1;
	// ...and so on...
	int pos = 0;

	// store zero separately
	arr[pos++] = f(0,1);

	// store 1 and -1 separately
	arr[pos++] = f(1,1);
	//arr[pos++] = f(-1,1);

    for (int i = 2; i <= LUT_LoopSize; i++)
    {
		for (int j = 1; j < i; j++)
		{
			if (std::gcd(i,j) > 1) {
				continue;
			}

			arr[pos++] = f(j,i);
			//arr[pos++] = f(-j,i);
			arr[pos++] = f(i,j);
			//arr[pos++] = f(-i,j);
		}
        
		// note we subtract 1 from pos because we've already incremented it
		// if (i == 15) {
		// 	arr[0] = pos - 1; // cutoff15
		// } else if (i == 30) {
		// 	arr[1] = pos - 1; // cutoff30
		// } else if (i == 60) {
		// 	arr[2] = pos - 1; // cutoff60
		// } else if (i == 100) {
		// 	arr[3] = pos - 1; // cutoff100
		// } else if (i == 500) {
		// 	arr[4] = pos - 1; // cutoff500
		// } else if (i == 1000) {
		// 	arr[5] = pos - 1; // cutoff1000
		// }
			
    }

    return arr;
}();

#endif