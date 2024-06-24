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
	std::array<float, (1'216'773)> arr = {};
	
	// I'm having trouble figuring out how to save off the cutoff15, cutoff30, etc.
	// variables separately so I'm hacking this up and making the resulting lookup table
	// start with the 6 cutoff values I'm looking for. Hence why pos starts at 6. In other words:
	// LUT[0] = cutoff15 = 292 (so array size needs to be 293)
	// LUT[1] = cutoff30 = 1,116 (so 1,117)
	// LUT[2] = cutoff60 = 4,412 (so 4,413)
	// LUT[3] = cutoff100 = 12,180 (so 12,181)
	// LUT[4] = cutoff500 = 304,468 (so 304,469)
	// LUT[5] = cutoff1000 = actualSize = 1,216,772 (so 1,216,773)
	// LUT[6] = 0;
	// LUT[7] = 1;
	// LUT[8] = -1;
	// ...and so on...
	int pos = 6;

	// store zero separately
	arr[pos++] = f(0,1);

	// store 1 and -1 separately
	arr[pos++] = f(1,1);
	arr[pos++] = f(-1,1);

    for (int i = 2; i <= LUT_LoopSize; i++)
    {
		for (int j = 1; j < i; j++)
		{
			if (std::gcd(i,j) > 1) {
				continue;
			}

			arr[pos++] = f(j,i);
			arr[pos++] = f(-j,i);
			arr[pos++] = f(i,j);
			arr[pos++] = f(-i,j);
		}
        
		// note we subtract 1 from pos because we've already incremented it
		if (i == 15) {
			arr[0] = pos - 1; // cutoff15
		} else if (i == 30) {
			arr[1] = pos - 1; // cutoff30
		} else if (i == 60) {
			arr[2] = pos - 1; // cutoff60
		} else if (i == 100) {
			arr[3] = pos - 1; // cutoff100
		} else if (i == 500) {
			arr[4] = pos - 1; // cutoff500
		} else if (i == 1000) {
			arr[5] = pos - 1; // cutoff1000
		}
			
    }

    return arr;
}();

#endif