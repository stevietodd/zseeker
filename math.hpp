#ifndef MATH_HPP
#define MATH_HPP

#include <cmath>

#define DOUBLE_POS_ERROR_DEFAULT .00000001 // approximate precision in digits = EIGHT
#define DOUBLE_NEG_ERROR_DEFAULT -DOUBLE_POS_ERROR_DEFAULT
#define DOUBLE_BASICALLY_EQUAL_DEFAULT(d1,d2) ((d1 - d2) >= DOUBLE_NEG_ERROR_DEFAULT && (d1 - d2) <= DOUBLE_POS_ERROR_DEFAULT)
#define DOUBLE_BASICALLY_EQUAL(d1,d2,tolerance) ((d1 - d2) >= -tolerance && (d1 - d2) <= tolerance)

// Possible TODO: Original checkz3constantswithz5usingLUTandCPU.cpp used .0000001, not .00001
#define FLOAT_POS_ERROR_DEFAULT .00001 // approximate precision in digits = FIVE
#define FLOAT_NEG_ERROR_DEFAULT -FLOAT_POS_ERROR_DEFAULT
#define FLOAT_BASICALLY_EQUAL_DEFAULT(f1,f2) ((f1 - f2) >= FLOAT_NEG_ERROR_DEFAULT && (f1 - f2) <= FLOAT_POS_ERROR_DEFAULT)
#define FLOAT_BASICALLY_EQUAL(f1,f2,tolerance) ((f1 - f2) >= -tolerance && (f1 - f2) <= tolerance)

#define ZETA2 1.64493406684822643647
#define ZETA4 1.08232323371113819152
#define ZETA5 1.03692775514336992633 // 1.036927755143369926331365486457034168L

#define USE_DEFAULT 1'000'000 // this is kind of a hack. Only works because the number of total coeffs currently is 608,384

// Every roots_checked_slice tile is this many quint indices by this many quart indices.
// The last quart strip of each quint may be shorter when the LUT span is not a multiple of 100.
#define SLICE_QUINT_CHUNK 1
#define SLICE_QUART_CHUNK 100

// Full LUT index bounds used when tiling roots_checked_slice (matches populate_slices_for_cubic_root).
#define SLICE_QUINT_MIN -608383
#define SLICE_QUINT_MAX 608383
#define SLICE_QUART_MIN -152231
#define SLICE_QUART_MAX 152231

// Unstarted tiles to insert at a time while this worker drains a cubic root.
#define SLICE_ENQUEUE_BATCH 8

// logic taken from https://blog.demofox.org/2017/11/21/floating-point-precision/
static inline double getDoublePrecisionBasedOnMaxValue(const double maxValue) {
    // note we subtract 51 instead of 52 because I am reducing the precision allowed "to be safe"
    // also note that for maxValue == 0 we use the precision for 1 to avoid returning 0 ourselves
    return pow(2, (floor(log2(abs(maxValue ?: 1))) - 51));
}

// logic taken from https://blog.demofox.org/2017/11/21/floating-point-precision/
static inline float getFloatPrecisionBasedOnMaxValue(const float maxValue) {
    // note we subtract 22 instead of 23 because I am reducing the precision allowed "to be safe"
    // also note that for maxValue == 0 we use the precision for 1 to avoid returning 0 ourselves
    return pow(2, (floor(log2(abs(maxValue ?: 1))) - 22));
	//return pow(2, (floor(log2(abs(maxValue ?: 1))) - 23));  //TODO: MegaMan explore speedup with this!
}

#endif // MATH_HPP