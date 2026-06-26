#include "lookupTableAccessor.hpp"
#include "lookupTable.hpp"

const float* getLookupTableFloat()
{
    return LUT.data();
}

const double* getLookupTableDouble()
{
    return doubleLUT.data();
}

const LutRational* getLookupTableRational()
{
    return rationalLUT.data();
}

std::size_t getLookupTableSize()
{
    return LUT.size();
}

