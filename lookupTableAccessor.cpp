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

std::size_t getLookupTableSize()
{
    return LUT.size();
}


