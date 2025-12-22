#ifndef LOOKUP_TABLE_ACCESSOR_HPP
#define LOOKUP_TABLE_ACCESSOR_HPP

#include <cstddef>

const float* getLookupTableFloat();
const double* getLookupTableDouble();
std::size_t getLookupTableSize();

#endif // LOOKUP_TABLE_ACCESSOR_HPP

