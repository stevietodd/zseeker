#ifndef LOOKUP_TABLE_ACCESSOR_HPP
#define LOOKUP_TABLE_ACCESSOR_HPP

#include <cstddef>
#include <cstdint>

struct LutRational {
    int16_t num;
    int16_t den;
};

const float* getLookupTableFloat();
const double* getLookupTableDouble();
const LutRational* getLookupTableRational();
std::size_t getLookupTableSize();

#endif // LOOKUP_TABLE_ACCESSOR_HPP

