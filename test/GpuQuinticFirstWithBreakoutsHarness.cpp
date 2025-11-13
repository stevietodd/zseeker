#include "GpuPolynomialChecker.hpp"
#include "lookupTable.hpp"
#include "math.hpp"

#include <iostream>
#include <vector>

int main()
{
    GpuQuinticFirstWithBreakoutsChecker checker;

    long floatHitCount = 0;

    std::vector<int> loopRanges = {
        -128, 128,
        -32, 32,
        -16, 16,
        -8, 8,
        -4, 4,
        -2, 2
    };

    auto* hits = checker.findHits(
        ZETA5,
        1.0,
        5,
        LUT.data(),
        &loopRanges,
        floatHitCount);

    std::cout << "GPU hits: " << hits->size() << "\n";
    std::cout << "Float hit count: " << floatHitCount << "\n";

    for (auto* hit : *hits)
    {
        std::cout << "hit: "
                  << hit[0] << ", "
                  << hit[1] << ", "
                  << hit[2] << ", "
                  << hit[3] << ", "
                  << hit[4] << ", "
                  << hit[5] << "\n";
        delete[] hit;
    }
    delete hits;

    return 0;
}

