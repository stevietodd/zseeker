#ifndef POSITIVE_ONLY_SKIP_HPP
#define POSITIVE_ONLY_SKIP_HPP

// Host-side counterpart of the positive-only GPU kernel skips:
//   if (quint == 0 && quart < 0) return;
//   if (quint == 0 && quart == 0 && cubic < 0) return;
// loopStartEnds is [quint, quart, cubic, x, y, z] start/end pairs.
inline bool positiveOnlyKernelHasNoWork(const int loopStartEnds[12]) {
    const int quintStart = loopStartEnds[0];
    const int quintEnd = loopStartEnds[1];
    const int quartStart = loopStartEnds[2];
    const int quartEnd = loopStartEnds[3];
    const int cubicStart = loopStartEnds[4];
    const int cubicEnd = loopStartEnds[5];
    const int xStart = loopStartEnds[6];
    const int xEnd = loopStartEnds[7];
    const int yStart = loopStartEnds[8];
    const int yEnd = loopStartEnds[9];
    const int zStart = loopStartEnds[10];
    const int zEnd = loopStartEnds[11];

    if (quintStart > quintEnd || quartStart > quartEnd || cubicStart > cubicEnd
        || xStart > xEnd || yStart > yEnd || zStart > zEnd) {
        return true;
    }

    // Any quint other than 0 is searched in full.
    if (quintStart != 0 || quintEnd != 0) {
        return false;
    }

    if (quartEnd > 0) {
        return false;
    }
    if (quartEnd < 0) {
        return true;
    }
    // quart range includes 0 and no positive quarts; remaining work is cubic >= 0.
    return cubicEnd < 0;
}

#endif
