# zseeker

zseeker attempts to find a constant c such that

(A/B)*c^3 + (C/D)*c^2 + (E/F)*c + (G/H) = zeta(3)
(I/J)*c^5 + (K/L)*c^4 + (M/N)*c^3 + (O/P)*c^2 + (Q/R)*c + (S/T) = zeta(5)

for reasonably selected integers A through T. These integers are loosely bounded by the coefficients that define the closed terms for zeta(4) and zeta(6).

Coefficients come from a lookup table of small rationals divided by each other and filtered to remove duplicates. Checkers scan combinations in three stages: a float pass to throw out obvious misses, a double confirmation, then a `__float128` (libquadmath) re-evaluation using the LUT numerators and denominators.

`findHits` reports the double-verified count in `doubleHitCount`. After return, `hits->size()` is the float128-refined count. Those two numbers are stored as `double_hit_count*` and `float128_hit_count*` on `roots_checked` (and on slice rows when slicing is enabled).

## Build

Needs CMake, CUDA, MySQL client headers (`libmysqlclient-dev` or `mysql-client`), and libquadmath.

```sh
mkdir -p build && cd build
cmake ..
cmake --build . --target zseeker2
```

The worker binary is `zseeker2`.

## Run

```sh
export DB_HOST=...
export DB_USER=...
export DB_PASSWORD=...
./zseeker2 gfpo
```

Checker argument (default is CPU quintic-first with breakouts):

| Arg | Checker |
| --- | --- |
| `gf` | GPU quintic-first |
| `gfpo` | GPU quintic-first, positive-only |
| `gl` | GPU quintic-last |
| `cf` | CPU quintic-first |
| `cfwb` | CPU quintic-first with breakouts |
| `cl` | CPU quintic-last |

Optional environment:

- `ZSEEKER_USE_ROOT_SLICES=1` — pull work from `roots_checked_slice` instead of whole-root rows
- `ZSEEKER_REFINE_TOL` — float128 tolerance (default `1e-12`)
