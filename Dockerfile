# zseeker2 is compiled for compute capability 7.5 (CMakeLists.txt).
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        g++ \
        libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /build --target zseeker2 --parallel

FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# libquadmath0 exists only on amd64. aarch64 long double is already binary128.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmysqlclient21 \
    && if [ "$(dpkg --print-architecture)" = "amd64" ]; then \
         apt-get install -y --no-install-recommends libquadmath0; \
       fi \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/zseeker2 /usr/local/bin/zseeker2

ENTRYPOINT ["zseeker2"]
CMD ["gfpo"]
