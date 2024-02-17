FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get -y install cmake

WORKDIR /app

COPY . .

WORKDIR /app/build

RUN cmake -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build . --parallel 8

CMD ["./zseeker"]