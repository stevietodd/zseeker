FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 as build

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get -y install cmake

WORKDIR /app

COPY . .

WORKDIR /app/build

RUN cmake -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build . --parallel 8

FROM ubuntu
#COPY --from=build /app/build/PrecisionTestSuite /app/build/PrecisionTestSuite
COPY --from=build /app/build/ArithmeticTestSuite /app/build/ArithmeticTestSuite
#CMD ["sleep", "infinity"]
#CMD ["app/build/PrecisionTestSuite"]
CMD ["app/build/ArithmeticTestSuite"]
#CMD ["./zseeker"]

# docker run -it --entrypoint /bin/sh [image]